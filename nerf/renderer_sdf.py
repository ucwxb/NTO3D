import trimesh

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def sample_pdf(bins, weights, n_samples, det=False):
    """
    Sample points from a probability density function (PDF) using inverse CDF method.
    Implementation based on NeRF paper.
    
    Args:
        bins: [B, T], old_z_vals
        weights: [B, T - 1], bin weights
        n_samples: number of samples to generate
        det: whether to use deterministic sampling
    
    Returns:
        [B, n_samples], new_z_vals
    """
    # Prevent nans
    weights = weights + 1e-5
    
    # Get pdf and cdf
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def near_far_from_bound(rays_o, rays_d, bound, type='cube'):
    """
    Calculate near and far bounds for ray sampling based on scene bounds.
    
    Args:
        rays_o: [B, N, 3], ray origins
        rays_d: [B, N, 3], ray directions
        bound: int, radius for ball or half-edge-length for cube
        type: 'sphere' or 'cube', type of bounding volume
    
    Returns:
        near: [B, N, 1], near bounds
        far: [B, N, 1], far bounds
    """
    radius = rays_o.norm(dim=-1, keepdim=True)

    if type == 'sphere':
        near = radius - bound
        far = radius + bound
    elif type == 'cube':
        # Calculate intersection points with cube bounds
        tmin = (-bound - rays_o) / (rays_d + 1e-15)
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        near = torch.clamp(near, min=0.05)

    return near, far


def plot_pointcloud(pc, color=None):
    """
    Visualize point cloud using trimesh.
    
    Args:
        pc: [N, 3], point cloud coordinates
        color: [N, 3/4], point colors
    """
    pc = trimesh.PointCloud(pc, color)
    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()

class NeRFRenderer(nn.Module):
    """
    Base class for NeRF renderer with SDF support.
    Implements core rendering functionality and utilities.
    """
    def __init__(self,
                 cuda_ray=False,
                 curvature_loss = False
                 ):
        super().__init__()

        # extra state for cuda raymarching
        self.cuda_ray = cuda_ray
        self.curvature_loss = curvature_loss
        if cuda_ray:
            # density grid
            density_grid = torch.zeros([128 + 1] * 3) # +1 because we save values at grid
            self.register_buffer('density_grid', density_grid)
            self.mean_density = 0
            self.iter_density = 0
            # step counter
            step_counter = torch.zeros(64, 2, dtype=torch.int32) # 64 is hardcoded for averaging...
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0
    
    def forward(self, x, d, bound):
        raise NotImplementedError()
    
    def forward_color(self, x, d, n,geo_feat, bound):
        raise NotImplementedError()
    
    def forward_sdf(self, x, bound):
        raise NotImplementedError()

    def forward_instance(self, x, bound):
        raise NotImplementedError()

    def forward_feat(self, x, color, depth, normal_map, normal, color_feat, feature_vector, weights, bound):
        raise NotImplementedError()
    
    # def forward_instance(self, color_feat):
    #     raise NotImplementedError()

    def finite_difference_normals_approximator(self, x, bound, epsilon):
        raise NotImplementedError()
    
    def forward_variance(self):
        raise NotImplementedError()
    
    def gradient(self, x, bound, epsilon = 0.0005):
        raise NotImplementedError()

    def density(self, x, bound):
        raise NotImplementedError()

    def run(self, rays_o, rays_d, num_steps, bound, upsample_steps, bg_color, cos_anneal_ratio = 1.0, normal_epsilon_ratio = 1.0, staged=False):
        """
        Core rendering function that implements the NeuS rendering algorithm.
        
        Args:
            rays_o: [B, N, 3], ray origins
            rays_d: [B, N, 3], ray directions
            num_steps: number of sampling steps
            bound: scene bound
            upsample_steps: number of upsampling steps
            bg_color: background color
            cos_anneal_ratio: annealing ratio for cosine term
            normal_epsilon_ratio: epsilon ratio for normal computation
            staged: whether to use staged rendering
            
        Returns:
            depth: [B, N], rendered depth
            image: [B, N, 3], rendered RGB
            normal_map: [B, N, 3], rendered normal map
            instance_map_ray_wise: [B, N], instance segmentation
            gradient_error: scalar, gradient error for Eikonal loss
        """
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        B, N = rays_o.shape[:2]
        device = rays_o.device

        rays_o = rays_o.reshape(-1,3)
        rays_d = rays_d.reshape(-1,3)

        # sample steps
        near, far = near_far_from_bound(rays_o, rays_d, bound, type='cube')
        # near, far = near_far_from_bound(rays_o, rays_d, bound, type='sphere')
     
        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)# [1, T]
        z_vals = z_vals.expand((N, num_steps)) # [N, T]
        z_vals = near + (far - near) * z_vals # [N, T], in [near, far]

        # perturb z_vals
        sample_dist = (far - near) / num_steps
        if self.training:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist

        # generate pts
        pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 3] -> [N, T, 3]
        pts = pts.clamp(-bound, bound) # must be strictly inside the bounds, else lead to nan in hashgrid encoder!

        if upsample_steps > 0:
            with torch.no_grad():
                # query SDF and RGB
                sdf_nn_output = self.forward_sdf(pts.reshape(-1, 3), bound)
                sdf = sdf_nn_output[:, :1]
                sdf = sdf.reshape(N, num_steps) # [N, T]
                
                for i in range(upsample_steps // 16):
                    new_z_vals = self.up_sample(rays_o, rays_d, z_vals, sdf, 16, 64 * 2 **i)
                    z_vals, sdf = self.cat_z_vals(rays_o, rays_d, z_vals, new_z_vals, sdf, bound, last=(i + 1 == upsample_steps // 16))
                    

            num_steps += upsample_steps

        ### render core
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # [N, T-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[:, :1])], dim=-1)

        # sample pts on new z_vals
        z_vals_mid = (z_vals[:, :-1] + 0.5 * deltas[:, :-1]) # [N, T-1]
        z_vals_mid = torch.cat([z_vals_mid, z_vals[:,-1:]], dim=-1)

        new_pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals_mid.unsqueeze(-1) # [N, 1, 3] * [N, t, 3] -> [N, t, 3]
        new_pts = new_pts.clamp(-bound, bound)

        # only forward new points to save computation
        new_dirs = rays_d.unsqueeze(-2).expand_as(new_pts)

        sdf_nn_output = self.forward_sdf(new_pts.reshape(-1, 3), bound)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradient = self.gradient(new_pts.reshape(-1, 3), bound, 0.005 * (1.0 - normal_epsilon_ratio)).squeeze()
        # gradient = self.gradient(new_pts.reshape(-1, 3), bound, 0.001).squeeze()
        normal =  gradient / (1e-5 + torch.linalg.norm(gradient, ord=2, dim=-1,  keepdim = True))

        # color = self.forward_color(new_pts.reshape(-1, 3), new_dirs.reshape(-1, 3), normal.reshape(-1, 3), feature_vector, bound)
        color = self.forward_color(new_pts.reshape(-1, 3), new_dirs.reshape(-1, 3), normal.reshape(-1, 3), feature_vector, bound)
        instance_map_point_wise = self.forward_instance(new_pts.reshape(-1, 3), bound)
        # instance_map = self.forward_instance(color_feat)

        inv_s = self.forward_variance()     # Single parameter
        inv_s = inv_s.expand(N * num_steps, 1)

        true_cos = (new_dirs.reshape(-1, 3) * normal).sum(-1, keepdim=True)
    
        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        # version relu
        # iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
        #             F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive
        
        # version Softplus
        activation = nn.Softplus(beta=100)
        iter_cos = -(activation(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                    activation(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * deltas.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * deltas.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        # Equation 13 in NeuS
        alpha = ((prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)).reshape(N, num_steps).clip(0.0, 1.0)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([N, 1],device=alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        weights_sum = weights.sum(dim=-1, keepdim=True)

        # calculate instance
        # instance_map = self.forward_instance_feature(new_pts.reshape(-1, 3), weights)

        instance_map_point_wise = instance_map_point_wise.reshape(N, num_steps)
        weights_for_instance = weights.detach()
        instance_map_point_wise = instance_map_point_wise * weights_for_instance
        # instance_map_point_wise = instance_map_point_wise
        # instance_map_point_wise = weights_for_instance
        instance_map_ray_wise = instance_map_point_wise.max(dim=-1)[0]
        if staged:
            self.foreground_threshold = 0.90
            self.point_wise_foreground_threshold = 0.01
            self.foreground_pool_kernel_size = 21
            temporal_mask = torch.sigmoid(instance_map_ray_wise) > self.foreground_threshold
            smoothed_instance_prediction = F.max_pool1d(instance_map_point_wise.unsqueeze(dim=1),\
                                                        kernel_size=self.foreground_pool_kernel_size, stride=1,
                                                        padding=(self.foreground_pool_kernel_size-1)//2).squeeze()
            point_wise_temporal_mask = (torch.sigmoid(smoothed_instance_prediction) > self.point_wise_foreground_threshold) & \
                            (temporal_mask.unsqueeze(dim=1).expand(-1, num_steps))
            instance_map_ray_wise = point_wise_temporal_mask.any(dim=1)

        # calculate color 
        color = color.reshape(N, num_steps, 3) # [N, T, 3]
        image = (color * weights[:, :, None]).sum(dim=1)
        # calculate normal 
        normal_map = normal.reshape(N, num_steps, 3) # [N, T, 3]
        normal_map = torch.sum(normal_map * weights[:, :, None], dim=1)
        
        # calculate depth 
        ori_z_vals = ((z_vals - near) / (far - near)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)

        # TODO:Eikonal loss 
        pts_norm = torch.linalg.norm(new_pts.reshape(-1, 3), ord=2, dim=-1, keepdim=True).reshape(N, num_steps)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        gradient_error = (torch.linalg.norm(gradient.reshape(N, num_steps, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)
        # gradient_error = ((torch.linalg.norm(gradient, ord=2, dim=-1) - 1.)**2).mean()
        if not (gradient == gradient).all():
            breakpoint()
        assert (gradient == gradient).all(), 'Nan or Inf found!'

        # mix background color
        if bg_color is None:
            bg_color = 1
    
        image = image + (1 - weights_sum) * bg_color

        depth = depth.reshape(B, N)
        image = image.reshape(B, N, 3)
        normal_map = normal_map.reshape(B, N, 3)
        instance_map_ray_wise = instance_map_ray_wise.reshape(B, N)

        return depth, image, normal_map, instance_map_ray_wise, gradient_error

    def render(self, rays_o, rays_d, num_steps, bound, upsample_steps, staged=False, max_ray_batch=4096, bg_color=None, cos_anneal_ratio = 1.0, normal_epsilon_ratio = 1.0, **kwargs):
        """
        Main rendering function that handles batched rendering.
        
        Args:
            rays_o: [B, N, 3], ray origins
            rays_d: [B, N, 3], ray directions
            num_steps: number of sampling steps
            bound: scene bound
            upsample_steps: number of upsampling steps
            staged: whether to use staged rendering
            max_ray_batch: maximum batch size for ray processing
            bg_color: background color
            cos_anneal_ratio: annealing ratio for cosine term
            normal_epsilon_ratio: epsilon ratio for normal computation
            
        Returns:
            Dictionary containing rendered results
        """
        _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device

        gradient_error = 0.0
        if staged:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)
            normal = torch.empty((B, N, 3), device=device)
            instance_map_ray_wise = torch.empty((B, N), device=device)
            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)

                    depth_, image_, normal_, instance_map_ray_wise_ ,gradient_error_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], num_steps, bound, upsample_steps, bg_color, 
                                                                cos_anneal_ratio = cos_anneal_ratio, normal_epsilon_ratio = normal_epsilon_ratio, staged=staged)
                    
                    depth[b:b+1, head:tail] = depth_.detach()
                    image[b:b+1, head:tail] = image_.detach()
                    normal[b:b+1, head:tail] = normal_.detach()
                    gradient_error_ = gradient_error_.detach()
                    instance_map_ray_wise[b:b+1, head:tail] = instance_map_ray_wise_.detach()
                    head += max_ray_batch
                    del depth_, image_, normal_, gradient_error_
        else:
            depth, image, normal, instance_map_ray_wise, gradient_error = _run(rays_o, rays_d, num_steps, bound, upsample_steps, bg_color, cos_anneal_ratio, normal_epsilon_ratio, staged=staged)
        results = {}
        results['depth'] = depth
        results['rgb'] = image
        results['instance_map_ray_wise'] = instance_map_ray_wise
        results['normal'] = normal
        results['gradient_error'] = gradient_error
        return results

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Upsample points based on SDF values.
        
        Args:
            rays_o: ray origins
            rays_d: ray directions
            z_vals: current z values
            sdf: SDF values
            n_importance: number of importance samples
            inv_s: inverse variance parameter
            
        Returns:
            New z values for upsampled points
        """
        batch_size, n_samples = z_vals.shape

        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)

        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=sdf.device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1], device=alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()

        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, bound, last=False):
        """
        Concatenate and sort z values from upsampling.
        
        Args:
            rays_o: ray origins
            rays_d: ray directions
            z_vals: current z values
            new_z_vals: new z values from upsampling
            sdf: current SDF values
            bound: scene bound
            last: whether this is the last upsampling step
            
        Returns:
            Concatenated and sorted z values and SDF values
        """
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        pts = pts.clamp(-bound, bound)
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)
        if not last:
            new_sdf = self.forward_sdf(pts.reshape(-1, 3), bound)[...,:1].reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf
