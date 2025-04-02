import os
import cv2
import glob
import json
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch

def nerf_matrix_to_ngp_scale(pose, aabb, bound, scale=None, offset=[0.0, 0.0, 0.0]):

    scale = max(0.000001,max(max(abs(float(aabb[1][0])-float(aabb[0][0])),
                                abs(float(aabb[1][1])-float(aabb[0][1]))),
                                abs(float(aabb[1][2])-float(aabb[0][2]))))
    scale =  2.0 * bound / scale
    offset = [((float(aabb[1][0]) + float(aabb[0][0])) * 0.5) * -scale,
                ((float(aabb[1][1]) + float(aabb[0][1])) * 0.5) * -scale, 
                ((float(aabb[1][2]) + float(aabb[0][2])) * 0.5) * -scale]
    pose[:3, 3] = pose[:3, 3] * scale + np.array(offset)
    new_pose = pose.astype(np.float32)
    # new_pose = new_pose[[1, 2, 0, 3], :]
    return new_pose

def nerf_matrix_to_ngp(pose, scale=1.0, offset=[0, 0, 0]):
    pose[:3, 3] = pose[:3, 3] * scale + np.array(offset)
    # pose = pose[[1, 2, 0, 3], :]
    new_pose = pose.astype(np.float32)
    # new_pose = new_pose[[1, 2, 0, 3], :]
    return new_pose

def visualize_poses(poses, size=0.1, bound=1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=[2*bound]*3).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.exchange.export.export_mesh(objects, file_obj='poses.obj')
    # trimesh.Scene(objects).export('poses.obj')

def load_K_Rt_from_P(P):
    
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsic = np.array(K).astype(np.float32)
    # intrinsic = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsic, pose


class NeRFDataset:
    def __init__(self, path, type='train', mode='dtu', preload=True, downscale=1, bound=0.33, n_test=10, camera_traj='else', scale=1.0):
        super().__init__()
        
        self.mode = mode
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = path
        self.preload = preload # preload data into GPU
        # bound = SCENE_TO_BOUND[os.path.basename(path)]
        # self.bound = bound # bounding box half length, also used as the radius to random sample poses.
        self.bound = bound
        self.camera_traj = camera_traj
        self.scale = scale
        self.offset = [0, 0, 0]
            
        self.training = self.type in ['train', 'all', 'trainval']

        camera_dict = np.load(os.path.join(self.root_path, 'cameras_sphere.npz'))
        image_paths = sorted(glob.glob(os.path.join(self.root_path, 'image', '*.jpg')))
        mask_paths = sorted(glob.glob(os.path.join(self.root_path, 'mask_map', '*.png')))

        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(len(image_paths))]
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(len(image_paths))]
        self.scale_mats_scale = scale_mats[0][0, 0]
        self.scale_mats_offset =  scale_mats[0][:3, 3][None]

        intrinsics = []
        poses = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsic, pose = load_K_Rt_from_P(P)
            intrinsics.append(intrinsic)
            poses.append(pose)

        aabb_scale = 1.0 #transform['aabb_scale']
        pts = []
        for pose in poses:
            pts.append(pose[:3,3]) # [4, 4]
        pts = np.stack(pts, axis=0).astype(np.float32)

        minxyz=np.min(pts, axis=0) * aabb_scale
        maxxyz=np.max(pts, axis=0) * aabb_scale

        self.aabb = [[minxyz[0], minxyz[1], minxyz[2]],
                    [maxxyz[0], maxxyz[1], maxxyz[2]]]

        new_poses = []
        for pose in poses:
            # new_poses.append(nerf_matrix_to_ngp(pose, scale=self.scale)) # [4, 4])
            new_poses.append(nerf_matrix_to_ngp_scale(pose, aabb=self.aabb, bound=self.bound)) # [4, 4])

        self.intrinsics = torch.from_numpy(np.stack(intrinsics)).float() # [N, 4]
        self.poses = np.stack(new_poses) # [N, 4, 4]

        # self.poses[:, :3, 1:3] *= -1
        # self.poses = self.poses[:, [1, 2, 0, 3], :]
        # self.poses[:, 2] *= -1

        # visualize_poses(self.poses, bound=self.bound)
        # we have to actually read an image to get H and W later.
        self.H = self.W = None
        # make split
        if self.type == 'test':
            
            poses = []

            if self.camera_traj == 'circle':

                print(f'[INFO] use circular camera traj for testing.')
                
                # circle 360 pose
                # radius = np.linalg.norm(self.poses[:, :3, 3], axis=-1).mean(0)
                radius = 0.1
                theta = np.deg2rad(80)
                for i in range(100):
                    phi = np.deg2rad(i / 100 * 360)
                    center = np.array([
                        radius * np.sin(theta) * np.sin(phi),
                        radius * np.sin(theta) * np.cos(phi),
                        radius * np.cos(theta),
                    ])
                    # look at
                    def normalize(v):
                        return v / (np.linalg.norm(v) + 1e-10)
                    forward_v = normalize(center)
                    up_v = np.array([0, 0, 1])
                    right_v = normalize(np.cross(forward_v, up_v))
                    up_v = normalize(np.cross(right_v, forward_v))
                    # make pose
                    pose = np.eye(4)
                    pose[:3, :3] = np.stack((right_v, up_v, forward_v), axis=-1)
                    pose[:3, 3] = center
                    poses.append(pose)
                
                self.poses = np.stack(poses, axis=0)
            
            # choose some random poses, and interpolate between.
            else:

                fs = np.random.choice(len(self.poses), 5, replace=False)
                pose0 = self.poses[fs[0]]
                for i in range(1, len(fs)):
                    pose1 = self.poses[fs[i]]
                    rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
                    slerp = Slerp([0, 1], rots)    
                    for i in range(n_test + 1):
                        ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                        pose = np.eye(4, dtype=np.float32)
                        pose[:3, :3] = slerp(ratio).as_matrix()
                        pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                        poses.append(pose)
                    pose0 = pose1

                self.poses = np.stack(poses, axis=0)

            # fix intrinsics for test case
            self.intrinsics = self.intrinsics[[0]].repeat(self.poses.shape[0], 1, 1)

            self.images = None
            self.H = 576
            self.W = 768

        else:
            if type == 'train':
                image_paths = image_paths[1:]
                mask_paths = mask_paths[1:]
                self.poses = self.poses[1:]
                self.intrinsics = self.intrinsics[1:]
            elif type == 'val':
                image_paths = image_paths[0:1]
                mask_paths = mask_paths[0:1]
                self.poses = self.poses[0:1]
                self.intrinsics = self.intrinsics[0:1]
            elif type == 'mesh':
                image_paths = image_paths[0:1]
                mask_paths = mask_paths[0:1]
                self.poses = self.poses[0:1]
                self.intrinsics = self.intrinsics[0:1]
            # else 'all' or 'trainval' : use all frames
        
            # read images
            self.images = []
            self.images_sam = []
            self.masks_gt = []
            self.file_index = []
            for i in tqdm.tqdm(range(len(image_paths)), desc=f'Loading {type} data'):

                f_path = image_paths[i]
                m_path = mask_paths[i]

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                image_sam = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
                image_sam = cv2.cvtColor(image_sam, cv2.COLOR_BGR2RGB)
                image_sam = image_sam[:,:,:3].astype(np.uint8)
                # if use mask, add as an alpha channel
                mask_gt = cv2.imread(m_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]
                # image = mask_gt
                # image = np.concatenate([image, mask_gt[..., :1]], axis=-1)
                
                if self.H is None or self.W is None:
                    self.H = image.shape[0] // self.downscale
                    self.W = image.shape[1] // self.downscale

                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                image = image.astype(np.float32) / 255
                mask_gt = (mask_gt[...,0].astype(np.float32) / 255).astype(np.uint8)
                self.file_index.append(str(os.path.basename(f_path).split(".")[0]))
                self.images_sam.append(image_sam)
                self.images.append(image)
                self.masks_gt.append(mask_gt)

            self.images = np.stack(self.images, axis=0)
            self.images_sam = np.stack(self.images_sam, axis=0)
            self.masks_gt = np.stack(self.masks_gt, axis=0)

        self.poses = torch.from_numpy(self.poses.astype(np.float32)) # [N, 4, 4]
        if self.preload and self.type != 'test':
            self.images = torch.from_numpy(self.images).cuda()
            self.images_sam = torch.from_numpy(self.images_sam).cuda()
            self.masks_gt = torch.from_numpy(self.masks_gt).cuda()

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):

        results = {
            'pose': self.poses[index],
            'intrinsic': self.intrinsics[index],
        }
        if self.type == 'test':
            # only string can bypass the default collate, so we don't need to call item: https://github.com/pytorch/pytorch/blob/67a275c29338a6c6cc405bf143e63d53abe600bf/torch/utils/data/_utils/collate.py#L84
            results['H'] = str(self.H)
            results['W'] = str(self.W)
            return results
        else:
            results['H'] = str(self.H)
            results['W'] = str(self.W)
            results['image'] = self.images[index]
            results['image_sam'] = self.images_sam[index]
            results['index'] = self.file_index[index]
            results['mask_gt'] = self.masks_gt[index]
            return results