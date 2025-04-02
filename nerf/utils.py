import os
import glob
import tqdm
import random
import tensorboardX
import json
import numpy as np
from sklearn.cluster import KMeans
import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lpips
from torchmetrics.functional import structural_similarity_index_measure
import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage
from segment_anything import sam_model_registry, SamPredictor
from nerf.instance import Instance

def seed_everything(seed):
    """
    Set random seed for reproducibility across different libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def lift(x, y, z, intrinsics):
    """
    Lift 2D pixel coordinates to 3D camera space using camera intrinsics.
    
    Args:
        x, y, z: [B, N], pixel coordinates and depth
        intrinsics: [B, 3, 3], camera intrinsics matrix
        
    Returns:
        [B, N, 4], homogeneous 3D points in camera space
    """
    fx = intrinsics[..., 0, 0].unsqueeze(-1)
    fy = intrinsics[..., 1, 1].unsqueeze(-1)
    cx = intrinsics[..., 0, 2].unsqueeze(-1)
    cy = intrinsics[..., 1, 2].unsqueeze(-1)
    sk = intrinsics[..., 0, 1].unsqueeze(-1)

    x_lift = (x - cx + cy * sk / fy - sk * y / fy) / fx * z
    y_lift = (y - cy) / fy * z

    # Convert to homogeneous coordinates
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z)), dim=-1)

def get_rays(c2w, intrinsics, H, W, N_rays=-1):
    """
    Generate camera rays from camera parameters.
    
    Args:
        c2w: [B, 4, 4], camera to world transformation matrix
        intrinsics: [B, 3, 3], camera intrinsics matrix
        H, W: Image height and width
        N_rays: Number of rays to sample (-1 for all pixels)
        
    Returns:
        rays_o: [B, N_rays, 3], ray origins
        rays_d: [B, N_rays, 3], ray directions
        select_inds: [B, N_rays], selected pixel indices
    """
    device = c2w.device
    rays_o = c2w[..., :3, 3] # [B, 3]
    prefix = c2w.shape[:-2]

    # Create pixel coordinate grid
    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), 
                         torch.linspace(0, H-1, H, device=device))
    i = i.t().reshape([*[1]*len(prefix), H*W]).expand([*prefix, H*W])
    j = j.t().reshape([*[1]*len(prefix), H*W]).expand([*prefix, H*W])

    # Sample random rays if N_rays > 0
    if N_rays > 0:
        N_rays = min(N_rays, H*W)
        select_hs = torch.randint(0, H, size=[N_rays], device=device)
        select_ws = torch.randint(0, W, size=[N_rays], device=device)
        select_inds = select_hs * W + select_ws
        select_inds = select_inds.expand([*prefix, N_rays])
        i = torch.gather(i, -1, select_inds)
        j = torch.gather(j, -1, select_inds)
    else:
        select_inds = torch.arange(H*W, device=device).expand([*prefix, H*W])

    # Convert pixel coordinates to 3D points
    pixel_points_cam = lift(i, j, torch.ones_like(i), intrinsics=intrinsics)
    pixel_points_cam = pixel_points_cam.transpose(-1, -2)

    # Transform points to world space
    world_coords = torch.bmm(c2w, pixel_points_cam).transpose(-1, -2)[..., :3]
    
    # Compute ray directions
    rays_d = world_coords - rays_o[..., None, :]
    rays_d = F.normalize(rays_d, dim=-1)

    rays_o = rays_o[..., None, :].expand_as(rays_d)

    return rays_o, rays_d, select_inds


def extract_fields(bound_min, bound_max, resolution, query_func):
    """
    Extract field values in a 3D volume using a query function.
    
    Args:
        bound_min: [3], minimum bounds of the volume
        bound_max: [3], maximum bounds of the volume
        resolution: Resolution of the volume in each dimension
        query_func: Function to query field values at points
        
    Returns:
        [resolution, resolution, resolution], field values in the volume
    """
    N = 256
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)
                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                val = query_func(pts).reshape(len(xs), len(ys), len(zs))
                u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val.detach().cpu().numpy()
                del val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, use_sdf = False):
    """
    Extract geometry from field values using marching cubes.
    
    Args:
        bound_min: [3], minimum bounds of the volume
        bound_max: [3], maximum bounds of the volume
        resolution: Resolution of the volume in each dimension
        threshold: Threshold for surface extraction
        query_func: Function to query field values at points
        use_sdf: Whether to use SDF values
        
    Returns:
        vertices: [N, 3], vertex positions
        triangles: [M, 3], triangle indices
    """
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    if use_sdf:
        u = - 1.0 *u
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


class Trainer(object):
    """
    Trainer class for NeRF model training with instance segmentation support.
    
    Handles training, evaluation, and testing of NeRF models with additional functionality
    for instance segmentation using SAM (Segment Anything Model).
    """
    def __init__(self, 
                    name, # name of this experiment
                    conf, # extra conf
                    model, # network 
                    criterion=None, # loss function, if None, assume inline implementation in train_step
                    optimizer=None, # optimizer
                    ema_decay=None, # if use EMA, set the decay
                    lr_scheduler=None, # scheduler
                    metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                    local_rank=0, # which GPU am I
                    device=None, # device to use, usually setting to None is OK. (auto choose device)
                    mute=False, # whether to mute all print
                    fp16=False, # amp optimize level
                    eval_interval=1, # eval once every $ epoch
                    max_keep_ckpt=2, # max num of saved ckpts in disk
                    workspace='workspace', # workspace to save logs & ckpts
                    best_mode='min', # the smaller/larger result, the better
                    use_loss_as_metric=True, # use loss as the first metirc
                    use_checkpoint="latest", # which ckpt to use at init time
                    use_tensorboardX=True, # whether to use tensorboard for logging
                    scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                    white_background = False,
                    sam_ckp_path = "",
                    anno_path = "",
                    anno_point = 0,
                    pretrain_epoch = 30,
                    update_pseudo_label_interval = 30,
                    agg_point_num = 5,
                    sam_thr = 0.0,
                    sam_feat_path = "",
                    train_with_sam_mask = False,
                    iterative = 100,
                ):
        """
        Initialize the trainer.
        
        Args:
            name: Name of the experiment
            conf: Configuration dictionary
            model: NeRF model to train
            criterion: Loss function
            optimizer: Optimizer for training
            ema_decay: Decay rate for exponential moving average
            lr_scheduler: Learning rate scheduler
            metrics: List of metrics to track
            local_rank: Local rank for distributed training
            device: Device to use for training
            mute: Whether to mute output
            fp16: Whether to use mixed precision training
            eval_interval: Interval between evaluations
            max_keep_ckpt: Maximum number of checkpoints to keep
            workspace: Directory to save logs and checkpoints
            best_mode: Whether to minimize or maximize metrics
            use_loss_as_metric: Whether to use loss as the primary metric
            use_checkpoint: Which checkpoint to load at initialization
            use_tensorboardX: Whether to use tensorboard for logging
            scheduler_update_every_step: Whether to update scheduler every step
            white_background: Whether to use white background
            sam_ckp_path: Path to SAM model checkpoint
            anno_path: Path to annotations
            anno_point: Number of annotation points
            pretrain_epoch: Number of pretraining epochs
            update_pseudo_label_interval: Interval for updating pseudo labels
            agg_point_num: Number of points to aggregate
            sam_thr: Threshold for SAM predictions
            sam_feat_path: Path to save SAM features
            train_with_sam_mask: Whether to train with SAM masks
            iterative: Number of iterative training steps
        """
        self.name = name
        self.conf = conf
        self.mute = mute
        self.metrics = metrics
        self.metrics_val = [PSNRMeter(), SSIMMeter(), LPIPSMeter()]
        self.metrics_seg = {
            "SAM": MiouMeter(type="SAM"),
            "Pred": MiouMeter(type="Pred"),
        }
        self.local_rank = local_rank
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.white_background = white_background
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.loss_instances_hash = Instance()

        # Initialize SAM model if checkpoint path is provided
        if sam_ckp_path != "":
            self.sam_ckp_path = sam_ckp_path
            self.sam_init()

        # Load annotations if path is provided
        if anno_path != "" and os.path.exists(anno_path):
            self.anno_path = anno_path
            self.annotations = json.load(open(self.anno_path, 'r'))
            self.anno_point = anno_point
            self.annotations = self.annotations["frames"]
            self.annotation_index = []
            self.train_cur_index_list = []
            for anno in self.annotations:
                if int(anno["index"]) == self.anno_point:
                    self.train_cur_index_list.append(int(anno["index"]))
                    self.annotation_index.append(int(anno["index"]))

        # Set up SAM-related parameters
        self.agg_point_num = agg_point_num
        self.sam_thr = sam_thr
        self.sam_feat_path = sam_feat_path
        os.makedirs(self.sam_feat_path, exist_ok=True)
        self.train_with_sam_mask = train_with_sam_mask

        # Set up training parameters
        self.update_pseudo_label_interval = update_pseudo_label_interval
        self.pretrain_epoch = pretrain_epoch
        self.iterative = iterative
        self.cur_stage = 0

        # Create output directories
        self.pseudo_masks_path = os.path.join(self.workspace, "pseudo_masks")
        os.makedirs(self.pseudo_masks_path, exist_ok=True)
        self.mask_output = os.path.join(self.workspace, "masks")
        os.makedirs(self.mask_output, exist_ok=True)
        self.validation_output = os.path.join(self.workspace, "validation")
        os.makedirs(self.validation_output, exist_ok=True)

        # Initialize SAM pseudo mask dictionary
        self.sam_pseudo_mask_dict = {}
        for anno in self.annotations:
            prompt = {}
            prompt["point"] = np.array(anno["point"])
            prompt["label"] = np.array(anno["label"])
            prompt["len"] = 1
            prompt["box"] = []
            prompt["mask"] = []
            self.sam_pseudo_mask_dict[int(anno["index"])] = {
                "prompt": prompt
            }

        # Move model to device
        model.to(self.device)
        self.model = model

        # Set up criterion
        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        # Set up optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)
        else:
            self.optimizer = optimizer(self.model)

        # Set up learning rate scheduler
        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        # Set up exponential moving average
        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        # Set up gradient scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # Initialize training state
        self.epoch = 1
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],
            "checkpoints": [],
            "best_result": None,
        }

        # Auto fix best mode
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # Set up workspace and logging
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth.tar"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        # Load checkpoint if specified
        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        """Clean up log file pointer when the trainer is destroyed."""
        if self.log_ptr: 
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        """
        Log messages to console and file.
        
        Args:
            *args: Arguments to log
            **kwargs: Keyword arguments to log
        """
        if self.local_rank == 0:
            if not self.mute: 
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    def sam_init(self):
        """
        Initialize the SAM (Segment Anything Model) model.
        Loads the model checkpoint and sets up the predictor.
        """
        sam_checkpoint = self.sam_ckp_path
        model_type = "vit_h"

        device = "cuda"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)

        self.predictor = SamPredictor(self.sam)

    def show_anns(self, masks, img_src, index, points=[], boxes=[], pred_masks=[]):
        """
        Visualize annotations and predictions.
        
        Args:
            masks: Segmentation masks
            img_src: Source image
            index: Image index
            points: List of points to visualize
            boxes: List of bounding boxes to visualize
            pred_masks: List of predicted masks to visualize
        """
        img_src = img_src[:,:,::-1]
        img = np.ones((img_src.shape[0], img_src.shape[1], 3))
        masks_num = masks.shape[0]

        for mask_ind in range(masks_num):
            m_y,m_x = np.where(masks[mask_ind]==False)
            n_y,n_x = np.where(masks[mask_ind]==True)
            for i in range(3):
                img[m_y,m_x,i] = 0
                img[n_y,n_x,i] = 1
        
        img = np.array(img*255,dtype=np.uint8)
        img = cv2.addWeighted(img_src,0.3,img,0.7,0)
        if len(boxes)>0:
            cv2.rectangle(img, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0, 255, 0), 2)
        if len(points)>0:
            for point in points:
                cv2.circle(img, (point[0], point[1]), 5, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(self.mask_output, 'pseudo_mask_%d_%d.png'%(index, self.epoch)), img)

    def make_pseudo_mask(self, image, index):
        """
        Generate pseudo masks using SAM model.
        
        Args:
            image: Input image
            index: Image index
        """
        prompt = self.sam_pseudo_mask_dict[index]["prompt"]
        image_sam = image.detach().cpu().numpy()
        with torch.no_grad():
            self.sam_set_image(image_sam,index)
            prompt_len = prompt["len"]
            input_boxs = prompt["box"]
            input_masks = prompt["mask"]
            input_points = prompt["point"]
            input_labels = prompt["label"]
            for ind in range(prompt_len):
                if len(input_boxs) > 0:
                    input_box = input_boxs[ind]
                else:
                    input_box = None
                if len(input_masks) > 0:
                    input_mask = input_masks[ind]
                else:
                    input_mask = None
                if len(input_points) > 0:
                    input_point = np.array(input_points)
                    input_label = np.array(input_labels)
                else:
                    input_point = None
                    input_label = None
                masks, masks_qal, masks_low = self.predictor.predict(point_coords=input_point,\
                                            point_labels=input_label,\
                                            box=input_box,\
                                            mask_input=input_mask,\
                                            return_logits=True,
                                            multimask_output=False)
                masks = masks > self.sam_thr
                masks = np.array(masks)
                np.save(os.path.join(self.pseudo_masks_path, "pseudo_mask_{}.npy".format(index)), masks)
                self.show_anns(masks, image_sam, index, \
                            points=input_point if len(input_points) > 0 else [], \
                            pred_masks=input_mask if len(input_masks) > 0 else [], \
                            boxes=input_box if len(input_boxs) > 0 else [])

    def sam_set_image(self, image_sam, index):
        """
        Set up SAM model for processing a new image.
        
        Args:
            image_sam: Input image for SAM
            index: Image index
        """
        cur_feat_path = os.path.join(self.sam_feat_path, "%06d.pt"%(index))
        if os.path.exists(cur_feat_path):
            cur_feat = torch.load(cur_feat_path)
            self.predictor.features = cur_feat["features"].to(self.device)
            self.predictor.original_size = cur_feat["original_size"]
            self.predictor.input_size = cur_feat["input_size"]
            self.predictor.is_image_set = True
        else:
            self.predictor.set_image(image_sam)
            save_dict = {
                "features": self.predictor.features,
                "original_size": self.predictor.original_size,
                "input_size": self.predictor.input_size,
            }
            torch.save(save_dict, cur_feat_path)

    def get_pseudo_mask(self, prompt, image, index):
        """
        Get pseudo mask for an image using SAM model.
        
        Args:
            prompt: Prompt information for SAM
            image: Input image
            index: Image index
            
        Returns:
            Generated pseudo mask
        """
        masks_list = torch.zeros(image.shape[:3],device=image.device,dtype=torch.long)
        masks_cls = 1
        with torch.no_grad():
            image_sam = image[0].detach().cpu().numpy()
            self.sam_set_image(image_sam,index)
            prompt_len = prompt["len"]
            input_boxs = prompt["box"]
            input_masks = prompt["mask"]
            input_points = prompt["point"]
            input_labels = prompt["label"]
            for ind in range(prompt_len):
                if len(input_boxs) > 0:
                    input_box = input_boxs[ind]
                else:
                    input_box = None
                if len(input_masks) > 0:
                    input_mask = input_masks[ind]
                else:
                    input_mask = None
                if len(input_points) > 0:
                    input_point = np.array(input_points)
                    input_label = np.array(input_labels)
                else:
                    input_point = None
                    input_label = None

                masks, scores, logits = self.predictor.predict(
                    point_coords=input_point, 
                    point_labels=input_label,
                    multimask_output=False
                )
                best_idx = 0

                # Cascaded Post-refinement-1
                masks, scores, logits = self.predictor.predict(
                            point_coords=input_point,
                            point_labels=input_label,
                            mask_input=logits[best_idx: best_idx + 1, :, :], 
                            multimask_output=True)
                best_idx = np.argmax(scores)

                if (np.sum(masks[best_idx])<self.agg_point_num):
                    return []

                # Cascaded Post-refinement-2
                y, x = np.nonzero(masks[best_idx])
                x_min = x.min()
                x_max = x.max()
                y_min = y.min()
                y_max = y.max()
                input_box = np.array([x_min, y_min, x_max, y_max])
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box=input_box[None, :],
                    mask_input=logits[best_idx: best_idx + 1, :, :], 
                    multimask_output=False)

                masks = masks > 0.5
                masks = torch.tensor(masks)
                masks_list[masks] = masks_cls
            self.show_anns(masks_list.cpu().detach().numpy(), image_sam, index, \
                        points=input_point if len(input_points) > 0 else [], \
                        pred_masks=input_mask if len(input_masks) > 0 else [], \
                        boxes=input_box if len(input_boxs) > 0 else [])
        return masks_list

    def get_pseudo_masks(self, index, image_sam):
        """
        Get pseudo masks for an image using SAM model.
        
        Args:
            index: Image index
            image_sam: Input image for SAM
            
        Returns:
            Generated pseudo mask
        """
        prompt = {
            "box": [],
            "mask": [],
            "point": [],
            "label": [],
        }
        find_flag = False
        for anno in self.annotations:
            if anno["index"] == index:
                if index in self.annotation_index:
                    prompt["point"] = np.array(anno["point"])
                    prompt["label"] = np.array(anno["label"])
                    prompt["len"] = 1
                else:
                    prompt["point"] = np.array(anno["point"])
                    prompt["label"] = np.array(anno["label"])
                    prompt["len"] = 1
                    prompt["box"] = np.array(anno["box"])
                find_flag = True
                break
        if not find_flag:
            return []
        sam_mask = self.get_pseudo_mask(prompt, image_sam, index)
        return sam_mask

    def get_pseudo_masks_from_numpy(self, index):
        """
        Load pseudo mask from numpy file.
        
        Args:
            index: Image index
            
        Returns:
            Loaded pseudo mask
        """
        mask_npy = os.path.join(self.pseudo_masks_path, "pseudo_mask_%d.npy"%(index))
        if not os.path.exists(mask_npy):
            return []
        sam_mask_npy = np.load(mask_npy)
        sam_mask = torch.from_numpy(sam_mask_npy)
        sam_mask = sam_mask.to(device="cuda",dtype=torch.long)
        return sam_mask

    def update_pseudo_masks(self, index, instance, image_sam):
        """
        Update pseudo masks using instance predictions.
        
        Args:
            index: Image index
            instance: Instance predictions
            image_sam: Input image for SAM
            
        Returns:
            Updated pseudo mask
        """
        if len(instance.shape) == 3:
            instance = instance.argmax(dim=-1)
        
        labels = instance.unique()
        pseudo_box = []
        pseudo_mask = []
        pseudo_points = []
        pseudo_labels = []

        for label in labels:
            if label == 0:
                continue
            mask = (instance == label)
            if mask.sum() <= self.agg_point_num:
                return
            y, x = torch.where(mask==True)
            y = y.detach().cpu().numpy()
            x = x.detach().cpu().numpy()
            box = [x.min(), y.min(), x.max(), y.max()]
            pseudo_box.append(box)
            for y_,x_ in zip(y,x):
                pseudo_points.append([x_,y_])
            pseudo_points = np.array(pseudo_points)
            Kmeans = KMeans(n_clusters=self.agg_point_num)
            Kmeans.fit(pseudo_points)
            pseudo_points = np.array([[int(x_),int(y_)] for x_,y_ in Kmeans.cluster_centers_])
            pseudo_labels = np.array([1 for _ in range(len(pseudo_points))])

        if index in self.annotation_index:
            return
        find_flag = False
        for anno_ind in range(len(self.annotations)):
            anno = self.annotations[anno_ind]
            if anno["index"] == index:
                self.annotations[anno_ind]["box"] = np.array(pseudo_box)
                self.annotations[anno_ind]["mask"] = np.array(pseudo_mask)
                self.annotations[anno_ind]["point"] = np.array(pseudo_points)
                self.annotations[anno_ind]["label"] = np.array(pseudo_labels)
                find_flag = True
                break
        if not find_flag:
            self.annotations.append(
                {"index":index, "box":np.array(pseudo_box), "mask":np.array(pseudo_mask), "point":np.array(pseudo_points), "label":np.array(pseudo_labels)})

        sam_mask = self.get_pseudo_masks(index, image_sam)
        if len(sam_mask) <= 0:
            return 
        sam_mask = sam_mask.detach().cpu().numpy()
        np.save(os.path.join(self.pseudo_masks_path, "pseudo_mask_{}.npy".format(index)), sam_mask)
        return sam_mask

    def train_step(self, data):
        """
        Perform one training step.
        
        Args:
            data: Training data batch
            
        Returns:
            pred_rgb: Predicted RGB values
            gt_rgb: Ground truth RGB values
            loss: Training loss
        """
        images = data["image"] # [B, H, W, 3/4]
        poses = data["pose"] # [B, 4, 4]
        intrinsics = data["intrinsic"] # [B, 3, 3]
        index = int(data["index"][0])
        sam_mask = self.get_pseudo_masks_from_numpy(index)
        
        if (self.train_with_sam_mask or self.cur_stage in [2]) and len(sam_mask) > 0:
            images = torch.concat([images, sam_mask[:,:,:,None]], dim=-1)
        
        B, H, W, C = images.shape
        # sample rays 
        rays_o, rays_d, inds = get_rays(poses, intrinsics, H, W, self.conf['num_rays'])

        if len(sam_mask) > 0:
            sam_mask = torch.gather(sam_mask.reshape(B, -1), 1, inds)
        images = torch.gather(images.reshape(B, -1, C), 1, torch.stack(C*[inds], -1)) # [B, N, 3/4]

        bg_color = torch.rand(3, device=images.device)
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, 
                                    cos_anneal_ratio = min(self.epoch / 200, 1.0), normal_epsilon_ratio= min(self.epoch / 200, 0.95), **self.conf)

        pred_rgb = outputs['rgb']
        color_loss = self.criterion(pred_rgb, gt_rgb) * self.conf["color_loss"]
        try:
            eikonal_loss =  self.conf["eikonal_loss"] * outputs['gradient_error']
        except:
            eikonal_loss = 0.0 * outputs['rgb'].mean()

        instance_loss = 0.0 * outputs['rgb'].mean()
        if self.conf["instance_loss"] > 1e-6 and self.cur_stage in [0,1] and len(sam_mask) > 0:
            instance_map_ray_wise = outputs['instance_map_ray_wise']
            instance_loss = self.conf["instance_loss"] * self.loss_instances_hash.calculate_loss(sam_mask, instance_map_ray_wise)
            self.loss_instances_hash.compute_metrics()

        loss = color_loss + eikonal_loss + instance_loss
        return pred_rgb, gt_rgb, loss

    def eval_step(self, data):
        """
        Perform one evaluation step.
        
        Args:
            data: Evaluation data batch
            
        Returns:
            pred_rgb: Predicted RGB values
            pred_normal: Predicted normal map
            pred_depth: Predicted depth map
            gt_rgb: Ground truth RGB values
            loss: Evaluation loss
            mask_gt: Ground truth mask
            pred_sam_mask: Predicted SAM mask
            pred_instance_map_ray_wise: Predicted instance map
        """
        images = data["image"] # [B, H, W, 3/4]
        B, H, W, C = images.shape
        poses = data["pose"] # [B, 4, 4]
        intrinsics = data["intrinsic"] # [B, 3, 3]
        index = int(data["index"][0])
        try:
            mask_gt = data["mask_gt"]
        except:
            sam_mask = self.get_pseudo_masks_from_numpy(index)
            if len(sam_mask) <= 0:
                mask_gt = torch.ones((B, H, W)).to(images.device)
            else:
                mask_gt = sam_mask
        # sample rays 
        images = torch.concat([images, mask_gt[:,:,:,None]], dim=-1)
        C = 4
        
        rays_o, rays_d, _ = get_rays(poses, intrinsics, H, W, -1)

        bg_color = torch.zeros(3, device=images.device) # [3]
        # eval with fixed background color
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, 
                                    cos_anneal_ratio = min(self.epoch / 100, 1.0), normal_epsilon_ratio= min((self.epoch - 50) / 100, 0.99),
                                    using_occ=False,
                                    **self.conf)

        pred_rgb = outputs['rgb'].reshape(B, H, W, -1)
        pred_rgb = pred_rgb[..., :3] * mask_gt[:,:,:,None] + bg_color * (1 - mask_gt[:,:,:,None])
        pred_instance_map_ray_wise = outputs['instance_map_ray_wise'].reshape(B, H, W)
        
        cv2.imwrite(os.path.join(self.validation_output, "instance_%d.png"%(int(data["index"][0]))), (pred_instance_map_ray_wise[0].detach().cpu().numpy() * 255).astype(np.uint8))
        pred_sam_mask = self.update_pseudo_masks(int(data["index"][0]), pred_instance_map_ray_wise[0], data["image_sam"])

        pred_sam_mask = torch.from_numpy(pred_sam_mask).to(pred_rgb.device)
        if 'normal' in outputs.keys():
            pred_normal = outputs['normal'].reshape(B, H, W, -1)
            pred_normal = (pred_normal + 1.0) / 2.0
            if pred_normal.shape[-1] == 6:
                pred_normal = torch.cat([pred_normal[...,:3], pred_normal[...,3:]], dim = 2)        
        else:
            pred_normal = torch.ones_like(pred_rgb)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb)
        return pred_rgb, pred_normal, pred_depth, gt_rgb, loss, mask_gt, pred_sam_mask, pred_instance_map_ray_wise

    def test_step(self, data, bg_color=None, perturb=False):
        """
        Perform one test step.
        
        Args:
            data: Test data batch
            bg_color: Background color
            perturb: Whether to perturb ray sampling
            
        Returns:
            pred_rgb: Predicted RGB values
            pred_normal: Predicted normal map
            pred_depth: Predicted depth map
            pred_instance_map_ray_wise: Predicted instance map
        """
        poses = data["pose"] # [B, 4, 4]
        intrinsics = data["intrinsic"] # [B, 3, 3]
        H, W = int(data['H'][0]), int(data['W'][0]) # get the target size...

        B = poses.shape[0]

        rays_o, rays_d, _ = get_rays(poses, intrinsics, H, W, -1)

        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)

        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, 
                                    cos_anneal_ratio = min(self.epoch / 100, 1.0), normal_epsilon_ratio= min((self.epoch - 50) / 100, 0.99),
                                    **self.conf)

        pred_rgb = outputs['rgb'].reshape(B, H, W, -1)

        if 'normal' in outputs.keys():
            pred_normal = outputs['normal'].reshape(B, H, W, -1)
            pred_normal = (pred_normal + 1.0) / 2.0

            if pred_normal.shape[-1] == 6:
                pred_normal = torch.cat([pred_normal[...,:3], pred_normal[...,3:]], dim = 2)

        else:
            pred_normal = torch.ones_like(pred_rgb)

        pred_depth = outputs['depth'].reshape(B, H, W)
        pred_instance_map_ray_wise = outputs['instance_map_ray_wise'].reshape(B, H, W)
        return pred_rgb, pred_normal, pred_depth, pred_instance_map_ray_wise

    def save_mesh(self, save_path=None, resolution=256, aabb = None, bound=1, threshold=0.,  use_sdf = False, scale=1.0, offset=[0,0,0], query_color=True, format="dtu"):
        """
        Save the reconstructed mesh.
        
        Args:
            save_path: Path to save the mesh
            resolution: Resolution for mesh extraction
            aabb: Axis-aligned bounding box
            bound: Scene bound
            threshold: Threshold for surface extraction
            use_sdf: Whether to use SDF values
            scale: Scale factor for mesh
            offset: Offset for mesh
            query_color: Whether to query colors for vertices
            format: Output format ("dtu", "blender", or "blendedmvs")
        """
        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', f'{self.name}.obj')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                sdfs = self.model.density(pts.to(self.device), bound)
            return sdfs
        
        def query_func_color(pts):
            with torch.no_grad():
                N = pts.shape[0]
                rgb = np.zeros_like(pts)
                head = 0
                while head < N:
                    tail = min(head + 4096, N)
                    pts_ = pts[head:tail,:]
                    pts_ = torch.from_numpy(pts_).to(self.device).to(torch.float32)
                    sdfs, geo_feat = self.model.density(pts_, bound, return_feature=True)
                    gradient = self.model.gradient(pts_, bound, 0.005 * (1.0 - 0.99)).squeeze()
                    normal = F.normalize(gradient, p=2, dim=-1)
                    rgb_, _ = self.model.forward_color(pts_, -normal, normal, geo_feat, bound)
                    rgb[head:tail,:] = rgb_.detach().cpu().numpy()
                    head = tail
                    del pts_, sdfs, geo_feat, rgb_
                return rgb

        if format == "dtu":
            # w/ aabb
            bounds_min = torch.FloatTensor([-bound] * 3)
            bounds_max = torch.FloatTensor([bound] * 3)
            vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=threshold, query_func = query_func, use_sdf = use_sdf)
            if query_color:
                rgb = query_func_color(vertices)
            # align camera
            vertices = np.concatenate([vertices[:,2:3], vertices[:,0:1], vertices[:,1:2]], axis=-1)
        elif format == "blender":
            scale = max(0.000001,
                        max(max(abs(float(aabb[1][0])-float(aabb[0][0])),
                                abs(float(aabb[1][1])-float(aabb[0][1]))),
                                abs(float(aabb[1][2])-float(aabb[0][2]))))
                            
            scale = 2.0 * bound / scale

            offset =  torch.FloatTensor([
                        ((float(aabb[1][0]) + float(aabb[0][0])) * 0.5) * -scale,
                        ((float(aabb[1][1]) + float(aabb[0][1])) * 0.5) * -scale, 
                        ((float(aabb[1][2]) + float(aabb[0][2])) * 0.5) * -scale])
            bounds_min = torch.FloatTensor([-bound] * 3)
            bounds_max = torch.FloatTensor([bound] * 3)
            vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=threshold, query_func = query_func, use_sdf = use_sdf)
            if query_color:
                rgb = query_func_color(vertices)
            vertices = np.concatenate([vertices[:,2:3], vertices[:,0:1], vertices[:,1:2]], axis=-1)
            vertices = (vertices - offset.numpy()) / scale
        elif format == "blendedmvs":
            scale = max(0.000001,
                        max(max(abs(float(aabb[1][0])-float(aabb[0][0])),
                                abs(float(aabb[1][1])-float(aabb[0][1]))),
                                abs(float(aabb[1][2])-float(aabb[0][2]))))
                            
            scale = 2.0 * bound / scale

            offset =  torch.FloatTensor([
                        ((float(aabb[1][0]) + float(aabb[0][0])) * 0.5) * -scale,
                        ((float(aabb[1][1]) + float(aabb[0][1])) * 0.5) * -scale, 
                        ((float(aabb[1][2]) + float(aabb[0][2])) * 0.5) * -scale])
            bounds_min = torch.FloatTensor([-bound] * 3)
            bounds_max = torch.FloatTensor([bound] * 3)
            vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=threshold, query_func = query_func, use_sdf = use_sdf)
            if query_color:
                rgb = query_func_color(vertices)
            vertices = (vertices - offset.numpy()) / scale
        
        if query_color:
            mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=rgb, process=False)
        else:
            mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")
    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        """
        Train the model for a specified number of epochs.
        
        Args:
            train_loader: DataLoader for training data
            valid_loader: DataLoader for validation data
            max_epochs: Maximum number of training epochs
        """
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))
        
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch

            # Determine current training stage
            if self.epoch > 0 and self.epoch <= self.pretrain_epoch:
                self.cur_stage = 0
            elif self.epoch > self.pretrain_epoch and self.epoch <= self.pretrain_epoch + self.iterative:
                self.cur_stage = 1
            else:
                self.cur_stage = 2

            self.train_one_epoch(train_loader)
            if self.cur_stage in [0,1]:
                miou = self.loss_instances_hash.print_metrics()
                self.log(f"epoch: {self.epoch}, train miou: {miou}")
            
            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

            if self.epoch % 10 == 0:
                if self.workspace is not None and self.local_rank == 0:
                    self.save_checkpoint(full=True, best=False)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader):
        """
        Evaluate the model on a data loader.
        
        Args:
            loader: DataLoader for evaluation
        """
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None):
        """
        Test the model on a data loader and save results.
        
        Args:
            loader: DataLoader for testing
            save_path: Path to save test results
        """
        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()
        for i, data in enumerate(loader):
            data = self.prepare_data(data)

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, preds_normal, preds_depth, pred_instance, pred_instance_map_ray_wise = self.test_step(data)                
            
            path = os.path.join(save_path, f'{i:04d}.png')
            path_normal = os.path.join(save_path, f'{i:04d}_normal.png')
            path_depth = os.path.join(save_path, f'{i:04d}_depth.png')

            self.log(f"[INFO] saving test image to {path}")

            cv2.imwrite(path, cv2.cvtColor((preds[0].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(path_normal, cv2.cvtColor((preds_normal[0].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(path_depth, (preds_depth[0].detach().cpu().numpy() * 255).astype(np.uint8))

            pbar.update(loader.batch_size)

        self.log(f"==> Finished Test.")

    def prepare_data(self, data):
        """
        Prepare data by moving it to the correct device.
        
        Args:
            data: Input data
            
        Returns:
            Data moved to the correct device
        """
        if isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, np.ndarray):
                    data[i] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[i] = v.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[k] = v.to(self.device, non_blocking=True)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device, non_blocking=True)
        else:
            data = data.to(self.device, non_blocking=True)

        return data

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        self.local_step = 0
        for data in loader:
            
            self.local_step += 1
            self.global_step += 1
            data = self.prepare_data(data)

            if self.cur_stage in [1] and (self.epoch - 1) % self.update_pseudo_label_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        preds, preds_normal, preds_depth, pred_instance_map_ray_wise = self.test_step(data)
                
                cv2.imwrite(os.path.join(self.mask_output, "depth_%d_%d.png"%(int(data["index"][0]), self.epoch)), (preds_depth[0].detach().cpu().numpy() * 255).astype(np.uint8))
                
                cv2.imwrite(os.path.join(self.mask_output, "rgb_%d_%d.png"%(int(data["index"][0]), self.epoch)), cv2.cvtColor((preds[0].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                cv2.imwrite(os.path.join(self.mask_output, "instance_%d_%d.png"%(int(data["index"][0]), self.epoch)), (pred_instance_map_ray_wise[0].detach().cpu().numpy() * 255).astype(np.uint8))
                self.update_pseudo_masks(int(data["index"][0]), pred_instance_map_ray_wise[0], data["image_sam"])

                self.model.train()

            if self.cur_stage in [0]  and self.epoch == 1 and int(data["index"][0]) in self.annotation_index:
                self.make_pseudo_mask(data["image_sam"][0], int(data["index"][0]))

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.item()

            if self.local_rank == 0:
                for metric in self.metrics:
                    metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                    self.writer.add_scalar("train/inv_s", self.model.forward_variance()[0,0].detach(), self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                self.log(f"loss={loss.item():.4f} ({total_loss/self.local_step:.4f}), s_val={self.model.forward_variance()[0,0].detach().cpu():.2f}, lr={self.optimizer.param_groups[0]['lr']:.6f}")

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            for metric in self.metrics:
                self.log(metric.report(), style="red")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="train")
                metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")


    def evaluate_one_epoch(self, loader):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics_val:
                metric.clear()

            for metric_name, metric in self.metrics_seg.items():
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        #with torch.no_grad():
        self.local_step = 0
        for data in loader:    
            self.local_step += 1
            
            data = self.prepare_data(data)


            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, preds_normal, preds_depth, truths, loss, mask_gt, pred_sam_mask, pred_instance_map_ray_wise = self.eval_step(data)

            total_loss += loss.item()

            # only rank = 0 will perform evaluation.
            if self.local_rank == 0:

                for metric in self.metrics_val:
                    metric.update(preds, truths)

                for metric_name, metric in self.metrics_seg.items():
                    if metric_name == 'Pred':
                        metric.update(pred_instance_map_ray_wise, mask_gt)
                    elif metric_name == 'SAM':
                        metric.update(pred_sam_mask, mask_gt)

                    # save image
                save_path = os.path.join(self.validation_output, 'rgb_%d.png'%(int(data["index"][0])))
                save_path_depth = os.path.join(self.validation_output, 'depth_%d.png'%(int(data["index"][0])))
                save_path_gt = os.path.join(self.validation_output, 'gt_%d.png'%(int(data["index"][0])))

                cv2.imwrite(save_path, cv2.cvtColor((preds[0].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                cv2.imwrite(save_path_depth, (preds_depth[0].detach().cpu().numpy() * 255).astype(np.uint8))
                cv2.imwrite(save_path_gt, cv2.cvtColor((truths[0].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    
                pbar.set_description(f"loss={loss.item():.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics_val) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics_val:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

            for metric_name, metric in self.metrics_seg.items():
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, full=False, best=False):

        state = {
            'epoch': self.epoch,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}.pth.tar"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = self.stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.epoch = 1

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
        
        if self.optimizer and  'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer, use default.")
        
        # strange bug: keyerror 'lr_lambdas'
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler, use default.")
        
        if 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler, use default.")


class MiouMeter:
    """
    Meter for computing mean Intersection over Union (mIoU).
    """
    def __init__(self, type):
        """
        Initialize the mIoU meter.
        
        Args:
            type: Type of mIoU meter
        """
        self.V = 0
        self.N = 0
        self.type = type

    def clear(self):
        """Reset the meter."""
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        """
        Prepare inputs for computation.
        
        Args:
            *inputs: Input tensors
            
        Returns:
            Prepared inputs
        """
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        """
        Update the meter with new predictions and ground truth.
        
        Args:
            preds: Predictions
            truths: Ground truth
            
        Returns:
            Current IoU value
        """
        preds = preds.view(-1).to(torch.bool)
        truths = truths.view(-1).to(torch.bool)
        intersection = (preds*truths).sum(-1)
        union = (preds+truths).sum(-1)
        iou = intersection / (union + 1e-8)
        iou = iou.detach().cpu().numpy()
        
        self.V += iou
        self.N += 1

        return iou

    def measure(self):
        """
        Compute the mean IoU.
        
        Returns:
            Mean IoU value
        """
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        """
        Write the metric to tensorboard.
        
        Args:
            writer: Tensorboard writer
            global_step: Current global step
            prefix: Prefix for the metric name
        """
        writer.add_scalar(os.path.join(prefix, "mIoU(%s)"%(self.type)), self.measure(), global_step)

    def report(self):
        """
        Get a string report of the metric.
        
        Returns:
            String report
        """
        return f'mIoU({self.type:s})= {self.measure():.6f}'

class PSNRMeter:
    """
    Meter for computing Peak Signal-to-Noise Ratio (PSNR).
    """
    def __init__(self):
        """Initialize the PSNR meter."""
        self.V = 0
        self.N = 0

    def clear(self):
        """Reset the meter."""
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        """
        Prepare inputs for computation.
        
        Args:
            *inputs: Input tensors
            
        Returns:
            Prepared inputs
        """
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        """
        Update the meter with new predictions and ground truth.
        
        Args:
            preds: Predictions
            truths: Ground truth
            
        Returns:
            Current PSNR value
        """
        preds, truths = self.prepare_inputs(preds, truths)
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        self.V += psnr
        self.N += 1

        return psnr

    def measure(self):
        """
        Compute the mean PSNR.
        
        Returns:
            Mean PSNR value
        """
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        """
        Write the metric to tensorboard.
        
        Args:
            writer: Tensorboard writer
            global_step: Current global step
            prefix: Prefix for the metric name
        """
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        """
        Get a string report of the metric.
        
        Returns:
            String report
        """
        return f'PSNR = {self.measure():.6f}'

class LPIPSMeter:
    """
    Meter for computing Learned Perceptual Image Patch Similarity (LPIPS).
    """
    def __init__(self, net='vgg', device=None):
        """
        Initialize the LPIPS meter.
        
        Args:
            net: Network architecture for LPIPS
            device: Device to use for computation
        """
        self.V = 0
        self.N = 0
        self.net = net

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        """Reset the meter."""
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        """
        Prepare inputs for computation.
        
        Args:
            *inputs: Input tensors
            
        Returns:
            Prepared inputs
        """
        outputs = []
        for i, inp in enumerate(inputs):
            if len(inp.shape) == 3:
                inp = inp.unsqueeze(0)
            inp = inp.permute(0, 3, 1, 2).contiguous()
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs
    
    def update(self, preds, truths):
        """
        Update the meter with new predictions and ground truth.
        
        Args:
            preds: Predictions
            truths: Ground truth
            
        Returns:
            Current LPIPS value
        """
        preds, truths = self.prepare_inputs(preds, truths)
        v = self.fn(truths, preds, normalize=True).item()
        self.V += v
        self.N += 1

        return v
    
    def measure(self):
        """
        Compute the mean LPIPS.
        
        Returns:
            Mean LPIPS value
        """
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        """
        Write the metric to tensorboard.
        
        Args:
            writer: Tensorboard writer
            global_step: Current global step
            prefix: Prefix for the metric name
        """
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        """
        Get a string report of the metric.
        
        Returns:
            String report
        """
        return f'LPIPS ({self.net}) = {self.measure():.6f}'

class SSIMMeter:
    """
    Meter for computing Structural Similarity Index Measure (SSIM).
    """
    def __init__(self, device=None):
        """
        Initialize the SSIM meter.
        
        Args:
            device: Device to use for computation
        """
        self.V = 0
        self.N = 0

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def clear(self):
        """Reset the meter."""
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        """
        Prepare inputs for computation.
        
        Args:
            *inputs: Input tensors
            
        Returns:
            Prepared inputs
        """
        outputs = []
        for i, inp in enumerate(inputs):
            if len(inp.shape) == 3:
                inp = inp.unsqueeze(0)
            inp = inp.permute(0, 3, 1, 2).contiguous()
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        """
        Update the meter with new predictions and ground truth.
        
        Args:
            preds: Predictions
            truths: Ground truth
        """
        preds, truths = self.prepare_inputs(preds, truths)
        ssim = structural_similarity_index_measure(preds, truths)

        self.V += ssim
        self.N += 1

    def measure(self):
        """
        Compute the mean SSIM.
        
        Returns:
            Mean SSIM value
        """
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        """
        Write the metric to tensorboard.
        
        Args:
            writer: Tensorboard writer
            global_step: Current global step
            prefix: Prefix for the metric name
        """
        writer.add_scalar(os.path.join(prefix, "SSIM"), self.measure(), global_step)

    def report(self):
        """
        Get a string report of the metric.
        
        Returns:
            String report
        """
        return f'SSIM = {self.measure():.6f}'