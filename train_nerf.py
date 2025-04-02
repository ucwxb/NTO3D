import torch

from nerf.utils import *

import argparse

if __name__ == '__main__':

    #Basic Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--mode', type=str, default='train', help="running mode, supports (train, mesh, render)")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_rays', type=int, default=8192)
    parser.add_argument('--num_steps', type=int, default=16)
    parser.add_argument('--downscale', type=int, default=1)
    parser.add_argument('--upsample_steps', type=int, default=128)
    parser.add_argument('--max_ray_batch', type=int, default=2048)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--anno_point', type=int, default=0)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--iterative', type=int, default=200)
    parser.add_argument('--pretrain_epoch', type=int, default=30)
    parser.add_argument('--agg_point_num', type=int, default=5)
    parser.add_argument('--sam_thr', type=float, default=0.0)

    parser.add_argument('--update_pseudo_label_interval', type=int, default=30)
    parser.add_argument('--sam_ckp_path', type=str, default='')
    parser.add_argument('--anno_path', type=str, default='')
    parser.add_argument('--sam_feat_path', type=str, default='')
    parser.add_argument('--train_with_sam_mask', action='store_true')

    parser.add_argument('--instance_loss', type=float, default=0.1)
    parser.add_argument('--color_loss', type=float, default=1.0)
    parser.add_argument('--eikonal_loss', type=float, default=0.1)

    #Network Settings
    parser.add_argument('--network', type=str, default='sdf', help="network format, supports ( \
                                                                    sdf: use sdf representation, \
                                                                    phasor: use phasor encoding for sdf representation, \
                                                                    tcnn: use TCNN backend for sdf representation, \
                                                                    enc: use only TCNN encoding for sdf representation, \
                                                                    fp16: use amp mixed precision training for sdf representation,\
                                                                    ff: use fully-fused MLP for sdf representation)")
    
    parser.add_argument('--curvature_loss', '--C', action='store_true', help="use curvature loss term, slower but make surface smoother")

    #Dataset Settings
    parser.add_argument('--format', type=str, default='dtu', help="dataset format, supports (colmap, blender)")
    parser.add_argument('--bound', type=float, default=1.0, help="assume the scene is bounded in box(-size, size)")
    parser.add_argument('--scale', type=float, default=1.0, help="assume the scene is bounded in box(-size, size)")

    #Others
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch (unstable now)")

    opt = parser.parse_args()

    print(opt)

    from nerf.network_sdf import NeRFNetwork
    
    if opt.format == 'dtu':
        from nerf.dtu_provider import NeRFDataset
    elif opt.format == 'blendedmvs':
        from nerf.mvs_provider import NeRFDataset
    else:
        from nerf.provider import NeRFDataset

    seed_everything(opt.seed)

    model = NeRFNetwork(
        encoding="hashgrid", encoding_dir="sphere_harmonics", 
        num_layers=5, hidden_dim=64, geo_feat_dim=15, num_layers_color=3, hidden_dim_color=64, 
        cuda_ray=opt.cuda_ray, curvature_loss = opt.curvature_loss
    )

    #optimizer
    if opt.network in ['tcnn', 'enc', 'sdf', 'phasor']:
        optimizer = lambda model: torch.optim.Adam([
        {'name': 'encoding', 'params': list(model.encoder.parameters()) + list(model.encoder_instance.parameters())},
        {'name': 'net', 'params': list(model.sdf_net.parameters()) + list(model.color_net.parameters()) + list(model.color_feat_net.parameters()) + list(model.deviation_net.parameters()), 'weight_decay': 1e-6},
    ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)

    else:
        optimizer = lambda model: torch.optim.Adam([
            {'name': 'encoding', 'params': list(model.encoder.parameters())},
            {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': 1e-6},
        ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)

    scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / 20000, 1))

    criterion = torch.nn.HuberLoss()

    trainer = Trainer('ngp', 
                vars(opt), 
                model, 
                workspace=opt.workspace, 
                optimizer=optimizer, 
                criterion=criterion, 
                ema_decay=0.95, 
                fp16=(opt.network=='fp16'), 
                lr_scheduler=scheduler, 
                scheduler_update_every_step=True, 
                use_checkpoint='latest', 
                eval_interval=opt.eval_interval,
                sam_ckp_path = opt.sam_ckp_path,
                anno_path = opt.anno_path,
                anno_point=opt.anno_point,
                pretrain_epoch=opt.pretrain_epoch,
                update_pseudo_label_interval=opt.update_pseudo_label_interval,
                agg_point_num = opt.agg_point_num,
                sam_thr=opt.sam_thr,
                sam_feat_path=opt.sam_feat_path,
                train_with_sam_mask=opt.train_with_sam_mask,
                iterative=opt.iterative,
            )
    

    if opt.mode == 'train':
        train_dataset = NeRFDataset(opt.path, type='train', mode=opt.format, bound=opt.bound, scale=opt.scale)
        valid_dataset = NeRFDataset(opt.path, type='val', mode=opt.format, downscale=opt.downscale, bound=opt.bound, scale=opt.scale)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)
        trainer.metrics = [PSNRMeter(),]
        trainer.train(train_loader, valid_loader, opt.epoch)

    elif opt.mode == 'val':
        valid_dataset = NeRFDataset(opt.path, type='val', mode=opt.format, downscale=opt.downscale, bound=opt.bound, scale=opt.scale)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)
        trainer.metrics = [PSNRMeter(), SSIMMeter(), LPIPSMeter()]
        trainer.evaluate(valid_loader)

    elif opt.mode == 'mesh':
        valid_dataset = NeRFDataset(opt.path, type='mesh', mode=opt.format, downscale=opt.downscale, bound=opt.bound, scale=opt.scale)
        if opt.format == 'dtu': 
            trainer.save_mesh(aabb = valid_dataset.aabb, resolution=512, threshold=0.0, use_sdf=(opt.network=='sdf'), bound=opt.bound, format=opt.format, scale=valid_dataset.scale_mats_scale, offset=valid_dataset.scale_mats_offset)
        else:
            trainer.save_mesh(aabb = valid_dataset.aabb, resolution=512, threshold=0.0, use_sdf=(opt.network=='sdf'), bound=opt.bound, format=opt.format)

    elif opt.mode == 'render':
        test_dataset = NeRFDataset(opt.path, type='test', mode=opt.format, downscale=opt.downscale, bound=opt.bound, scale=opt.scale)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
        trainer.test(test_loader)
