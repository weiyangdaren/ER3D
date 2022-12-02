import torch
from pathlib import Path
import numpy as np
from easydict import EasyDict
from torch.utils.data import DataLoader

from dataset.kitti_dataset import KittiDataset
from model.pointpillar import PointPillar


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if isinstance(val, np.ndarray):
            batch_dict[key] = torch.from_numpy(val).float().cuda()
        elif isinstance(val, torch.Tensor):
            batch_dict[key] = val.float().cuda()
        else:
            continue


def parse_config():
    cfgs = EasyDict({
        'data_path': './data/kitti_sample',
        'data_split': 'val',
        'batch_size': 1,
        'voxel_size': [0.16, 0.16, 4],
        'point_range': [0, -39.68, -3, 69.12, 39.68, 1],
        'score_thresh': 0.1,
        'ckpt': 'ckpt/sgm_256_pointpillar_decoupled_xyz_ciou.pth'
    })
    nx = int((cfgs.point_range[3] - cfgs.point_range[0]) / cfgs.voxel_size[0])
    ny = int((cfgs.point_range[4] - cfgs.point_range[1]) / cfgs.voxel_size[1])
    nz = int((cfgs.point_range[5] - cfgs.point_range[2]) / cfgs.voxel_size[2])
    cfgs['grid_size'] = np.array([nx, ny, nz])

    return cfgs


if __name__ == '__main__':

    cfgs = parse_config()

    model = PointPillar(cfgs)
    model_state = model.state_dict()
    checkpoint = torch.load(cfgs.ckpt)
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()

    kitti_data = KittiDataset(cfgs, is_cuda=True)
    dataloader = DataLoader(kitti_data, batch_size=cfgs.batch_size, shuffle=False,
                            collate_fn=kitti_data.collate_batch)

    for i, data_dict in enumerate(dataloader):
        x = model(data_dict)
        print('pred_boxes: ', x[0]['pred_boxes'])
