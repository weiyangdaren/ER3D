import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict

import os
import sys

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("AR3D"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.anchor_generator import AnchorGenerator
from utils.box_coder_utils import ResidualCoder


class DetHead(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.decoupled_item = ['x', 'y', 'z']
        self.det_item = ['x', 'y', 'z', 'dx', 'dy', 'dz', 'rz']
        self.retain_item = [x for x in self.det_item if x not in self.decoupled_item]

        self.conv_cls = nn.Conv2d(384, 2, kernel_size=1)
        self.conv_box1 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=1),
            nn.BatchNorm2d(384, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(384, 6, kernel_size=1)
        )
        self.conv_box2 = nn.Conv2d(384, 8, kernel_size=1)
        self.conv_dir_cls = nn.Conv2d(384, 4, kernel_size=1)
        self.dir_offset = 0.78539

        self.box_coder = ResidualCoder(len(self.det_item))

        self.point_range = cfgs.point_range
        self.grid_size = cfgs.grid_size
        self.anchor_config = [
            EasyDict({
                'anchor_sizes': [[3.9, 1.6, 1.56]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-1.78],
                'align_center': False,
            })
        ]
        anchor_generator = AnchorGenerator(
            anchor_range=self.point_range,
            anchor_generator_config=self.anchor_config
        )
        feature_map_size = [self.grid_size[:2] // 2]
        anchors_list, _ = anchor_generator.generate_anchors(feature_map_size)
        self.anchors = [x.cuda() for x in anchors_list]

    def limit_period(self, val, offset=0.5, period=np.pi):
        ans = val - torch.floor(val / period + offset) * period
        return ans

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds):
        anchors = torch.cat(self.anchors, dim=-3)
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float()
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)
        dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1)
        dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

        dir_rot = self.limit_period(batch_box_preds[..., 6] - self.dir_offset, 0)
        batch_box_preds[..., 6] = dir_rot + self.dir_offset + np.pi * dir_labels.to(batch_box_preds.dtype)
        return batch_cls_preds, batch_box_preds

    def forward(self, x):
        batch_size = x.shape[0]
        cls_preds = self.conv_cls(x)
        box_preds1 = self.conv_box1(x)
        box_preds2 = self.conv_box2(x)
        dir_cls_preds = self.conv_dir_cls(x)

        box_preds_list = []
        box_preds_dict = {}
        for i in range(len(self.decoupled_item)):
            box_preds_dict[self.decoupled_item[i]] = box_preds1[:, i * 2:(i + 1) * 2, ...]
        for i in range(len(self.retain_item)):
            box_preds_dict[self.retain_item[i]] = box_preds2[:, i * 2:(i + 1) * 2, ...]
        for k in self.det_item:
            box_preds_list.append(box_preds_dict[k])
        box_preds = torch.cat(box_preds_list, dim=1)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(batch_size, cls_preds, box_preds,
                                                                         dir_cls_preds)

        return batch_cls_preds, batch_box_preds
