import torch
import torch.nn as nn
import numpy as np

import os
import sys
src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("AR3D"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils import nms_utils
from model.pillar_feature_encoder import PillarFeatureEnconder
from model.backbone import Backbone
from model.det_head import DetHead


class PointPillar(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        self.pfe = PillarFeatureEnconder(self.cfgs)
        self.backbone = Backbone()
        self.head = DetHead(self.cfgs)

    def post_processing(self, output):
        batch_cls_preds, batch_box_preds = output
        batch_size = batch_cls_preds.shape[0]
        pred_dicts = []

        for idx in range(batch_size):
            box_preds = batch_box_preds[idx]
            cls_preds = batch_cls_preds[idx]

            cls_preds, label_preds = torch.max(cls_preds, dim=-1)
            label_preds = label_preds + 1

            selected, selected_scores = nms_utils.class_agnostic_nms(
                box_scores=cls_preds, box_preds=box_preds,
                score_thresh=self.cfgs.score_thresh
            )

            final_scores = selected_scores
            final_labels = label_preds[selected]
            final_boxes = box_preds[selected]
            ret_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(ret_dict)

        return pred_dicts

    def forward(self, data_dict):
        x = self.pfe(data_dict)
        x = self.backbone(x)
        output = self.head(x)
        batch_cls_preds, batch_box_preds = output

        pred_dicts = self.post_processing(output)

        return pred_dicts