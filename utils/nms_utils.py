import torch

import os
import sys

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("AR3D"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)
from utils.iou3d_nms import iou3d_nms_utils


def class_agnostic_nms(box_scores, box_preds, score_thresh=None):
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(box_scores, k=min(4096, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        keep_idx, selected_scores = iou3d_nms_utils.nms_gpu(
                boxes_for_nms[:, 0:7], box_scores_nms, 0.01,
        )
        selected = indices[keep_idx[:500]]

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]