import math
import numpy as np
import torch


def points_sparsify_by_angle(raw_points, sparse_point, H=256, W=512, slice=1):
    """
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param slice: output every slice lines
    """

    dtheta = math.radians(0.4 * 64.0 / H)
    dphi = math.radians(90.0 / W)
    x, y, z, i = raw_points[:, 0], raw_points[:, 1], raw_points[:, 2], raw_points[:, 3]

    if isinstance(raw_points, np.ndarray):
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        r = np.sqrt(x ** 2 + y ** 2)
        phi = np.radians(45.) - np.arcsin(y / r)
        phi_ = (phi / dphi).astype(int)
        theta = np.radians(2.) - np.arcsin(z / d)
        theta_ = (theta / dtheta).astype(int)
    else:
        d = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
        r = torch.sqrt(x ** 2 + y ** 2)
        phi = math.radians(45.) - torch.arcsin(y / r)
        phi_ = (phi / dphi).long()
        theta = math.radians(2.) - torch.arcsin(z / d)
        theta_ = (theta / dtheta).long()

    d[d == 0] = 0.000001
    r[r == 0] = 0.000001

    phi_[phi_ < 0] = 0
    phi_[phi_ >= W] = W - 1

    theta_[theta_ < 0] = 0
    theta_[theta_ >= H] = H - 1

    sparse_point[theta_, phi_, 0] = x
    sparse_point[theta_, phi_, 1] = y
    sparse_point[theta_, phi_, 2] = z
    sparse_point[theta_, phi_, 3] = i

    sparse_point = sparse_point[0::slice, :, :]
    sparse_point = sparse_point.reshape((-1, 4))
    sparse_point = sparse_point[sparse_point[:, 0] != -1.0]
    return sparse_point