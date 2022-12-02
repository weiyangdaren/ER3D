import torch
import numpy as np
from pathlib import Path
import cv2.cv2 as cv2
import torch.utils.data as torch_data
import torch.nn.functional as F
from collections import defaultdict
from spconv.pytorch.utils import PointToVoxel
import ctypes

import os
import sys

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("AR3D"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from dataset import calibration_kitti
from dataset import sparsify

lib_SGM = ctypes.CDLL('utils/sgm/libsgm_lib.so')


def convert_type(input):
    ctypes_map = {int: ctypes.c_int,
                  float: ctypes.c_double,
                  str: ctypes.c_char_p
                  }
    input_type = type(input)
    if input_type is list:
        length = len(input)
        if length == 0:
            print("convert type failed...input is " + input)
            exit(0)
        else:
            arr = (ctypes_map[type(input[0])] * length)()
            for i in range(length):
                arr[i] = bytes(input[i], encoding="utf-8") if (type(input[0]) is str) else input[i]
            return arr
    else:
        if input_type in ctypes_map:
            return ctypes_map[input_type](bytes(input, encoding="utf-8") if type(input) is str else input)
        else:
            print("convert type failed...input is " + input)
            exit(0)


class KittiDataset(torch_data.Dataset):
    def __init__(self, cfgs, is_cuda=False):
        """
            :param root_path:
        """
        super().__init__()
        root_path = Path(cfgs.data_path)
        self.data_path = root_path / ('training' if cfgs.data_split != 'test' else 'testing')
        self.spilt_file = root_path / 'ImageSets' / (cfgs.data_split + '.txt')
        self.file_name_list = [x.strip() for x in open(self.spilt_file).readlines()]
        self.is_cuda = is_cuda
        self.device = torch.device("cuda:0") if self.is_cuda else torch.device("cpu:0")

        self.voxel_size = cfgs.voxel_size
        self.point_range = cfgs.point_range

    def get_calib(self, filename):
        calib_file = self.data_path / 'calib' / ('%s.txt' % filename)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file, self.is_cuda)

    def get_image(self, filename, image_id=2):
        img_file = self.data_path / ('image_%s' % image_id) / ('%s.png' % filename)
        assert img_file.exists()
        return cv2.imread(str(img_file))

    def get_disp(self, filename, left_img, right_img):
        rows, cols, dim = left_img.shape
        disparity = np.zeros(dtype=np.float32, shape=(rows, cols))

        lib_SGM.SGM_run(rows, cols, left_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                        right_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                        disparity.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        convert_type(7), convert_type(86), 0)

        # disp_file = self.data_path / ('disparities') / ('%s.npy' % filename)
        # assert disp_file.exists()
        # return np.load(disp_file).astype(np.float32)
        return disparity

    def trans_disp_to_points(self, disp, calib, image=None):
        disp[disp < 0] = 0
        mask = disp > 0

        if isinstance(disp, np.ndarray):
            depth = calib.fu * 0.54 / (disp + 1. - mask)
            rows, cols = depth.shape
            c, r = np.meshgrid(np.arange(cols), np.arange(rows))
            points = np.stack([c, r, depth])
        else:
            depth = calib.fu * 0.54 / (disp + 1. - mask.long())
            rows, cols = depth.shape
            c, r = torch.meshgrid(torch.arange(cols), torch.arange(rows), indexing='ij')
            points = torch.stack([c.t().to(self.device), r.t().to(self.device), depth])

        points = points.reshape((3, -1))
        points = points.T
        points = points[mask.reshape(-1)]

        points = calib.rect_to_lidar(calib.img_to_rect(points[:, 0], points[:, 1], points[:, 2]))
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if isinstance(disp, np.ndarray):
                image = (image / 255).astype(np.float32)
                image = image.reshape((-1, 1))
                image = image[mask.reshape(-1)]
                points = np.concatenate([points, image], 1).astype(np.float32)
            else:
                image = torch.from_numpy(image / 255).float().to(self.device)
                image = image.reshape((-1, 1))
                image = image[mask.reshape(-1)]
                points = torch.cat([points, image], 1)
        else:
            if isinstance(disp, np.ndarray):
                points = np.concatenate([points, np.ones((points.shape[0], 1))], 1).astype(np.float32)
            else:
                points = torch.cat([points, torch.ones((points.shape[0], 1), device=self.device)], 1)
        return points

    def mask_points_by_range(self, points):
        point_range = np.array(self.point_range, dtype=np.float32)
        mask = (points[:, 0] >= point_range[0]) & (points[:, 0] <= point_range[3]) \
               & (points[:, 1] >= point_range[1]) & (points[:, 1] <= point_range[4]) \
               & (points[:, 2] >= point_range[2]) & (points[:, 2] <= point_range[5])
        return points[mask]

    def points_sparsify(self, points):
        dH, dW = 256, 512
        if isinstance(points, np.ndarray):
            sparse_point = - np.ones((dH, dW, 4)).astype(np.float32)
        else:
            sparse_point = - torch.ones((dH, dW, 4), device=self.device)
        sparse_point = sparsify.points_sparsify_by_angle(points, sparse_point, dH, dW, 1)
        return sparse_point

    def generate_voxels(self, points):
        voxel_generator = PointToVoxel(
            vsize_xyz=self.voxel_size,
            coors_range_xyz=self.point_range,
            num_point_features=4,
            max_num_points_per_voxel=32,
            max_num_voxels=40000,
            device=self.device
        )
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points)

        voxels_th, indices_th, num_p_in_vx_th = voxel_generator(points)
        return voxels_th, indices_th, num_p_in_vx_th

    def get_data(self, filename):
        data_dict = {}
        calib = self.get_calib(filename)
        left_img = self.get_image(filename, 2)
        right_img = self.get_image(filename, 3)
        disp_map = self.get_disp(filename, left_img, right_img)

        if self.is_cuda:
            disp_map = torch.from_numpy(disp_map).to(self.device)

        points = self.trans_disp_to_points(disp_map, calib, left_img)
        points = self.mask_points_by_range(points)
        sparse_points = self.points_sparsify(points)
        voxels, coords, num_points = self.generate_voxels(sparse_points)

        data_dict['voxels'] = voxels
        data_dict['coords'] = coords
        data_dict['num_points'] = num_points
        return data_dict

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, idx):
        filename = self.file_name_list[idx]
        data_dict = self.get_data(filename)

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        ret = {}
        for key, val in data_dict.items():
            if key in ['voxels', 'num_points']:
                if isinstance(val, np.ndarray):
                    ret[key] = np.concatenate(val, axis=0)
                else:
                    ret[key] = torch.cat(val, dim=0)
            elif key in ['coords']:
                coors = []
                if isinstance(val[0], np.ndarray):
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                else:
                    for i, coor in enumerate(val):
                        coor_pad = F.pad(coor, (1, 0, 0, 0), mode='constant', value=i)
                        coors.append(coor_pad)
                    ret[key] = torch.cat(coors, dim=0)
        return ret

