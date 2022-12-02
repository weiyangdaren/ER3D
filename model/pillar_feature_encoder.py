import torch
import torch.nn as nn
import torch.nn.functional as F


class PillarFeatureEnconder(nn.Module):
    def __init__(self, cfgs):
        super().__init__()

        self.linear = nn.Linear(10, 64, bias=False)
        self.norm = nn.BatchNorm1d(64, eps=1e-3, momentum=0.01)

        self.voxel_size = cfgs.voxel_size
        self.point_range = cfgs.point_range
        self.grid_size = cfgs.grid_size

        self.voxel_x = self.voxel_size[0]
        self.voxel_y = self.voxel_size[1]
        self.voxel_z = self.voxel_size[2]
        self.x_offset = self.voxel_x / 2 + self.point_range[0]
        self.y_offset = self.voxel_y / 2 + self.point_range[1]
        self.z_offset = self.voxel_z / 2 + self.point_range[2]
        self.nx = self.grid_size[0]
        self.ny = self.grid_size[1]
        self.nz = self.grid_size[2]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def process(self, data_dict):
        voxels, coords, num_points = data_dict['voxels'], data_dict['coords'], data_dict['num_points']
        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / num_points.type_as(voxels).view(-1, 1, 1)
        f_cluster = voxels[:, :, :3] - points_mean
        f_center = torch.zeros_like(voxels[:, :, :3])
        f_center[:, :, 0] = voxels[:, :, 0] - (
                coords[:, 3].to(voxels.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxels[:, :, 1] - (
                coords[:, 2].to(voxels.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxels[:, :, 2] - (
                coords[:, 1].to(voxels.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        voxel_feature = torch.cat([voxels, f_cluster, f_center], dim=-1)
        voxel_count = voxel_feature.shape[1]
        mask = self.get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxels)
        voxel_feature *= mask
        data_dict['voxel_feature'] = voxel_feature
        return data_dict

    def pillar_scatter(self, voxel_features, coords):
        batch_size = coords[:, 0].max().int().item() + 1
        bacth_bev_features = []

        for idx in range(batch_size):
            bev_features = torch.zeros(
                64, self.nz * self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device)
            batch_mask = coords[:, 0] == idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = voxel_features[batch_mask, :]
            pillars = pillars.t()
            bev_features[:, indices] = pillars
            bacth_bev_features.append(bev_features)

        bacth_bev_features = torch.stack(bacth_bev_features, 0)
        bacth_bev_features = bacth_bev_features.view(batch_size, 64 * self.nz, self.ny, self.nx)

        return bacth_bev_features

    def forward(self, data_dict):
        data_dict = self.process(data_dict)
        voxel_feature = data_dict['voxel_feature']
        x = self.linear(voxel_feature)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        torch.backends.cudnn.enabled = True

        x = F.relu(x)
        voxel_feature = torch.max(x, dim=1, keepdim=True)[0]
        return self.pillar_scatter(voxel_feature.squeeze(), data_dict['coords'])
