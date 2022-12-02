import torch
import torch.nn as nn


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        c_in_list = [64, 64, 128]
        num_filters = [64, 128, 256]
        layer_nums = [3, 5, 5]
        upsample_strides = [1, 2, 4]

        for idx in range(3):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3, stride=2, padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(
                    num_filters[idx], 128, upsample_strides[idx], stride=upsample_strides[idx], bias=False),
                nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                nn.ReLU()
            ))

    def forward(self, x):
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            ups.append(self.deblocks[i](x))
        x = torch.cat(ups, dim=1)
        return x

