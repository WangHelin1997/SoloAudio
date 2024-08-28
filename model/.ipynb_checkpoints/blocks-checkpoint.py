import torch
import torch.nn as nn


class TimbreBlock(nn.Module):
    def __init__(self, out_dim):
        super(TimbreBlock, self).__init__()
        base_dim = out_dim // 4
        self.block11 = nn.Sequential(nn.Conv2d(1, 2 * base_dim, 3, 1, 1),
                                     nn.InstanceNorm2d(2 * base_dim, affine=True),
                                     nn.GLU(dim=1))

        self.block12 = nn.Sequential(nn.Conv2d(base_dim, 2 * base_dim, 3, 1, 1),
                                     nn.InstanceNorm2d(2 * base_dim, affine=True),
                                     nn.GLU(dim=1))
        self.block21 = nn.Sequential(nn.Conv2d(base_dim, 4 * base_dim, 3, 1, 1),
                                     nn.InstanceNorm2d(4 * base_dim, affine=True),
                                     nn.GLU(dim=1))

        self.block22 = nn.Sequential(nn.Conv2d(2 * base_dim, 4 * base_dim, 3, 1, 1),
                                     nn.InstanceNorm2d(4 * base_dim, affine=True),
                                     nn.GLU(dim=1))

        self.block31 = nn.Sequential(nn.Conv2d(2 * base_dim, 8 * base_dim, 3, 1, 1),
                                     nn.InstanceNorm2d(8 * base_dim, affine=True),
                                     nn.GLU(dim=1))

        self.block32 = nn.Sequential(nn.Conv2d(4 * base_dim, 8 * base_dim, 3, 1, 1),
                                     nn.InstanceNorm2d(8 * base_dim, affine=True),
                                     nn.GLU(dim=1))

        self.final_conv = nn.Conv2d(4 * base_dim, out_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = self.block11(x)
        y = self.block12(y)
        y = self.block21(y)
        y = self.block22(y)
        y = self.block31(y)
        y = self.block32(y)
        y = self.final_conv(y)

        return y.sum((2, 3)) / (y.shape[2] * y.shape[3])