import torch
import torch.nn as nn


class SimpleAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, num_residual_blocks=1):
        super(SimpleAdapter, self).__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=16)
        self.conv = nn.Conv2d(
            in_dim * 16 * 16,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_dim) for _ in range(num_residual_blocks)]
        )

    def forward(self, x):
        bs, c, f, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs * f, c, h, w)
        x = self.pixel_unshuffle(x)
        x = self.conv(x)
        x = self.residual_blocks(x)
        x = x.view(bs, f, x.size(1), x.size(2), x.size(3))
        return x.permute(0, 2, 1, 3, 4)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual
