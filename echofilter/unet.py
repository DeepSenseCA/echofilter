"""
Implementation of the U-Net model.

Adapted from
https://github.com/milesial/Pytorch-UNet/tree/060bdcd69886a3082a6f8fb7746e12d5fca3e360
under GPLv3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='border'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='border'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
            mode='replicate',
        )
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def rint(x):
    '''
    Returns rounded value, cast as an int.
    '''
    return int(round(x))


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        n_classes,
        bilinear=True,
        n_steps=4,
        latent_channels=64,
        expansion_factor=2,
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        lc = latent_channels
        xf = expansion_factor

        self.inc = DoubleConv(in_channels, rint(lc))

        self.down_steps = nn.ModuleList()
        self.up_steps = nn.ModuleList()
        for i_step in range(n_steps):
            nodes_here = rint(lc * (xf ** i_step))
            nodes_next = rint(lc * (xf ** min(n_steps - 1, i_step + 1)))
            self.down_steps.append(Down(nodes_here, nodes_next))
        for i_step in range(n_steps - 1, -1, -1):
            nodes_here = 2 * rint(lc * (xf ** i_step))
            nodes_next = rint(lc * (xf ** max(0, i_step - 1)))
            self.up_steps.append(Up(nodes_here, nodes_next, bilinear=bilinear))

        self.outc = OutConv(rint(lc), n_classes)

    def forward(self, x):
        x = self.inc(x)
        memory = [x]
        for step in self.down_steps:
            memory.append(step(memory[-1]))
        x = memory.pop()
        for step in self.up_steps:
            x = step(x, memory.pop())
        logits = self.outc(x)
        return logits
