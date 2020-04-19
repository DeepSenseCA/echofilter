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

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,
                      padding_mode='border', stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                      padding_mode='border'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, pool='max'):
        super().__init__()
        modules = []

        conv_stride = 1
        if pool == 'max':
            modules.append(nn.MaxPool2d(2))
        elif pool == 'avg':
            modules.append(nn.AvgPool2d(2))
        elif pool == 'stride':
            conv_stride = 2
        else:
            raise ValueError('Unsupported pooling method: {}'.format(pool))

        modules.append(DoubleConv(in_channels, out_channels, stride=conv_stride))
        self.pool_conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels_to_upscale, in_channels_skip, out_channels, bilinear=True):
        super().__init__()

        if in_channels_to_upscale is None:
            in_channels_to_upscale = in_channels // 2

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels_to_upscale, in_channels_to_upscale, kernel_size=2, stride=2,
            )

        self.conv = DoubleConv(in_channels_to_upscale + in_channels_skip, out_channels)

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
        exponent_matching='in',
        down_pool='max',
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Store generation parameters
        self.n_steps = n_steps
        self.latent_channels = latent_channels
        self.expansion_factor = expansion_factor
        self.exponent_matching = exponent_matching
        self.down_pool = down_pool

        lc = latent_channels
        xf = expansion_factor

        self.inc = DoubleConv(in_channels, rint(lc))

        self.down_steps = nn.ModuleList()
        self.up_steps = nn.ModuleList()
        for i_step in range(n_steps):
            # Number of channels in the input
            expo_prev = i_step
            nodes_here = rint(lc * (xf ** expo_prev))
            # Number of channels in the output
            expo_next = expo_prev + 1
            if self.exponent_matching == 'in':
                # Final step doesn't increase the number of channels
                expo_next = min(n_steps - 1, expo_next)
            nodes_next = rint(lc * (xf ** expo_next))
            # Create the layer and add it to the module list
            self.down_steps.append(Down(nodes_here, nodes_next, pool=down_pool))
        for i_step in range(n_steps - 1, -1, -1):
            # Either we have the same number of channels for both inputs
            # (the skip connection and the previous layer), or we can have
            # the number of channels match for the skip connection (which was
            # the output of Down at this step) and our output at this step.
            if self.exponent_matching == 'in':
                expo_prev = i_step
            elif self.exponent_matching == 'out':
                expo_prev = i_step + 1
            else:
                raise ValueError('Unrecognised exponent_matching: {}'.format(exponent_matching))
            expo_next = max(0, expo_prev - 1)
            # Number of channels which are passed through at the full spatial
            # resolution (skip connection)
            nodes_from_skip = rint(lc * (xf ** i_step))
            # Number of channels which come from the previous layer and need
            # to be upsampled
            nodes_from_prev = rint(lc * (xf ** expo_prev))
            nodes_incoming = nodes_from_prev + nodes_from_skip
            nodes_next = rint(lc * (xf ** expo_next))
            self.up_steps.append(
                Up(nodes_from_prev, nodes_from_skip, nodes_next, bilinear=bilinear)
            )

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
