"""
Connectors and pathing modules.
"""

import torch
from torch import nn
import torch.nn.functional as F

from .conv import PointwiseConv2d


class ResidualConnect(nn.Module):
    """
    Joins up a residual connection, with smart mapping for changes in the
    number of channels.
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualConnect, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels == out_channels:
            # No need to do anything to our passthru input
            self.op = nn.Identity()
        elif in_channels < out_channels:
            # The number of channels has increased, so keep the original
            # input as it is, and pad up to match the size of the "residual".
            self.op = PointwiseConv2d(in_channels, out_channels - in_channels)
        else:
            # The number of channels has decreased, so we need to map the
            # original down to the size of the "residual".
            self.op = PointwiseConv2d(in_channels, out_channels)

    def forward(self, residual, passed_thru):
        if self.in_channels < self.out_channels:
            return residual + torch.cat([passed_thru, self.op(passed_thru)], dim=1)
        return residual + self.op(passed_thru)


class FlexibleConcat2d(nn.Module):
    """
    Concatenate two inputs of nearly the same shape.
    """

    def forward(self, x1, x2):
        """
        Parameters
        ----------
        x1 : torch.Tensor
            Tensor, possibly smaller than `x2`.
        x2 : torch.Tensor
            Tensor, at least as large as `x1`.

        Returns
        -------
        torch.Tensor
            Concatenated `x1` (padded if necessary) and `x2`, along
            dimension `1`.
        """
        # input is CHW
        diffY = torch.tensor([x2.shape[-2] - x1.shape[-2]])
        diffX = torch.tensor([x2.shape[-1] - x1.shape[-1]])

        if diffX != 0 or diffY != 0:
            x1 = F.pad(
                x1,
                [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
                mode="replicate",
            )
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        return torch.cat([x1, x2], dim=1)
