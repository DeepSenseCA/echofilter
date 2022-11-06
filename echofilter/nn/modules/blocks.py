"""
Blocks of modules.
"""

import torch
from torch import nn

from .activations import str2actfnfactory
from .conv import Conv2dSame, PointwiseConv2d, DepthwiseConv2d
from .pathing import ResidualConnect
from .utils import _pair


__all__ = ["MBConv", "SqueezeExcite"]


class SqueezeExcite(nn.Module):
    """
    Squeeze and excitation block.

    See https://arxiv.org/abs/1709.01507

    Parameters
    ----------
    in_channels : int
        Number of input (and output) channels.
    reduction : int or float, optional
        Compression factor for the number of channels in the squeeze and
        excitation attention module. Default is `4`.
    actfn : str or callable, optional
        An activation class or similar generator. Default is an inplace
        ReLU activation. If this is a string, it is mapped to a generator with
        `activations.str2actfnfactory`.
    """

    def __init__(
        self,
        in_channels,
        reduction=4,
        actfn="InplaceReLU",
    ):
        super(SqueezeExcite, self).__init__()

        actfn_factory = actfn
        if isinstance(actfn, str):
            actfn_factory = str2actfnfactory(actfn)

        reduced_chns = int(max(1, round(in_channels / reduction)))

        layers = [
            nn.AdaptiveAvgPool2d(1),
            PointwiseConv2d(in_channels, reduced_chns),
            actfn_factory(),
            PointwiseConv2d(reduced_chns, in_channels),
            nn.Sigmoid(),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return input * self.layers(input)


class MBConv(nn.Module):
    """
    MobileNet style inverted residual block.

    See https://arxiv.org/abs/1905.11946 and https://arxiv.org/abs/1905.02244.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int, optional
        Number of output channels. Default is to match `in_channels`.
    expansion : int or float, optional
        Exansion factor for the inverted-residual bottleneck. Default is `6`.
    se_reduction : int, optional
        Reduction factor for squeeze-and-excite block. Default is `4`. Set
        to `None` or `0` to disable squeeze-and-excitation.
    fused : bool, optional
        If `True`, the pointwise and depthwise convolution are fused together
        into a single regular convolution. Default is `False` (a depthwise
        separable convolution).
    residual : bool, optional
        If `True`, the block is residual with a skip-through connection.
        Default is `True`.
    actfn : str or callable, optional
        An activation class or similar generator. Default is an inplace
        ReLU activation. If this is a string, it is mapped to a generator with
        `activations.str2actfnfactory`.
    bias : bool, optional
        If `True`, the main convolution has a bias term. Default is `False`.
        Note that the pointwise convolutions never have bias terms.
    **conv_args
        Additional arguments, such as kernel_size, stride, and padding, which
        will be passed to the convolution module.
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        expansion=6,
        se_reduction=4,
        fused=False,
        residual=True,
        actfn="InplaceReLU",
        bias=False,
        **conv_args,
    ):
        super(MBConv, self).__init__()

        if out_channels is None:
            out_channels = in_channels

        self.residual = residual
        self.fused = fused

        actfn_factory = actfn
        if isinstance(actfn, str):
            actfn_factory = str2actfnfactory(actfn)

        expanded_chns = int(round(in_channels * expansion))

        if expansion == 1 or fused:
            self.expansion_conv = nn.Identity()
        else:
            self.expansion_conv = nn.Sequential(
                PointwiseConv2d(in_channels, expanded_chns, bias=False),
                nn.BatchNorm2d(expanded_chns),
                actfn_factory(),
            )
        if fused:
            conv = Conv2dSame(in_channels, expanded_chns, bias=bias, **conv_args)
        else:
            conv = DepthwiseConv2d(expanded_chns, bias=bias, **conv_args)
        self.conv = nn.Sequential(conv, nn.BatchNorm2d(expanded_chns), actfn_factory())

        if se_reduction:
            self.se = SqueezeExcite(expanded_chns, reduction=se_reduction)
        else:
            self.se = nn.Identity()

        self.contraction_conv = nn.Sequential(
            PointwiseConv2d(expanded_chns, out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if residual:
            if any([k > 1 for k in _pair(conv_args.get("stride", 1))]):
                self.skip_pool = nn.AvgPool2d(conv_args["stride"])
            else:
                self.skip_pool = nn.Identity()
            self.connector = ResidualConnect(in_channels, out_channels)

    def forward(self, input):
        x = self.expansion_conv(input)
        x = self.conv(x)
        x = self.se(x)
        x = self.contraction_conv(x)
        if self.residual:
            x = self.connector(x, self.skip_pool(input))
        return x

    def extra_repr(self):
        return "residual={residual}, fused={fused}".format(
            residual=self.residual,
            fused=self.fused,
        )
