"""
Convolutional layers.
"""

import functools

import torch
from torch import nn

from .utils import same_to_padding


class Conv2dSame(nn.Conv2d):
    """
    2D Convolutions with same padding option.

    Same padding will only produce an output size which matches the input size
    if the kernel size is odd and the stride is 1.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        dilation=1,
        **kwargs
    ):
        if padding is "same":
            padding = same_to_padding(kernel_size, stride, dilation, ndim=2)

        super(Conv2dSame, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            **kwargs
        )


class PointwiseConv2d(nn.Conv2d):
    """
    2D Pointwise Convolution.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        if "kernel_size" in kwargs:
            raise ValueError("kernel_size must be 1 for a pointwise convolution.")
        super(PointwiseConv2d, self).__init__(
            in_channels, out_channels, kernel_size=1, **kwargs
        )


class DepthwiseConv2d(nn.Conv2d):
    """
    2D Depthwise Convolution.
    """

    def __init__(
        self, in_channels, kernel_size=3, stride=1, padding="same", dilation=1, **kwargs
    ):

        if "groups" in kwargs:
            raise ValueError(
                "Number of groups must equal number of input channels for a depthwise convolution."
            )

        if padding is "same":
            padding = same_to_padding(kernel_size, stride, dilation, ndim=2)

        super(DepthwiseConv2d, self).__init__(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            **kwargs
        )


class SeparableConv2d(nn.Module):
    """
    2D Depthwise Separable Convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        dilation=1,
        groups=1,
        **kwargs
    ):
        super(SeparableConv2d, self).__init__()

        if padding is "same":
            padding = same_to_padding(kernel_size, stride, dilation, ndim=2)

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            **kwargs
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=groups,
            **kwargs
        )

    def foward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
