"""
Convolutional layers.
"""

import functools
import itertools
import math
import numbers

import torch
from torch import nn
from torch.nn import functional as F

from .utils import _ntuple, same_to_padding


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
        **kwargs,
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
            **kwargs,
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
            **kwargs,
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
        **kwargs,
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
            **kwargs,
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=groups,
            **kwargs,
        )

    def foward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed
    seperately for each channel in the input using a depthwise convolution.

    Parameters
    ----------
    channels : int or sequence
        Number of channels of the input tensors. Output will have this number
        of channels as well.
    kernel_size : int or sequence
        Size of the gaussian kernel.
    sigma : float or sequence
        Standard deviation of the gaussian kernel.
    padding : int or sequence or "same", optional
        Amount of padding to use, for each side of each dimension. If this is
        `"same"` (default) the amount of padding will be set automatically
        to ensure the size of the tensor is unchanged.
    pad_mode : str, optional
        Padding mode. See :meth:`torch.nn.functional.pad` for options. Default
        is `"replicate"`.
    ndim : int, optional
        The number of dimensions of the data. Default value is 2 (spatial).

    Notes
    -----
    Based on https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
    """

    def __init__(
        self, channels, kernel_size, sigma, padding="same", pad_mode="replicate", ndim=2
    ):
        super(GaussianSmoothing, self).__init__()

        # Ensure arguments are sequences of the correct length
        fntuple = _ntuple(ndim)
        kernel_size = fntuple(kernel_size)
        sigmas = fntuple(sigma)

        self.noop = all(s == 0 for s in sigmas)
        if self.noop:
            return

        # Handle padding arguments
        if padding is "same":
            padding = same_to_padding(kernel_size, ndim=ndim)
            padding = tuple(
                itertools.chain.from_iterable(itertools.repeat(p, 2) for p in padding)
            )
        else:
            padding = _ntuple(ndim * 2)(padding)
        self.padding = padding
        self.pad_mode = pad_mode

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigmas, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(((mgrid - mean) / std) ** 2) / 2) / (
                std * math.sqrt(2 * math.pi)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if ndim == 1:
            self.conv = F.conv1d
        elif ndim == 2:
            self.conv = F.conv2d
        elif ndim == 3:
            self.conv = F.conv3d
        else:
            raise ValueError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(ndim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.

        Parameters
        ----------
        input : torch.Tensor
            Input to apply gaussian filter on.

        Returns
        -------
        filtered : torch.Tensor
            Filtered output, the same size as the input.
        """
        if self.noop:
            return input
        input = F.pad(input, self.padding, mode=self.pad_mode)
        return self.conv(input, weight=self.weight, groups=self.groups)
