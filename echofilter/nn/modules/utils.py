"""
nn.modules utility functions.
"""

import collections
from itertools import repeat

import torch
from torch import nn


__all__ = ["same_to_padding", "init_cnn"]


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            x = tuple(x)
            if len(x) == 0:
                raise ValueError("Input {} is an empty iterable".format(x))
            if len(x) > 1 or n == 1:
                return x
            x = x[0]
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def same_to_padding(kernel_size, stride=1, dilation=1, ndim=None):
    """
    Determines the amount of padding to use for a convolutional layer.

    Parameters
    ----------
    kernel_size : int or sequence
        Size of kernel for each dimension.
    stride : int or sequence, optional
        Amount of stride to apply in each dimension of the kernel. If `stride`
        is an int, the same value is applied for each dimension.
        Default is `1`.
    dilation : int or sequence, optional
        Amount of dilation to apply in each dimension of the kernel.
        If `dilation` is an int, the same value is applied for each dimension.
        Default is `1`.
    ndim : int or None, optional
        Number of dimensions of kernel to pad. If `None` (default), the number
        of dimensions is inferred from the number of dimensions to
        `kernel_size`.

    Returns
    -------
    padding : tuple
        Amount of padding to apply to each dimension before convolving with
        the kernel in order to preserve the size of input.
    """

    if isinstance(kernel_size, int):
        if kernel_size % 2 == 0:
            raise ValueError("Same padding is not implemented for even kernels sizes.")
    elif any(k % 2 == 0 for k in kernel_size):
        raise ValueError("Same padding is not implemented for even kernels sizes.")

    args = [kernel_size, stride, dilation]

    if all([isinstance(arg, int) for arg in args]):
        return (kernel_size - 1) * dilation // 2

    if ndim is not None:
        pass
    elif isinstance(kernel_size, collections.sequence):
        ndim = len(kernel_size)
    elif isinstance(stride, collections.sequence):
        ndim = len(stride)
    elif isinstance(dilation, collections.sequence):
        ndim = len(dilation)
    else:
        raise ValueError("Wrong argument type. Must be int or iterable.")

    ntuple = _ntuple(ndim)
    kernel_size = ntuple(kernel_size)
    dilation = ntuple(dilation)

    padding = tuple((k - 1) * d // 2 for k, d in zip(kernel_size, dilation))

    return padding


def rint(x, minval=None):
    """
    Rounds and casts as an int, optionally with a floor value.

    Parameters
    ----------
    x : float
        Number to round.
    minval : bool, optional
        A floor value for the output. If `None`, no floor is applied. Default
        is `None`.

    Returns
    -------
    int
        The number rounded to the nearest int, and cast as an int. If `minval`
        is set, the max with `minval` is taken.
    """
    x = int(round(x))
    if minval is not None:
        x = max(minval, x)
    return x


def init_cnn(m):
    """
    Initialise biases and weights for a CNN layer, using a Kaiming normal
    distribution for the weight and 0 for biases.

    Function is applied recursively within the module.

    Parameters
    ----------
    m : torch.nn.Module
        Module
    """
    if getattr(m, "bias", None) is not None:
        nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn._ConvNd, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
    for l in m.children():
        init_cnn(l)
