"""
Pytorch activation functions.

Swish and Mish implementations taken from https://github.com/fastai/fastai2
under the Apache License Version 2.0.
"""

import functools

import torch
from torch import nn
import torch.nn.functional as F


__all__ = [
    "str2actfnfactory",
    "InplaceReLU",
    "swish",
    "Swish",
    "HardSwish",
    "mish",
    "Mish",
    "HardMish",
]


def str2actfnfactory(actfn_name):
    """
    Maps an activation function name to a factory which generates that
    activation function as a :class:`torch.nn.Module` object.

    Parameters
    ----------
    actfn_name : str
        Name of the activation function.

    Returns
    -------
    callable
        A :class:`torch.nn.Module` subclass generator.
    """
    if hasattr(nn, actfn_name):
        return getattr(nn, actfn_name)

    actfn_name_og = actfn_name
    actfn_name = actfn_name.lower().replace("-", "").replace("_", "")
    if actfn_name == "inplacerelu" or actfn_name == "reluinplace":
        return InplaceReLU
    elif actfn_name == "swish":
        return Swish
    elif actfn_name == "hardswish":
        return HardSwish
    elif actfn_name == "mish":
        return Mish
    elif actfn_name == "hardmish":
        return HardMish
    else:
        raise ValueError("Unrecognised activation function: {}".format(actfn_name_og))


InplaceReLU = functools.partial(nn.ReLU, inplace=True)


# Swish
@torch.jit.script
def _swish_jit_fwd(x):
    return x.mul(torch.sigmoid(x))


@torch.jit.script
def _swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))


class _SwishJitAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        return _swish_jit_bwd(x, grad_output)


def swish(x, inplace=False):
    return _SwishJitAutoFn.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return _SwishJitAutoFn.apply(x)


class HardSwish(nn.Module):
    """
    A second-order approximation to the swish activation function.

    See https://arxiv.org/abs/1905.02244
    """

    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace
        self.relu6 = torch.nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return x * self.relu6(x + 3) / 6

    def extra_repr(self):
        inplace_str = "inplace=True" if self.inplace else "inplace=False"
        return inplace_str


# Mish
@torch.jit.script
def _mish_jit_fwd(x):
    return x.mul(torch.tanh(F.softplus(x)))


@torch.jit.script
def _mish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


class MishJitAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        return _mish_jit_bwd(x, grad_output)


def mish(x):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    See https://arxiv.org/abs/1908.08681
    """
    return MishJitAutoFn.apply(x)


class Mish(nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    See https://arxiv.org/abs/1908.08681
    """

    def forward(self, x):
        return MishJitAutoFn.apply(x)


class HardMish(nn.Module):
    """
    A second-order approximation to the mish activation function.

    Notes
    -----
    https://forums.fast.ai/t/hard-mish-activation-function/59238
    """

    def __init__(self, inplace=True):
        self.relu5 = nn.Hardtanh(0.0, 5.0, inplace)

    def forward(self, x):
        return x * self.relu5(x + 3) / 5

    def extra_repr(self):
        inplace_str = "inplace=True" if self.inplace else "inplace=False"
        return inplace_str
