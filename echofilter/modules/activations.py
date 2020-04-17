'''
Pytorch activation functions.
'''

import functools

import torch
from torch import nn
import torch.nn.functional as F


__all__ = [
    'str2actfnfactory',
    'InplaceReLU',
    'Swish',
    'MemoryEfficientSwishFunc',
    'MemoryEfficientSwish',
    'HardSwish',
    'mish',
    'Mish',
    'HardMish',
]


def str2actfnfactory(actfn_name):
    '''
    Maps an activation function name to a factory which generates that
    activation function as a `torch.nn.Module` object.

    Parameters
    ----------
    actfn_name : str
        Name of the activation function.

    Returns
    -------
    callable
        A `torch.nn.Module` subclass generator.
    '''
    if hasattr(nn, actfn_name):
        return getattr(nn, actfn_name)

    actfn_name_og = actfn_name
    actfn_name = actfn_name.lower().replace('-', '').replace('_', '')
    if actfn_name == 'inplacerelu' or actfn_name == 'reluinplace':
        return InplaceReLU
    elif actfn_name == 'swish':
        return MemoryEfficientSwish
    elif actfn_name == 'hardswish':
        return HardSwish
    elif actfn_name == 'mish':
        return Mish
    elif actfn_name == 'hardmish':
        return HardMish
    else:
        raise ValueError('Unrecognised activation function: {}'.format(actfn_name_og))


InplaceReLU = functools.partial(
    nn.ReLU,
    inplace=True,
)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MemoryEfficientSwishFunc(torch.autograd.Function):
    '''
    A more compute, less memory, version of Swish.

    Notes
    -----
    https://github.com/lukemelas/EfficientNet-PyTorch
    '''
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return MemoryEfficientSwishFunc.apply(x)


class HardSwish(nn.Module):
    '''
    A second-order approximation to the swish activation function.

    https://arxiv.org/abs/1905.02244
    '''
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace
        self.relu6 = torch.nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return x * self.relu6(x + 3) / 6

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else 'inplace=False'
        return inplace_str


@torch.jit.script
def mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    Notes
    -----
    Mish: A Self Regularized Non-Monotonic Neural Activation Function,
    Diganta Misra
    https://arxiv.org/abs/1908.08681
    https://github.com/digantamisra98/Mish
    '''
    return input * torch.tanh(F.softplus(input))


class Mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    Notes
    -----
    Mish: A Self Regularized Non-Monotonic Neural Activation Function,
    Diganta Misra
    https://arxiv.org/abs/1908.08681
    https://github.com/digantamisra98/Mish
    '''
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return mish(input)


class HardMish(nn.Module):
    '''
    A second-order approximation to the mish activation function.

    Notes
    -----
    https://forums.fast.ai/t/hard-mish-activation-function/59238
    '''
    def __init__(self, inplace=True):
        self.relu5 = nn.Hardtanh(0., 5., inplace)

    def forward(self, x):
        return x * self.relu5(x + 3) / 5

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else 'inplace=False'
        return inplace_str
