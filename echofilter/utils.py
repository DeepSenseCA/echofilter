"""
General utility functions.
"""

import numpy as np
import torch
from torch import nn


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


def first_nonzero(arr, axis=-1, invalid_val=-1):
    """
    Find the index of the first non-zero element in an array.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to search.
    axis : int, optional
        Axis along which to search for a non-zero element. Default is `-1`.
    invalid_val : any, optional
        Value to return if all elements are zero. Default is `-1`.
    """
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def last_nonzero(arr, axis=-1, invalid_val=-1):
    """
    Find the index of the last non-zero element in an array.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to search.
    axis : int, optional
        Axis along which to search for a non-zero element. Default is `-1`.
    invalid_val : any, optional
        Value to return if all elements are zero. Default is `-1`.
    """
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def get_indicator_onoffsets(indicator):
    """
    Find the onsets and offsets of nonzero entries in an indicator.

    Parameters
    ----------
    indicator : 1d numpy.ndarray
        Input vector, which is sometimes zero and sometimes nonzero.

    Returns
    -------
    onsets : list
        Onset indices, where each entry is the start of a sequence of nonzero
        values in the input `indicator`.
    offsets : list
        Offset indices, where each entry is the last in a sequence of nonzero
        values in the input `indicator`, such that
        `indicator[onsets[i] : offsets[i] + 1] != 0`.
    """
    indices = np.nonzero(indicator)[0]

    if len(indices) == 0:
        return [], []

    onsets = [indices[0]]
    offsets = []
    breaks = np.nonzero(indices[1:] - indices[:-1] > 1)[0]
    for break_idx in breaks:
        offsets.append(indices[break_idx])
        onsets.append(indices[break_idx + 1])
    offsets.append(indices[-1])

    return onsets, offsets


def get_current_lr(optimizer):
    """
    Get the learning rate of an optimizer.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        An optimizer, with a learning rate common to all parameter groups.

    Returns
    -------
    float
        The learning rate of the first parameter group.
    """
    return optimizer.param_groups[0]["lr"]


def get_current_momentum(optimizer):
    """
    Get the momentum of an optimizer.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        An optimizer which implements momentum or betas (where momentum is the
        first beta, c.f. torch.optim.Adam) with a momentum common to all
        parameter groups.

    Returns
    -------
    float
        The momentum of the first parameter group.
    """
    if "momentum" not in optimizer.defaults and "betas" not in optimizer.defaults:
        raise ValueError(
            "optimizer {} does not support momentum".format(optimizer.__class__)
        )

    group = optimizer.param_groups[0]
    if "momentum" in group:
        return group["momentum"]
    else:
        return group["betas"][0]


class TensorDict(nn.ParameterDict):
    r"""Holds tensors in a dictionary.

    TensorDict can be indexed like a regular Python dictionary, but implements
    methods such as `to` which operate on all elements within it.

    :class:`~TensorDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~TensorDict.update`, the order of the merged ``OrderedDict``
      or another :class:`~TensorDict` (the argument to :meth:`~TensorDict.update`).

    Note that :meth:`~TensorDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping.

    Arguments:
        parameters (iterable, optional): a mapping (dictionary) of
            (string : :class:`~torch.Tensor`) or an iterable of key-value pairs
            of type (string, :class:`~torch.Tensor`)
    """

    def __init__(self, tensors=None):
        super(TensorDict, self).__init__(tensors)

    def __setitem__(self, key, parameter):
        r"""Adds a tensor to the module.

        The parameter can be accessed as an attribute using given key.

        Args:
            key (string): key of the parameter. The parameter can be accessed
                from this module using the given key
            parameter (Parameter): parameter to be added to the module.
        """
        if "_parameters" not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call"
            )

        elif not isinstance(key, torch._six.string_classes):
            raise TypeError(
                "parameter key should be a string. " "Got {}".format(torch.typekey(key))
            )
        elif "." in key:
            raise KeyError('parameter key can\'t contain "."')
        elif key == "":
            raise KeyError('parameter key can\'t be empty string ""')
        elif hasattr(self, key) and key not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(key))

        if parameter is None:
            self._parameters[key] = None
        else:
            self._parameters[key] = parameter

    def detach(self):
        out = TensorDict()
        for k, p in self._parameters.items():
            out[k] = p.detach()
        return out

    def detach_(self):
        for k, p in self._parameters.items():
            self._parameters[k] = p.detach()
        return self._parameters

    def extra_repr(self):
        child_lines = []
        for k, p in self._parameters.items():
            size_str = "x".join(str(size) for size in p.size())
            device_str = "" if not p.is_cuda else " (GPU {})".format(p.get_device())
            parastr = "{} containing: [{} of size {}{}]".format(
                p.__class__, torch.typename(p.data), size_str, device_str
            )
            child_lines.append("  (" + k + "): " + parastr)
        tmpstr = "\n".join(child_lines)
        return tmpstr
