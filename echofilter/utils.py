"""
General utility functions.
"""

import random

import numpy as np
import torch


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


def worker_seed_fn(worker_id):
    """
    A worker initialization function for `torch.utils.data.DataLoader` objects
    which seeds builtin `random` and `numpy` with `torch.randint` (which is
    stable if torch is manually seeded in the main program).

    Parameters
    ----------
    worker_id : int
        The ID of the worker.
    """
    np.random.seed((torch.randint(0, 4294967296, (1,)).item() + worker_id) % 4294967296)
    random.seed(torch.randint(0, 4294967296, (1,)).item() + worker_id)


def worker_staticseed_fn(worker_id):
    """
    A worker initialization function for `torch.utils.data.DataLoader` objects
    which produces the same seed for builtin `random`, `numpy`, and `torch`
    every time, so it is the same for every epoch.

    Parameters
    ----------
    worker_id : int
        The ID of the worker.
    """
    random.seed(worker_id)
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)
