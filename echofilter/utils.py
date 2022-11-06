"""
General utility functions.
"""

# This file is part of Echofilter.
#
# Copyright (C) 2020-2022  Scott C. Lowe and Offshore Energy Research Association (OERA)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import contextlib

import numpy as np
import scipy.stats


def mode(a, axis=None, keepdims=False, **kwargs):
    """
    Return an array of the modal (most common) value in the passed array.

    If there is more than one such value, only the smallest is returned.

    Parameters
    ----------
    a : array_like
        n-dimensional array of which to find mode(s).
    axis : int or None, optional
        Axis or axes along which the mode is computed. The default,
        `axis=None`, will sum all of the elements of the input array.
        If axis is negative it counts from the last to the first axis.
    keepdims : bool, optional
        If this is set to `True`, the axes which are reduced are left
        in the result as dimensions with size one. With this option, the result
        will broadcast correctly against the input array. Default is `False`.
    **kwargs
        Additional arguments as per :meth:`scipy.stats.mode`.

    Returns
    -------
    mode_along_axis : numpy.ndarray
        An array with the same shape as `a`, with the specified axis removed.
        If `keepdims=True` and either `a` is a 0-d array or `axis` is None,
        a scalar is returned.

    See also
    --------
    scipy.stats.mode
    """
    m = scipy.stats.mode(a, axis=axis, **kwargs)[0]
    if keepdims:
        return m
    m = m.squeeze(0 if axis is None else axis)
    if m.size == 1:
        m = np.asscalar(m)
    return m


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
        ``indicator[onsets[i] : offsets[i] + 1] != 0``.
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
