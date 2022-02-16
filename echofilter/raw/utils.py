"""
Loader utility functions.
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

import numpy as np
import scipy.interpolate

from ..utils import get_indicator_onoffsets, mode


def interp1d_preserve_nan(
    x, y, x_samples, nan_threshold=0.0, bounds_error=False, **kwargs
):
    """
    Interpolate a 1-D function, preserving NaNs.

    `x` and `y` are arrays of values used to approximate some function f:
    ``y = f(x)``.  We exclude NaNs for the interpolation and then mask out
    entries which are adjacent (or close to) a NaN in the input.

    Parameters
    ----------
    x : (N,) array_like
        A 1-D array of real values. Must not contain NaNs.
    y : (...,N,...) array_like
        A N-D array of real values. The length of `y` along the interpolation
        axis must be equal to the length of `x`. May contain NaNs.
    x_samples : array_like
        A 1-D array of real values at which the interpolation function will
        be sampled.
    nan_threshold : float, optional
        Minimum amount of influence a NaN must have on an output sample for it
        to become a NaN. Default is `0.` i.e. any influence.
    bounds_error : bool, optional
        If `True`, a ValueError is raised any time interpolation is attempted
        on a value outside of the range of `x` (where extrapolation is
        necessary). If `False` (default), out of bounds values are assigned
        value `fill_value` (whose default is NaN).
    **kwargs
        Additional keyword arguments are as per :meth:`scipy.interpolate.interp1d`.

    Returns
    -------
    y_samples : (...,N,...) np.ndarray
        The result of interpolating, with sample points close to NaNs in the
        input returned as NaN.
    """
    # First, run with NaNs masked out.
    is_nan = np.isnan(y)
    if np.sum(~is_nan) < 2 and np.ndim(y) == 1:
        y_samples = np.empty(x_samples.shape, dtype=y.dtype)
        y_samples[:] = np.nan
        return y_samples
    y_samples = scipy.interpolate.interp1d(
        x[~is_nan], y[~is_nan], bounds_error=bounds_error, **kwargs
    )(x_samples)
    if np.sum(is_nan) == 0:
        # Shortcut if there are no NaNs
        return y_samples
    # Then find the points close to NaNs
    influence = scipy.interpolate.interp1d(
        x, is_nan, bounds_error=bounds_error, **kwargs
    )(x_samples)
    # and remove the points too close to a NaN in the input
    y_samples[influence > nan_threshold] = np.nan
    return y_samples


def pad1d(array, pad_width, axis=0, **kwargs):
    """
    Pad an array along a single axis only.

    Parameters
    ----------
    array : numpy.ndarary
        Array to be padded.
    pad_width : int or tuple
        The amount to pad, either a length two tuple of values for each edge,
        or an int if the padding should be the same for each side.
    axis : int, optional
        The axis to pad. Default is `0`.
    **kwargs
        As per :meth:`numpy.pad`.

    Returns
    -------
    numpy.ndarary
        Padded array.

    See also
    --------
    numpy.pad
    """
    pads = [(0, 0) for _ in range(array.ndim)]
    if hasattr(pad_width, "__len__"):
        pads[axis] = pad_width
    else:
        pads[axis] = (pad_width, pad_width)
    return np.pad(array, pads, **kwargs)


def medfilt1d(signal, kernel_size, axis=-1, pad_mode="reflect"):
    """
    Median filter in 1d, with support for selecting padding mode.

    Parameters
    ----------
    signal : array_like
        The signal to filter.
    kernel_size
        Size of the median kernel to use.
    axis : int, optional
        Which axis to operate along. Default is `-1`.
    pad_mode : str, optional
        Method with which to pad the vector at the edges.
        Must be supported by :meth:`numpy.pad`. Default is `"reflect"`.

    Returns
    -------
    filtered : array_like
        The filtered signal.

    See also
    --------
    - `scipy.signal.medfilt`
    - `pad1d`
    """
    offset = kernel_size // 2
    signal = pad1d(signal, offset, axis=axis, mode=pad_mode)
    filtered = scipy.signal.medfilt(signal, kernel_size)[offset:-offset]
    return filtered


def squash_gaps(mask, max_gap_squash, axis=-1, inplace=False):
    """
    Merge small gaps between zero values in a boolean array.

    Parameters
    ----------
    mask : boolean array
        The input mask, with small gaps between zero values which will be
        squashed with zeros.
    max_gap_squash : int
        Maximum length of gap to squash.
    axis : int, optional
        Axis on which to operate. Default is `-1`.
    inplace : bool, optional
        Whether to operate on the original array. If `False`, a copy is
        created and returned.

    Returns
    -------
    merged_mask : boolean array
        Mask as per the input, but with small gaps squashed.
    """
    if not inplace:
        mask = mask.copy()
    L = mask.shape[axis]
    for i in range(min(max_gap_squash, L - 1), 1, -1):
        check = np.stack(
            [
                pad1d(
                    mask.take(range(i // 2, L), axis=axis),
                    (0, i // 2),
                    axis=axis,
                    mode="constant",
                ),
                pad1d(
                    mask.take(range(0, L - ((i + 1) // 2)), axis=axis),
                    ((i + 1) // 2, 0),
                    axis=axis,
                    mode="constant",
                ),
            ]
        )
        li = ~np.any(check, axis=0)
        mask[li] = 0
    return mask


def integrate_area_of_contour(x, y, closed=None, preserve_sign=False):
    """
    Compute the area within a contour, using Green's algorithm.

    Parameters
    ----------
    x : array_like vector
        x co-ordinates of nodes along the contour.
    y : array_like vector
        y co-ordinates of nodes along the contour.
    closed : bool or None, optional
        Whether the contour is already closed. If `False`, it will be closed
        before deterimining the area. If `None` (default), it is automatically
        determined as to whether the contour is already closed, and is closed
        if necessary.
    preserve_sign : bool, optional
        Whether to preserve the sign of the area. If `True`, the area is
        positive if the contour is anti-clockwise and negative if it is
        clockwise oriented. Default is `False`, which always returns a positive
        area.

    Returns
    -------
    area : float
        The integral of the area witihn the contour.

    Notes
    -----
    https://en.wikipedia.org/wiki/Green%27s_theorem#Area_calculation
    """
    if closed is None:
        closed = x[0] == x[-1] and y[0] == y[-1]
    if not closed:
        x = np.concatenate([x, x[[0]]])
        y = np.concatenate([y, y[[0]]])
    # Integrate to find the area
    A = 0.5 * np.sum(y[:-1] * np.diff(x) - x[:-1] * np.diff(y))
    # Take the abs in case the curve was clockwise instead of anti-clockwise
    A = np.abs(A)
    return A
