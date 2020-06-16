"""
Loader utility functions.
"""

import numpy as np
import scipy.interpolate
import scipy.stats

from ..utils import get_indicator_onoffsets


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
        Additional keyword arguments are as per `scipy.interpolate.interp1d`.

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
        As per `numpy.pad`.

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
    for i in range(max_gap_squash, 1, -1):
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
