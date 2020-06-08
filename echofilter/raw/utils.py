"""
Loader utility functions.
"""

import numpy as np
import scipy.interpolate


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
    # Then find the points close to NaNs
    influence = scipy.interpolate.interp1d(
        x, is_nan, bounds_error=bounds_error, **kwargs
    )(x_samples)
    # and remove the points too close to a NaN in the input
    y_samples[influence > nan_threshold] = np.nan
    return y_samples
