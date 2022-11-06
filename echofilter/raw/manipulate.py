"""
Manipulating lines and masks contained in Echoview files.
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

import copy
import os
import warnings

import numpy as np
import scipy.interpolate
import scipy.ndimage

from . import loader
from . import metadata
from . import utils


ROOT_DATA_DIR = loader.ROOT_DATA_DIR


def find_passive_data(signals, n_depth_use=38, threshold=25.0, deviation=None):
    """
    Find segments of Sv recording which correspond to passive recording.

    Parameters
    ----------
    signals : array_like
        Two-dimensional array of Sv values, shaped `[timestamps, depths]`.
    n_depth_use : int, optional
        How many Sv depths to use, starting with the first depths (closest
        to the sounder device). If `None` all depths are used. Default is `38`.
    threshold : float, optional
        Threshold for start/end of passive regions. Default is `25`.
    deviation : float, optional
        Threshold for start/end of passive regions is `deviation` times the
        interquartile-range of the difference between samples at neigbouring
        timestamps. Default is `None`. Only one of `threshold` and `deviation`
        should be set.

    Returns
    -------
    passive_start : numpy.ndarray
        Indices of rows of `signals` at which passive segments start.
    passive_end : numpy.ndarray
        Indices of rows of `signals` at which passive segments end.

    Notes
    -----
    Works by looking at the difference between consecutive recordings and
    finding large deviations.
    """
    # Ensure signals is numpy array
    signals = np.asarray(signals)

    if n_depth_use is None:
        n_depth_use = signals.shape[1]

    md = np.median(np.diff(signals[:, :n_depth_use], axis=0), axis=1)

    if threshold is not None and deviation is not None:
        raise ValueError("Only one of `threshold` and `deviation` should be set.")
    if threshold is None:
        if deviation is None:
            raise ValueError("Neither of `threshold` and `deviation` were set.")
        threshold = (np.percentile(md, 75) - np.percentile(md, 25)) * deviation

    threshold_high = threshold
    threshold_low = -threshold
    indices_possible_start = np.nonzero(md < threshold_low)[0]
    indices_possible_end = np.nonzero(md > threshold_high)[0]

    current_index = 0
    indices_passive_start = []
    indices_passive_end = []

    if len(indices_possible_start) == 0 and len(indices_possible_end) == 0:
        return np.array(indices_passive_start), np.array(indices_passive_end)

    if len(indices_possible_start) > 0:
        indices_possible_start += 1

    if len(indices_possible_end) > 0:
        indices_possible_end += 1

    if len(indices_possible_end) > 0 and (
        len(indices_possible_start) == 0
        or indices_possible_end[0] < indices_possible_start[0]
    ):
        indices_passive_start.append(0)
        current_index = indices_possible_end[0]
        indices_passive_end.append(current_index)
        indices_possible_start = indices_possible_start[
            indices_possible_start > current_index
        ]
        indices_possible_end = indices_possible_end[
            indices_possible_end > current_index
        ]

    while len(indices_possible_start) > 0:
        current_index = indices_possible_start[0]
        indices_passive_start.append(current_index)
        baseline = signals[current_index - 1, :n_depth_use]

        # Find first column which returns to the baseline value seen before passive region
        offsets = np.nonzero(
            np.median(baseline - signals[current_index:, :n_depth_use], axis=1)
            < threshold_high
        )[0]
        if len(offsets) == 0:
            current_index = signals.shape[0]
        else:
            current_index += offsets[0]
        indices_passive_end.append(current_index)

        # Remove preceding indices from the list of candidates
        indices_possible_start = indices_possible_start[
            indices_possible_start > current_index
        ]
        indices_possible_end = indices_possible_end[
            indices_possible_end > current_index
        ]

        # Check the start was sufficiently inclusive
        if current_index < signals.shape[0]:
            baseline = signals[current_index, :n_depth_use]
            nonpassives = np.nonzero(
                np.median(baseline - signals[:current_index, :n_depth_use], axis=1)
                < threshold_high
            )[0]
            if len(nonpassives) == 0:
                indices_passive_start[-1] = 0
            else:
                indices_passive_start[-1] = min(
                    indices_passive_start[-1],
                    nonpassives[-1] + 1,
                )

        # Combine with preceding passive segments if they overlap
        while (
            len(indices_passive_start) > 1
            and indices_passive_start[-1] <= indices_passive_end[-2]
        ):
            indices_passive_start = indices_passive_start[:-1]
            indices_passive_end = indices_passive_end[:-2] + indices_passive_end[-1:]

    return np.array(indices_passive_start), np.array(indices_passive_end)


def find_passive_data_v2(
    signals,
    n_depth_use=38,
    threshold_inner=None,
    threshold_init=None,
    deviation=None,
    sigma_depth=0,
    sigma_time=1,
):
    """
    Find segments of Sv recording which correspond to passive recording.

    Parameters
    ----------
    signals : array_like
        Two-dimensional array of Sv values, shaped `[timestamps, depths]`.
    n_depth_use : int, optional
        How many Sv depths to use, starting with the first depths (closest
        to the sounder device). If `None` all depths are used. Default is `38`.
        The median is taken across the depths, after taking the temporal
        derivative.
    threshold_inner : float, optional
        Theshold to apply to the temporal derivative of the signal when
        detected fine-tuned start/end of passive regions.
        Default behaviour is to use a threshold automatically determined using
        `deviation` if it is set, and otherwise use a threshold of `35.0`.
    threshold_init : float, optional
        Theshold to apply during the initial scan of the start/end of passive
        regions, which seeds the fine-tuning search.
        Default behaviour is to use a threshold automatically determined using
        `deviation` if it is set, and otherwise use a threshold of `12.0`.
    deviation : float, optional
        Set `threshold_inner` to be `deviation` times the standard deviation of
        the temporal derivative of the signal. The standard deviation is
        robustly estimated based on the interquartile range.
        If this is set, `threshold_inner` must not be `None`.
        Default is `None`
    sigma_depth : float, optional
        Width of kernel for filtering signals across second dimension (depth).
        Default is `0` (no filter).
    sigma_time : float, optional
        Width of kernel for filtering signals across second dimension (time).
        Default is `1`. Set to `0` to not filter.

    Returns
    -------
    passive_start : numpy.ndarray
        Indices of rows of `signals` at which passive segments start.
    passive_end : numpy.ndarray
        Indices of rows of `signals` at which passive segments end.

    Notes
    -----
    Works by looking at the difference between consecutive recordings and
    finding large deviations.
    """
    # Ensure signals is numpy array
    signals = np.asarray(signals)

    if n_depth_use is None:
        n_depth_use = signals.shape[1]

    if sigma_depth > 0:
        signals_smooth = scipy.ndimage.gaussian_filter1d(
            signals.astype(np.float32), sigma_depth, axis=-1
        )
    else:
        signals_smooth = signals

    md_inner = np.median(np.diff(signals_smooth[:, :n_depth_use], axis=0), axis=1)

    if sigma_time > 0:
        signals_init = scipy.ndimage.gaussian_filter1d(
            signals_smooth.astype(np.float32), sigma_time, axis=0
        )
        md_init = np.median(np.diff(signals_init[:, :n_depth_use], axis=0), axis=1)
    else:
        signals_init = signals
        md_init = md_inner

    if threshold_inner is not None and deviation is not None:
        raise ValueError("Only one of `threshold_inner` and `deviation` should be set.")
    if threshold_init is None:
        if deviation is None:
            threshold_init = 12.0
        else:
            threshold_inner = (
                (np.percentile(md_init, 75) - np.percentile(md_init, 25))
                / 1.35
                * deviation
            )
    if threshold_inner is None:
        if deviation is None:
            threshold_inner = 35.0
        else:
            threshold_inner = (
                (np.percentile(md_inner, 75) - np.percentile(md_inner, 25))
                / 1.35
                * deviation
            )

    threshold_high_inner = threshold_inner
    threshold_low_inner = -threshold_inner
    threshold_high_init = threshold_init
    threshold_low_init = -threshold_init
    indices_possible_start_init = np.nonzero(md_init < threshold_low_init)[0]
    indices_possible_end_init = np.nonzero(md_init > threshold_high_init)[0]

    if len(indices_possible_start_init) == 0 and len(indices_possible_end_init) == 0:
        return np.array([]), np.array([])

    # Fine tune indices without smoothing
    indices_possible_start = []
    indices_possible_end = []

    capture_start = None
    for i, index_p in enumerate(indices_possible_start_init):
        if capture_start is None:
            capture_start = index_p
        if (
            i + 1 >= len(indices_possible_start_init)
            or indices_possible_start_init[i + 1] > index_p + 3
        ):
            # break capture
            capture_end = index_p
            capture = np.arange(capture_start, capture_end + 1)
            indices_possible_start.append(capture[np.argmin(md_init[capture])])
            capture_start = None

    capture_start = None
    for i, index_p in enumerate(indices_possible_end_init):
        if capture_start is None:
            capture_start = index_p
        if (
            i + 1 >= len(indices_possible_end_init)
            or indices_possible_end_init[i + 1] > index_p + 3
        ):
            # break capture
            capture_end = index_p
            capture = np.arange(capture_start, capture_end + 1)
            indices_possible_end.append(capture[np.argmax(md_init[capture])])
            capture_start = None

    indices_possible_start = np.array(indices_possible_start)
    indices_possible_end = np.array(indices_possible_end)

    current_index = 0
    indices_passive_start = []
    indices_passive_end = []

    if len(indices_possible_start) > 0:
        indices_possible_start += 1

    if len(indices_possible_end) > 0:
        indices_possible_end += 1

    if len(indices_possible_end) > 0 and (
        len(indices_possible_start) == 0
        or indices_possible_end[0] < indices_possible_start[0]
    ):
        indices_passive_start.append(0)
        current_index = indices_possible_end[0]
        indices_passive_end.append(current_index)
        indices_possible_start = indices_possible_start[
            indices_possible_start > current_index
        ]
        indices_possible_end = indices_possible_end[
            indices_possible_end > current_index
        ]

    while len(indices_possible_start) > 0:
        current_index = indices_possible_start[0]
        indices_passive_start.append(current_index)
        baseline_index = max(0, current_index - 2)
        baseline = signals[baseline_index, :n_depth_use]

        # Find first column which returns to the baseline value seen before passive region
        offsets = np.nonzero(
            np.median(baseline - signals[current_index:, :n_depth_use], axis=1)
            < threshold_high_inner
        )[0]
        if len(offsets) == 0:
            current_index = signals.shape[0]
        else:
            current_index += offsets[0]
        indices_passive_end.append(current_index)

        # Remove preceding indices from the list of candidates
        indices_possible_start = indices_possible_start[
            indices_possible_start > current_index
        ]
        indices_possible_end = indices_possible_end[
            indices_possible_end > current_index
        ]

        # Check the start was sufficiently inclusive
        if current_index < signals.shape[0]:
            baseline_index = min(signals.shape[0] - 1, current_index + 1)
            baseline = signals[baseline_index, :n_depth_use]
            nonpassives = np.nonzero(
                np.median(baseline - signals[:current_index, :n_depth_use], axis=1)
                < threshold_high_inner
            )[0]
            if len(nonpassives) == 0:
                indices_passive_start[-1] = 0
            else:
                indices_passive_start[-1] = min(
                    indices_passive_start[-1],
                    nonpassives[-1] + 1,
                )

        # Combine with preceding passive segments if they overlap
        while (
            len(indices_passive_start) > 1
            and indices_passive_start[-1] <= indices_passive_end[-2]
        ):
            indices_passive_start = indices_passive_start[:-1]
            indices_passive_end = indices_passive_end[:-2] + indices_passive_end[-1:]

    return np.array(indices_passive_start), np.array(indices_passive_end)


def make_lines_from_mask(mask, depths=None, max_gap_squash=1.0):
    """
    Determines turbulence and bottom lines for a mask array.

    Parameters
    ----------
    mask : array_like
        A two-dimensional logical array, where for each row dimension 1 takes
        the value `False` for some unknown continuous stretch at the start and
        end of the column, with `True` values between these two masked-out
        regions.
    depths : array_like, optional
        Depth of each sample point along dim 1 of `mask`. Must be either
        monotonically increasing or monotonically decreasing. Default is the
        index of `mask`, `arange(mask.shape[1])`.
    max_gap_squash : float, optional
        Maximum gap to merge together, in metres. Default is `1.`.

    Returns
    -------
    d_turbulence : numpy.ndarray
        Depth of turbulence line. This is the line of smaller depth which
        separates the `False` region of `mask` from the central region of
        `True` values. (If `depths` is monotonically increasing, this is
        for the start of the columns of `mask`, otherwise it is at the end.)
    d_bottom : numpy.ndarray
        Depth of bottom line. As for `d_turbulence`, but for the other end of the
        array.
    """
    # Ensure input is an array. Make a copy, so we don't modify the input.
    mask = np.array(mask, copy=True)

    # Autocomplete depth with index.
    if depths is None:
        depths = np.arange(mask.shape[1])
    depths = np.asarray(depths)
    if len(depths) != mask.shape[1]:
        raise ValueError("Length of depths input must match dim 1 of mask.")
    depth_intv = np.median(np.diff(depths))

    # Merge small gaps between masked out data, so the middle is masked out too
    # We merge all gaps smaller than 120 pixels apart.
    max_gap_squash_idx = int(np.round(max_gap_squash / depth_intv))
    mask = utils.squash_gaps(mask, max_gap_squash_idx)

    # Check which depths were removed for each timestamp
    nonremoved_depths = np.tile(depths, (mask.shape[0], 1)).astype("float")
    nonremoved_depths[~mask] = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN (slice|axis) encountered")
        # Top line is the smallest non-removed depth at each timepoint.
        # We offset by depth_intv / 2 to get a depth in between the last kept
        # value at the top and the first removed value.
        d_turbulence = np.nanmin(nonremoved_depths, axis=1) - depth_intv / 2
        # Bottom line is the largest non-removed depth at each timepoint,
        # offset similarly.
        d_bottom = np.nanmax(nonremoved_depths, axis=1) + depth_intv / 2

    return d_turbulence, d_bottom


def make_lines_from_masked_csv(fname):
    """
    Load a masked csv file output from Echoview and generate lines which
    reproduce the mask.

    Parameters
    ----------
    fname : str
        Path to file containing masked Echoview output data in csv format.

    Returns
    -------
    timestamps : numpy.ndarray
        Sample timestamps.
    d_turbulence : numpy.ndarray
        Depth of turbulence line.
    d_bottom : numpy.ndarray
        Depth of bottom line.
    """
    # Load the masked data
    timestamps, depths, signals_mskd = loader.transect_loader(fname)
    mask = ~np.isnan(signals_mskd)
    d_turbulence, d_bottom = make_lines_from_mask(mask, depths)
    return timestamps, d_turbulence, d_bottom


def write_lines_for_masked_csv(fname_mask, fname_turbulence=None, fname_bottom=None):
    """
    Write new turbulence and bottom lines based on csv containing masked Echoview
    output.

    Parameters
    ----------
    fname_mask : str
        Path to input file containing masked Echoview output data in csv
        format.
    fname_turbulence : str, optional
        Destination of generated turbulence line, written in evl format. If `None`
        (default), the output name is `<fname_base>_mask-turbulence.evl`, where
        `<fname_base>` is `fname_mask` without extension and without any
        occurence of the substrings `_Sv_raw` or `_Sv` in the base file name.
    fname_bottom : str
        Destination of generated bottom line, written in evl format. If `None`
        (default), the output name is `<fname_base>_mask-bottom.evl`.
    """
    if fname_turbulence is None or fname_bottom is None:
        fname_base = os.path.splitext(fname_mask)[0]
        dirname, fname_base = os.path.split(fname_base)
        fname_base = fname_base.replace("_Sv_raw", "").replace("_Sv", "")
        fname_base = os.path.join(dirname, fname_base)
    if fname_turbulence is None:
        fname_turbulence = fname_base + "_mask-turbulence.evl"
    if fname_bottom is None:
        fname_bottom = fname_base + "_mask-bottom.evl"
    # Generate the new lines.
    timestamps, d_turbulence, d_bottom = make_lines_from_masked_csv(fname_mask)
    # Write the new lines to their output files.
    loader.evl_writer(fname_turbulence, timestamps, d_turbulence)
    loader.evl_writer(fname_bottom, timestamps, d_bottom)


def find_nonzero_region_boundaries(v):
    """
    Find the start and end indices for nonzero regions of a vector.

    Parameters
    ----------
    v : array_like
        A vector.

    Returns
    -------
    starts : numpy.ndarray
        Indices for start of regions of nonzero elements in vector `v`
    ends : numpy.ndarray
        Indices for end of regions of nonzero elements in vector `v`
        (exclusive).

    Notes
    -----
    For `i` in `range(len(starts))`, the set of values `v[starts[i]:ends[i]]`
    are nonzero. Values in the range `v[ends[i]:starts[i+1]]` are zero.
    """

    v = np.asarray(v)
    v = v != 0
    v = v.astype(np.float)

    starts = np.nonzero(np.diff(v) > 0)[0] + 1
    ends = np.nonzero(np.diff(v) < 0)[0] + 1

    if v[0]:
        starts = np.r_[0, starts]

    if v[-1]:
        ends = np.r_[ends, len(vector)]

    return starts, ends


def fixup_lines(
    timestamps,
    depths,
    mask,
    t_turbulence=None,
    d_turbulence=None,
    t_bottom=None,
    d_bottom=None,
):
    """
    Extend existing turbulence/bottom lines based on masked target Sv output.

    Parameters
    ----------
    timestamps : array_like
        Shaped `(num_timestamps, )`.
    depths : array_like
        Shaped `(num_depths, )`.
    mask : array_like
        Boolean array, where `True` denotes kept entries.
        Shaped `(num_timestamps, num_depths)`.
    t_turbulence : array_like, optional
        Sampling times for existing turbulence line.
    d_turbulence : array_like, optional
        Depth of existing turbulence line.
    t_bottom : array_like, optional
        Sampling times for existing bottom line.
    d_bottom : array_like, optional
        Depth of existing bottom line.

    Returns
    -------
    d_turbulence_new : numpy.ndarray
        Depth of new turbulence line.
    d_bottom_new : numpy.ndarray
        Depth of new bottom line.
    """
    # Handle different sampling grids
    if d_turbulence is not None:
        if t_turbulence is None:
            raise ValueError(
                "t_turbulence must be provided if d_turbulence is provided"
            )
        d_turbulence = np.interp(timestamps, t_turbulence, d_turbulence)

    if d_bottom is not None:
        if t_bottom is None:
            raise ValueError("t_bottom must be provided if d_bottom is provided")
        d_bottom = np.interp(timestamps, t_bottom, d_bottom)

    # Generate fresh lines corresponding to said mask
    d_turbulence_new, d_bottom_new = make_lines_from_mask(mask, depths)

    # Ensure nans in the lines are replaced with prior values, if possible
    li = np.isnan(d_turbulence_new)
    if d_turbulence is not None:
        d_turbulence_new[li] = d_turbulence[li]
    elif np.any(~li):
        d_turbulence_new[li] = np.interp(
            timestamps[li],
            timestamps[~li],
            d_turbulence_new[~li],
        )
    li = np.isnan(d_bottom_new)
    if d_bottom is not None:
        d_bottom_new[li] = d_bottom[li]
    elif np.any(~li):
        d_bottom_new[li] = np.interp(
            timestamps[li],
            timestamps[~li],
            d_bottom_new[~li],
        )

    # Ensure that the lines cover at least as much material as they did before
    if d_turbulence is not None:
        d_turbulence_new = np.maximum(d_turbulence, d_turbulence_new)
    if d_bottom is not None:
        d_bottom_new = np.minimum(d_bottom, d_bottom_new)

    # This mask can't handle regions where all the data was removed.
    # Find those and replace them with the original lines, if they were
    # provided. If they weren't, interpolate to fill the holes (if there is
    # something to interpolate).
    all_removed = ~np.any(mask, axis=1)
    any_all_removed = np.any(all_removed)
    everything_removed = np.all(all_removed)
    if not any_all_removed:
        pass
    elif d_turbulence is not None:
        d_turbulence_new[all_removed] = d_turbulence[all_removed]
    elif ~everything_removed:
        d_turbulence_new[all_removed] = np.interp(
            timestamps[all_removed],
            timestamps[~all_removed],
            d_turbulence_new[~all_removed],
        )
    else:
        d_turbulence_new[all_removed] = np.nan

    if not any_all_removed:
        pass
    elif d_bottom is not None:
        d_bottom_new[all_removed] = d_bottom[all_removed]
    elif ~everything_removed:
        d_bottom_new[all_removed] = np.interp(
            timestamps[all_removed],
            timestamps[~all_removed],
            d_bottom_new[~all_removed],
        )
    else:
        d_bottom_new[all_removed] = np.nan

    return d_turbulence_new, d_bottom_new


def remove_anomalies_1d(
    signal, thr=5, thr2=4, kernel=201, kernel2=31, return_filtered=False
):
    """
    Remove anomalies from a temporal signal.

    Applies a median filter to the data, and replaces datapoints which
    deviate from the median filtered signal by more than some threshold
    with the median filtered data. This process is repeated until no
    datapoints deviate from the filtered line by more than the threshold.

    Parameters
    ----------
    signal : array_like
        The signal to filter.
    thr : float, optional
        The initial threshold will be `thr` times the standard deviation of the
        residuals. The standard deviation is robustly estimated from the
        interquartile range. Default is `5`.
    thr2 : float, optional
        The threshold for repeated iterations will be `thr2` times the standard
        deviation of the remaining residuals. The standard deviation is
        robustly estimated from interdecile range. Default is `4`.
    kernel : int, optional
        The kernel size for the initial median filter. Default is `201`.
    kernel2 : int, optional
        The kernel size for subsequent median filters. Default is `31`.
    return_filtered : bool, optional
        If `True`, the median filtered signal is also returned.
        Default is `False`.

    Returns
    -------
    signal : numpy.ndarray like signal
        The input signal with anomalies replaced with median values.
    is_replaced : bool numpy.ndarray shaped like signal
        Indicator for which datapoints were replaced.
    filtered : numpy.ndarray like signal, optional
        The final median filtered signal. Returned if `return_filtered=True`.

    See also
    --------
    `echofilter.raw.utils.medfilt1d`
    """
    signal = np.copy(signal)

    # Median filtering, with reflection padding
    filtered = utils.medfilt1d(signal, kernel)
    # Measure the residual between the original and median filtered signal
    residual = signal - filtered
    # Replace datapoints more than thr sigma away from the median filter
    # with the filtered signal. We use a robust estimate of the standard
    # deviation, using the central 50% of datapoints.
    stdev = np.diff(np.percentile(residual, [25, 75])).item() / 1.35
    is_replaced = np.abs(residual) > thr * stdev
    signal[is_replaced] = filtered[is_replaced]

    # Filter again, with a narrower kernel but tighter threshold
    while True:
        filtered = utils.medfilt1d(signal, kernel2)
        # Mesure new residual
        residual = signal - filtered
        # Make sure to only include original data points when determining
        # the standard deviation. We use the interdecile range.
        stdev = np.diff(np.percentile(residual[~is_replaced], [10, 90])).item() / 2.56
        is_replaced_now = np.abs(residual) > thr2 * stdev
        is_replaced |= is_replaced_now
        signal[is_replaced] = filtered[is_replaced]
        # We are done when no more datapoints had to be replaced
        if not np.any(is_replaced_now):
            break

    if return_filtered:
        return signal, is_replaced, filtered
    return signal, is_replaced


def fix_surface_line(timestamps, d_surface, is_passive):
    """
    Fix anomalies in the surface line.

    Parameters
    ----------
    timestamps : array_like sized (N, )
        Timestamps for each ping.
    d_surface : array_like sized (N, )
        Surface line depths.
    is_passive : array_like sized (N, )
        Indicator for passive data. Values for the surface line during passive
        data collection will not be used.

    Returns
    -------
    fixed_surface : numpy.ndarray
        Surface line depths, with anomalies replaced with median filtered
        values and passive data replaced with linear interpolation.
        Has the same size and dtype as `d_surface`.
    is_replaced : boolean numpy.ndarray sized (N, )
        Indicates which datapoints were replaced. Note that passive data is
        always replaced and is marked as such.
    """
    # Ensure is_passive is a boolean array
    is_passive = is_passive > 0.5

    # Initialise
    out_timestamps = []
    out_surface = []
    out_is_replaced = []
    out_filtered = []

    # Process each segment separately. We divide up into segments after
    # removing any passive data.
    for segment in split_transect(
        timestamps[~is_passive], d_surface=d_surface[~is_passive]
    ):
        fixed, is_replaced, filtered = remove_anomalies_1d(
            segment["d_surface"], return_filtered=True
        )
        out_timestamps.append(segment["d_surface"])
        out_surface.append(fixed)
        out_is_replaced.append(is_replaced)
        out_filtered.append(filtered)

    # Merge segments into a single array for each output
    out_timestamps = np.concatenate(out_timestamps)
    out_surface = np.concatenate(out_surface)
    out_is_replaced = np.concatenate(out_is_replaced)
    out_filtered = np.concatenate(out_filtered)

    # Add back datapoints during passive by interpolating the filtered signal
    # over time
    fixed_surface = np.zeros_like(d_surface)
    fixed_surface[~is_passive] = out_surface
    fixed_surface[is_passive] = np.interp(
        timestamps[is_passive], timestamps[~is_passive], out_filtered
    )
    # Include indication that passive data was always replaced as well as other
    # datapoints
    is_replaced = np.ones_like(is_passive)
    is_replaced[~is_passive] = out_is_replaced

    return fixed_surface, is_replaced


def load_decomposed_transect_mask(sample_path):
    """
    Loads a raw and masked transect and decomposes the mask into turbulence and bottom
    lines, and passive and removed regions.

    Parameters
    ----------
    sample_path : str
        Path to sample, without extension. The raw data should be located at
        ``sample_path + "_Sv_raw.csv"``.

    Returns
    -------
    dict
        A dictionary with keys:

            - "timestamps" : numpy.ndarray
                Timestamps (in seconds since Unix epoch), for each recording
                timepoint.
            - "depths" : numpy.ndarray
                Depths from the surface (in metres), with each entry
                corresponding to each column in the `signals` data.
            - "Sv" : numpy.ndarray
                Echogram Sv data, shaped (num_timestamps, num_depths).
            - "mask" : numpy.ndarray
                Logical array indicating which datapoints were kept (`True`)
                and which removed (`False`) for the masked Sv output.
                Shaped (num_timestamps, num_depths).
            - "turbulence" : numpy.ndarray
                For each timepoint, the depth of the shallowest datapoint which
                should be included for the mask. Shaped (num_timestamps, ).
            - "bottom" : numpy.ndarray
                For each timepoint, the depth of the deepest datapoint which
                should be included for the mask. Shaped (num_timestamps, ).
            - "is_passive" : numpy.ndarray
                Logical array showing whether a timepoint is of passive data.
                Shaped (num_timestamps, ). All passive recording data should
                be excluded by the mask.
            - "is_removed" : numpy.ndarray
                Logical array showing whether a timepoint is entirely removed
                by the mask. Shaped (num_timestamps, ). Does not include
                periods of passive recording.
            - "is_upward_facing" : bool
                Indicates whether the recording source is located at the
                deepest depth (i.e. the seabed), facing upwards. Otherwise, the
                recording source is at the shallowest depth (i.e. the surface),
                facing downwards.
    """

    # Load raw data
    fname_raw = os.path.join(sample_path + "_Sv_raw.csv")
    fname_masked = os.path.join(sample_path + "_Sv.csv")

    ts_raw, depths_raw, signals_raw = loader.transect_loader(fname_raw)
    ts_mskd, depths_mskd, signals_mskd = loader.transect_loader(fname_masked)
    mask = ~np.isnan(signals_mskd)

    # Determine whether depths are ascending or descending
    is_upward_facing = depths_raw[-1] < depths_raw[0]

    fname_turbulence1 = sample_path + "_turbulence.evl"
    fname_turbulence2 = sample_path + "_air.evl"
    fname_bottom = sample_path + "_bottom.evl"
    fname_surface = sample_path + "_surface.evl"

    if os.path.isfile(fname_turbulence1):
        fname_turbulence = fname_turbulence1
        if os.path.isfile(fname_turbulence2):
            raise ValueError(
                "Only one of {} and {} should exist.".format(
                    fname_turbulence1, fname_turbulence2
                )
            )
    elif os.path.isfile(fname_turbulence2):
        fname_turbulence = fname_turbulence2
    else:
        raise ValueError(
            "Neither {} nor {} were found.".format(fname_turbulence1, fname_turbulence2)
        )
    t_turbulence, d_turbulence = loader.evl_loader(fname_turbulence)

    if os.path.isfile(fname_bottom):
        t_bottom, d_bottom = loader.evl_loader(fname_bottom)
    elif not is_upward_facing:
        raise ValueError(
            "Expected {} to exist when transect is downfacing.".format(fname_bottom)
        )
    else:
        # Default bottom depth is the bottom of the field of view
        t_bottom = ts_raw
        d_bottom = np.ones_like(ts_raw) * np.max(depths_raw)

    if os.path.isfile(fname_surface):
        t_surface, d_surface = loader.evl_loader(fname_surface)
    elif is_upward_facing:
        raise ValueError(
            "Expected {} to exist when transect is upfacing.".format(fname_surface)
        )
    else:
        # Default surface depth of 0m for downward facing data
        t_surface = ts_raw
        d_surface = np.zeros_like(ts_raw)

    # Generate new lines from mask
    d_turbulence_new, d_bottom_new = fixup_lines(
        ts_mskd,
        depths_mskd,
        mask,
        t_turbulence=t_turbulence,
        d_turbulence=d_turbulence,
        t_bottom=t_bottom,
        d_bottom=d_bottom,
    )

    def tidy_up_line(t, d):
        if d is None:
            return np.nan * np.ones_like(ts_raw)
        is_usable = np.isfinite(d)
        if np.sum(is_usable) > 0:
            t = t[is_usable]
            d = d[is_usable]
        return np.interp(ts_raw, t, d)

    # Mask and data derived from it is sampled at the correct timestamps and
    # depths for the raw data. It should be, but might not be if either of the
    # CSV files contained invalid data.
    if (
        len(ts_raw) != len(ts_mskd)
        or len(depths_raw) != len(depths_mskd)
        or not np.allclose(ts_raw, ts_mskd)
        or not np.allclose(depths_raw, depths_mskd)
    ):
        # Interpolate depth lines to timestamps used for raw data
        d_turbulence_new = tidy_up_line(ts_mskd, d_turbulence_new)
        d_bottom_new = tidy_up_line(ts_mskd, d_bottom_new)
        # Interpolate mask
        if is_upward_facing:
            mask = scipy.interpolate.RectBivariateSpline(
                ts_mskd,
                depths_mskd[::-1],
                mask[:, ::-1].astype(np.float),
            )(ts_raw, depths_raw[::-1])[:, ::-1]
        else:
            mask = scipy.interpolate.RectBivariateSpline(
                ts_mskd,
                depths_mskd,
                mask.astype(np.float),
            )(ts_raw, depths_raw)
        # Binarise
        mask = mask > 0.5

    # Find location of passive data
    passive_edges = metadata.recall_passive_edges(sample_path, ts_raw)
    if passive_edges[0] is not None:
        passive_starts, passive_ends = passive_edges
    else:
        passive_starts, passive_ends = find_passive_data(signals_raw)
    # Determine whether each timestamp is for a period of passive recording
    is_passive = np.zeros(ts_raw.shape, dtype=bool)
    for pass_start, pass_end in zip(passive_starts, passive_ends):
        is_passive[pass_start:pass_end] = True

    # Determine whether each timestamp is for recording which was completely
    # removed from analysis (but not because it is passive recording)
    if np.all(mask):
        print("No data points were masked out in {}".format(fname_masked))
        # Use lines to create a mask
        ddepths = np.broadcast_to(depths_mskd, signals_mskd.shape)
        mask[ddepths < np.expand_dims(d_turbulence_new, -1)] = 0
        mask[ddepths > np.expand_dims(d_bottom_new, -1)] = 0
        mask[is_passive] = 0
    allnan = np.all(~mask, axis=1)

    # Timestamp is entirely removed if everything is nan and it isn't passive
    is_removed_raw = allnan & ~is_passive
    # But we don't want to include removed segments which are marked as
    # removed just because the lines crossed each other.
    r_starts_raw, r_ends_raw = utils.get_indicator_onoffsets(is_removed_raw)
    r_starts = []
    r_ends = []
    is_removed = np.zeros_like(is_removed_raw)
    for r_start, r_end in zip(r_starts_raw, r_ends_raw):
        # Check how many points in the fully removed region don't have
        # overlapping turbulence and bottom lines
        n_without_overlap = np.sum(
            d_turbulence_new[r_start : r_end + 1] < d_bottom_new[r_start : r_end + 1]
        )
        if n_without_overlap == 0:
            # Region is removed only by virtue of the lines crossing; we
            # don't include this
            continue
        if r_end - r_start >= 4 and n_without_overlap <= 2:
            # We expect more than just the edges of the boundary for the region
            # to have uncrossed lines
            continue
        r_starts.append(r_start)
        r_ends.append(r_end)
        is_removed[r_start : r_end + 1] = 1

    # Ensure depth is always increasing (which corresponds to descending from
    # the air down the water column)
    if is_upward_facing:
        depths_raw = depths_raw[::-1].copy()
        signals_raw = signals_raw[:, ::-1].copy()
        mask = mask[:, ::-1].copy()

    # Offset by a small amount to catch pixels on the edge of the line
    depth_intv = abs(depths_raw[1] - depths_raw[0])
    if d_turbulence is not None:
        d_turbulence += depth_intv / 4
    if d_bottom is not None:
        d_bottom -= depth_intv / 4

    d_turbulence = tidy_up_line(t_turbulence, d_turbulence)
    d_bottom = tidy_up_line(t_bottom, d_bottom)
    d_surface = tidy_up_line(t_surface, d_surface)

    # Fix up surface line, removing anomalies
    d_surface, is_surrogate_surface = fix_surface_line(ts_raw, d_surface, is_passive)

    # Make a mask indicating left-over patches. This is 0 everywhere,
    # except 1s wherever pixels in the overall mask are removed for
    # reasons not explained by the turbulence and bottom lines, and is_passive and
    # is_removed indicators.
    mask_patches = ~mask
    mask_patches[is_passive] = 0
    mask_patches[is_removed > 0.5] = 0
    mask_patches_og = mask_patches.copy()
    mask_patches_ntob = mask_patches.copy()
    ddepths = np.broadcast_to(depths_raw, signals_raw.shape)
    mask_patches[ddepths <= np.expand_dims(d_turbulence_new, -1)] = 0
    mask_patches[ddepths >= np.expand_dims(d_bottom_new, -1)] = 0
    mask_patches_og[ddepths <= np.expand_dims(d_turbulence, -1)] = 0
    mask_patches_og[ddepths >= np.expand_dims(d_bottom, -1)] = 0
    mask_patches_ntob[ddepths <= np.expand_dims(d_turbulence_new, -1)] = 0
    mask_patches_ntob[ddepths >= np.expand_dims(d_bottom, -1)] = 0
    # Remove trivial mask patches. If the pixel above and below are both empty,
    # delete a mask with a height of only one-pixel.
    mask_patches[
        ~(
            np.concatenate(
                (
                    mask_patches[:, 2:],
                    np.ones((mask_patches.shape[0], 2), dtype="bool"),
                ),
                axis=-1,
            )
            | np.concatenate(
                (
                    np.ones((mask_patches.shape[0], 2), dtype="bool"),
                    mask_patches[:, :-2],
                ),
                axis=-1,
            )
        )
    ] = 0
    mask_patches_og[
        ~(
            np.concatenate(
                (
                    mask_patches_og[:, 2:],
                    np.ones((mask_patches_og.shape[0], 2), dtype="bool"),
                ),
                axis=-1,
            )
            | np.concatenate(
                (
                    np.ones((mask_patches_og.shape[0], 2), dtype="bool"),
                    mask_patches_og[:, :-2],
                ),
                axis=-1,
            )
        )
    ] = 0
    mask_patches_ntob[
        ~(
            np.concatenate(
                (
                    mask_patches_ntob[:, 2:],
                    np.ones((mask_patches_ntob.shape[0], 2), dtype="bool"),
                ),
                axis=-1,
            )
            | np.concatenate(
                (
                    np.ones((mask_patches_ntob.shape[0], 2), dtype="bool"),
                    mask_patches_ntob[:, :-2],
                ),
                axis=-1,
            )
        )
    ] = 0

    # Collate transect as a dictionary
    transect = {}
    transect["timestamps"] = ts_raw
    transect["depths"] = depths_raw
    transect["Sv"] = signals_raw
    transect["mask"] = mask
    transect["mask_patches"] = mask_patches
    transect["mask_patches-original"] = mask_patches_og
    transect["mask_patches-ntob"] = mask_patches_ntob
    transect["turbulence"] = d_turbulence_new
    transect["turbulence-original"] = d_turbulence
    transect["bottom"] = d_bottom_new
    transect["bottom-original"] = d_bottom
    transect["surface"] = d_surface
    transect["is_surrogate_surface"] = is_surrogate_surface
    transect["is_passive"] = is_passive
    transect["is_removed"] = is_removed
    transect["is_upward_facing"] = is_upward_facing

    return transect


def split_transect(timestamps=None, threshold=20, percentile=97.5, **transect):
    """
    Splits a transect into segments each containing contiguous recordings.

    Parameters
    ----------
    timestamps : array_like
        A 1-d array containing the timestamp at which each recording was
        measured. The sampling is assumed to high-frequency with
        occassional gaps.
    threshold : int, optional
        Threshold for splitting timestamps into segments. Any timepoints
        further apart than `threshold` times the `percentile` percentile of the
        difference between timepoints will be split apart into new segments.
        Default is `20`.
    percentile : float, optional
        The percentile at which to sample the timestamp intervals to establish
        a baseline typical interval. Default is `97.5`.
    **kwargs
        Arbitrary additional transect variables, which will be split into
        segments as appropriate in accordance with `timestamps`.

    Yields
    ------
    dict
        Containing segmented data, key/value pairs as per given in `**kwargs`
        in addition to `timestamps`.
    """

    if timestamps is None:
        raise ValueError("The `timestamps` argument is required.")

    dt = np.diff(timestamps)
    break_indices = np.where(dt > np.percentile(dt, percentile) * threshold)[0]
    if len(break_indices) > 0:
        break_indices += 1

    for seg_start, seg_end in zip(
        np.r_[0, break_indices],
        np.r_[break_indices, len(timestamps)],
    ):
        segment = {}
        segment["timestamps"] = timestamps[seg_start:seg_end]
        for key in transect:
            if key in ("depths",) or np.asarray(transect[key]).size <= 1:
                segment[key] = copy.deepcopy(transect[key])
            else:
                segment[key] = transect[key][seg_start:seg_end]
        yield segment


def join_transect(transects):
    """
    Joins segmented transects together into a single dictionary.

    Parameters
    ----------
    transects : iterable of dict
        Transect segments, each with the same fields and compatible shapes.

    Yields
    ------
    dict
        Transect data.
    """

    non_timelike_dims = ["depths"]

    for i, transect in enumerate(transects):
        if "depths" not in transect:
            raise ValueError(
                "'depths' is a required field, not found in transect {}".format(i)
            )
        if i == 0:
            output = {k: [] for k in transect}
            output["depths"] = transect["depths"]
            for key in transect:
                if np.asarray(transect[key]).size <= 1:
                    output[key] = transect[key]
                    non_timelike_dims.append(key)
        if not np.allclose(output["depths"], transect["depths"]):
            raise ValueError("'depths' must be the same for all segments.")
        if transect.keys() != output.keys():
            raise ValueError("Keys mismatch.")
        for key in output:
            if key in non_timelike_dims:
                continue
            output[key].append(transect[key])

    for key in output:
        if key in non_timelike_dims:
            continue
        output[key] = np.concatenate(output[key], axis=0)

    return output
