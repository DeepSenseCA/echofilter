'''
Manipulating lines and masks contained in echoview files.
'''

import os
import warnings

import numpy as np

from . import loader


ROOT_DATA_DIR = loader.ROOT_DATA_DIR


def find_passive_data(signals, n_depth_use=26, threshold=10, deviation=None):
    '''
    Find segments of Sv recording which correspond to passive recording.

    Parameters
    ----------
    signals : array_like
        Two-dimensional array of Sv values, shaped `[timestamps, depths]`.
    n_depth_use : int, optional
        How many Sv depths to use, starting with the first depths (closest
        to the sounder device). If `None` all depths are used. Default is `26`.
    threshold : float, optional
        Threshold for start/end of passive regions. Default is `10`.
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
    '''
    # Ensure signals is numpy array
    signals = np.asarray(signals)

    if n_depth_use is None:
        n_depth_use = signals.shape[1]

    md = np.median(np.diff(signals[:, :n_depth_use], axis=0), axis=1)

    if threshold is not None and deviation is not None:
        raise ValueError('Only one of `threshold` and `deviation` should be set.')
    if threshold is None:
        if deviation is None:
            raise ValueError('Neither of `threshold` and `deviation` were set.')
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

    if len(indices_possible_start) == 0 or indices_possible_end[0] < indices_possible_start[0]:
        indices_passive_start.append(0)
        current_index = indices_possible_end[0]
        indices_passive_end.append(current_index)
        indices_possible_start = indices_possible_start[indices_possible_start > current_index]
        indices_possible_end = indices_possible_end[indices_possible_end > current_index]

    while len(indices_possible_start) > 0:
        current_index = indices_possible_start[0]
        indices_passive_start.append(current_index)
        baseline = signals[current_index - 1, :n_depth_use]

        # Find first column which returns to the baseline value seen before passive region
        offsets = np.nonzero(
            np.median(baseline - signals[current_index:, :n_depth_use], axis=1) < threshold_high
        )[0]
        if len(offsets) == 0:
            current_index = signals.shape[0]
        else:
            current_index += offsets[0]
        indices_passive_end.append(current_index)

        # Remove preceding indices from the list of candidates
        indices_possible_start = indices_possible_start[indices_possible_start > current_index]
        indices_possible_end = indices_possible_end[indices_possible_end > current_index]

        # Check the start was sufficiently inclusive
        if current_index < signals.shape[0]:
            baseline = signals[current_index, :n_depth_use]
            nonpassives = np.nonzero(
                np.median(baseline - signals[:current_index, :n_depth_use], axis=1) < threshold_high
            )[0]
            if len(nonpassives) == 0:
                indices_passive_start[-1] = 0
            else:
                indices_passive_start[-1] = min(
                    indices_passive_start[-1],
                    nonpassives[-1] + 1,
                )

        # Combine with preceding passive segments if they overlap
        while len(indices_passive_start) > 1 and indices_passive_start[-1] <= indices_passive_end[-2]:
            indices_passive_start = indices_passive_start[:-1]
            indices_passive_end = indices_passive_end[:-2] + indices_passive_end[-1:]

    return np.array(indices_passive_start), np.array(indices_passive_end)


def make_lines_from_mask(mask, depths=None):
    '''
    Determines top and bottom lines for a mask array.

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

    Returns
    -------
    d_top : numpy.ndarray
        Depth of top line. This is the line of smaller depth which
        separates the `False` region of `mask` from the central region of
        `True` values. (If `depths` is monotonically increasing, this is
        for the start of the columns of `mask`, otherwise it is at the end.)
    d_bot : numpy.ndarray
        Depth of bottom line. As for `d_top`, but for the other end of the
        array.
    '''
    # Ensure input is an array.
    mask = np.asarray(mask)

    # Autocomplete depth with index.
    if depths is None:
        depths = np.arange(mask.shape[1])
    depths = np.asarray(depths)
    if len(depths) != mask.shape[1]:
        raise ValueError('Length of depths input must match dim 1 of mask.')

    # If depths is decreasing, we need to multiply through by -1 so the min
    # and max we do later will work (max finding the last, largest, depth, and
    # min finding the first, smallest, depth).
    depth_is_reversed = depths[-1] < depths[0]
    if depth_is_reversed:
        depths = depths * -1

    # Find the midway point (median) of non-masked data in each column.
    # This corresponds to the middle of the water column.
    indices_v = np.arange(mask.shape[1])
    indices = np.tile(indices_v, (mask.shape[0], 1)).astype('float')
    indices[~mask] = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN (slice|axis) encountered')
        midway_indices = np.round(np.nanmedian(indices, axis=1))

    # Find the last nan value in the top half of each column.
    top_depths = np.tile(
        np.concatenate([(depths[:-1] + depths[1:]) / 2, depths[-1:]]),
        (mask.shape[0], 1),
    )
    top_depths[mask] = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'invalid value encountered in (greater|less)')
        is_lower_half = np.expand_dims(indices_v, 0) > np.expand_dims(midway_indices, -1)
    top_depths[is_lower_half] = np.nan
    d_top = np.nanmax(top_depths, axis=1)

    # Find the first nan value in the bottom half of each column.
    bot_depths = np.tile(
        np.concatenate([depths[:1], (depths[:-1] + depths[1:]) / 2]),
        (mask.shape[0], 1),
    )
    bot_depths[mask] = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'invalid value encountered in (greater|less)')
        is_upper_half = np.expand_dims(indices_v, 0) < np.expand_dims(midway_indices, -1)
    bot_depths[is_upper_half] = np.nan
    d_bot = np.nanmin(bot_depths, axis=1)

    if depth_is_reversed:
        d_top *= -1
        d_bot *= -1
        return d_bot, d_top

    return d_top, d_bot


def make_lines_from_masked_csv(fname):
    '''
    Load a masked csv file output from echoview and generate lines which
    reproduce the mask.

    Parameters
    ----------
    fname : str
        Path to file containing masked echoview output data in csv format.

    Returns
    -------
    timestamps : numpy.ndarray
        Sample timestamps.
    d_top : numpy.ndarray
        Depth of top line.
    d_bot : numpy.ndarray
        Depth of bottom line.
    '''
    # Load the masked data
    timestamps, depths, signals_mskd = loader.transect_loader(fname)
    mask = ~np.isnan(signals_mskd)
    d_top, d_bot = make_lines_from_mask(mask, depths)
    return timestamps, d_top, d_bot


def write_lines_for_masked_csv(fname_mask, fname_top=None, fname_bot=None):
    '''
    Write new top and bottom lines based on csv containing masked echoview
    output.

    Parameters
    ----------
    fname_mask : str
        Path to input file containing masked echoview output data in csv
        format.
    fname_top : str, optional
        Destination of generated top line, written in evl format. If `None`
        (default), the output name is `<fname_base>_mask-top.evl`, where
        `<fname_base>` is `fname_mask` without extension and without any
        occurence of the substrings `_Sv_raw` or `_Sv` in the base file name.
    fname_bot : str
        Destination of generated bottom line, written in evl format. If `None`
        (default), the output name is `<fname_base>_mask-bottom.evl`.
    '''
    if fname_top is None or fname_bot is None:
        fname_base = os.path.splitext(fname_mask)[0]
        dirname, fname_base = os.path.split(fname_base)
        fname_base = fname_base.replace('_Sv_raw', '').replace('_Sv', '')
        fname_base = os.path.join(dirname, fname_base)
    if fname_top is None:
        fname_top = fname_base + '_mask-top.evl'
    if fname_bot is None:
        fname_bot = fname_base + '_mask-bottom.evl'
    # Generate the new lines.
    timestamps, d_top, d_bot = make_lines_from_masked_csv(fname_mask)
    # Write the new lines to their output files.
    loader.evl_writer(fname_top, timestamps, d_top)
    loader.evl_writer(fname_bot, timestamps, d_bot)


def find_nonzero_region_boundaries(v):
    '''
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
    '''

    v = np.asarray(v)
    v = (v != 0)
    v = v.astype(np.float)

    starts = np.nonzero(np.diff(v) > 0)[0] + 1
    ends = np.nonzero(np.diff(v) < 0)[0] + 1

    if v[0]:
        starts = np.concatenate(([0], starts))

    if v[-1]:
        ends = np.concatenate((ends, [len(vector)]))

    return starts, ends


def fixup_lines(
        timestamps,
        depths,
        signals_raw,
        mask,
        t_top=None,
        d_top=None,
        t_bot=None,
        d_bot=None,
        return_passive_boundaries=False,
    ):
    '''
    Extend existing top/bottom lines based on masked target Sv output.

    Parameters
    ----------
    timestamps : array_like
        Shaped `(num_timestamps, )`.
    depths : array_like
        Shaped `(num_depths, )`.
    signals_raw : array_like
        Shaped `(num_timestamps, num_depths)`.
    mask : array_like
        Boolean array, where `True` denotes kept entries. Same shape as
        `signals_raw`.
    t_top : array_like, optional
        Sampling times for existing top line.
    d_top : array_like, optional
        Depth of existing top line.
    t_bot : array_like, optional
        Sampling times for existing bottom line.
    d_bot : array_like, optional
        Depth of existing bottom line.
    return_passive_boundaries : bool, optional
        Whether to return `passive_starts` and `passive_ends`. Default is
        `False`.

    Returns
    -------
    d_top_new : numpy.ndarray
        Depth of new top line.
    d_bot_new : numpy.ndarray
        Depth of new bottom line.
    passive_starts : numpy.ndarray, optional
        Start indices for passive segments of recording in `signals_raw`.
        Included in returned tuple if `return_passive_boundaries` is `True`.
    passive_ends : numpy.ndarray, optional
        Start indices for passive segments of recording in `signals_raw`.
        Included in returned tuple if `return_passive_boundaries` is `True`.
    '''
    # Handle different sampling grids
    if d_top is not None:
        if t_top is None:
            raise ValueError('t_top must be provided if d_top is provided')
        d_top = np.interp(timestamps, t_top, d_top)

    if d_bot is not None:
        if t_bot is None:
            raise ValueError('t_bot must be provided if d_bot is provided')
        d_bot = np.interp(timestamps, t_bot, d_bot)

    # Generate fresh lines corresponding to said mask
    d_top_new, d_bot_new = make_lines_from_mask(mask, depths)

    # This mask can't handle regions where all the data was removed.
    # Find those and replace them with the original lines, if they were
    # provided. If they weren't, interpolate to fill the holes.
    all_removed = ~np.any(mask, axis=1)
    if d_top is not None:
        d_top_new[all_removed] = d_top[all_removed]
    else:
        d_top_new[all_removed] = np.interp(
            timestamps[all_removed],
            timestamps[~all_removed],
            d_top_new[~all_removed],
        )
    if d_bot is not None:
        d_bot_new[all_removed] = d_bot[all_removed]
    else:
        d_bot_new[all_removed] = np.interp(
            timestamps[all_removed],
            timestamps[~all_removed],
            d_bot_new[~all_removed],
        )

    # Convert this into start and end points for later use(?)
    # removal_starts, removal_ends = find_nonzero_region_boundaries(all_removed)

    # For passive data, we set the target to be NaN as a placeholder.
    # A working target can be set downstream.
    passive_starts, passive_ends = find_passive_data(signals_raw)
    for start, end in zip(passive_starts, passive_ends):
        d_top_new[start:end] = np.nan
        d_bot_new[start:end] = np.nan

    if return_passive_boundaries:
        return d_top_new, d_bot_new, passive_starts, passive_ends
    return d_top_new, d_bot_new


def load_decomposed_transect_mask(
        sample,
        dataset='mobile',
        root_data_dir=ROOT_DATA_DIR,
    ):
    '''
    Loads a raw and masked transect and decomposes the mask into top and bottom
    lines, and passive and removed regions.

    Parameters
    ----------
    sample : str
        Name of sample (its relative path within dataset directory).
    dataset : str, optional
        Name of dataset (corresponding to name of directory within
        `root_data_dir` which contains `sample`). Default is `'mobile'`.
    root_data_dir : str, optional
        Path to root directory where data is located.
        Default is as given in `raw.loader.ROOT_DATA_DIR`.

    Returns
    -------
    dict
        A dictionary with keys:

            - 'timestamps' : numpy.ndarray
                Timestamps (in seconds since Unix epoch), for each recording
                timepoint.
            - 'depths' : numpy.ndarray
                Depths from the surface (in metres), with each entry
                corresponding to each column in the `signals` data.
            - 'Sv' : numpy.ndarray
                Echogram Sv data, shaped (num_timestamps, num_depths).
            - 'mask' : numpy.ndarray
                Logical array indicating which datapoints were kept (`True`)
                and which removed (`False`) for the masked Sv output.
                Shaped (num_timestamps, num_depths).
            - 'top' : numpy.ndarray
                For each timepoint, the depth of the shallowest datapoint which
                should be included for the mask. Shaped (num_timestamps, ).
            - 'bottom' : numpy.ndarray
                For each timepoint, the depth of the deepest datapoint which
                should be included for the mask. Shaped (num_timestamps, ).
            - 'is_passive' : numpy.ndarray
                Logical array showing whether a timepoint is of passive data.
                Shaped (num_timestamps, ). All passive recording data should
                be excluded by the mask.
            - 'is_removed' : numpy.ndarray
                Logical array showing whether a timepoint is entirely removed
                by the mask. Shaped (num_timestamps, ). Does not include
                periods of passive recording.
            - 'is_source_bottom' : bool
                Indicates whether the recording source is located at the
                deepest depth (i.e. the seabed), facing upwards. Otherwise, the
                recording source is at the shallowest depth (i.e. the surface),
                facing downwards.
    '''

    root_data_dir = loader.remove_trailing_slash(root_data_dir)

    # Load raw data
    fname_raw = os.path.join(root_data_dir, dataset, sample + '_Sv_raw.csv')
    fname_masked = os.path.join(root_data_dir, dataset, sample + '_Sv.csv')

    ts_raw, depths_raw, signals_raw = loader.transect_loader(fname_raw)
    ts_mskd, depths_mskd, signals_mskd = loader.transect_loader(fname_masked)
    mask = ~np.isnan(signals_mskd)

    fname_top = os.path.join(root_data_dir, dataset, sample + '_turbulence.evl')
    fname_bot = os.path.join(root_data_dir, dataset, sample + '_bottom.evl')
    t_top, d_top = loader.evl_loader(fname_top)
    t_bot, d_bot = loader.evl_loader(fname_bot)

    # Generate new lines from mask
    d_top_new, d_bot_new, passive_starts, passive_ends = fixup_lines(
        ts_raw,
        depths_raw,
        signals_raw,
        mask,
        t_top=t_top,
        d_top=d_top,
        t_bot=t_bot,
        d_bot=d_bot,
        return_passive_boundaries=True,
    )
    # Determine whether each timestamps is for a period of passive recording
    is_passive = np.zeros(ts_raw.shape, dtype=bool)
    for pass_start, pass_end in zip(passive_starts, passive_ends):
        is_passive[pass_start:pass_end] = True

    # Determine whether each timestamp is for recording which was completely
    # removed from analysis (but not because it is passive recording)
    allnan = np.all(np.isnan(signals_mskd), axis=1)
    is_removed = allnan & ~is_passive

    out = {}
    out['timestamps'] = ts_raw
    out['depths'] = depths_raw
    out['Sv'] = signals_raw
    out['mask'] = mask
    out['top'] = d_top_new
    out['bottom'] = d_bot_new
    out['is_passive'] = is_passive
    out['is_removed'] = is_removed
    out['is_source_bottom'] = (depths_raw[-1] < depths_raw[0])

    return out
