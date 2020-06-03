'''
Manipulating lines and masks contained in echoview files.
'''

import copy
import os
import warnings

import numpy as np

from . import loader
from .. import utils


ROOT_DATA_DIR = loader.ROOT_DATA_DIR


def find_passive_data(signals, n_depth_use=38, threshold=25., deviation=None):
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

    if len(indices_possible_end) > 0 and (
        len(indices_possible_start) == 0 or indices_possible_end[0] < indices_possible_start[0]
    ):
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


def make_lines_from_mask(mask, depths=None, max_gap_squash=1.):
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
    max_gap_squash : float, optional
        Maximum gap to merge together, in metres. Default is `1.`.

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
    # Ensure input is an array. Make a copy, so we don't modify the input.
    mask = np.array(mask, copy=True)

    # Autocomplete depth with index.
    if depths is None:
        depths = np.arange(mask.shape[1])
    depths = np.asarray(depths)
    if len(depths) != mask.shape[1]:
        raise ValueError('Length of depths input must match dim 1 of mask.')
    depth_intv = np.median(np.diff(depths))

    # Merge small gaps between masked out data, so the middle is masked out too
    # We merge all gaps smaller than 120 pixels apart.
    max_gap_squash_idx = int(np.round(max_gap_squash / depth_intv))
    for i in range(max_gap_squash_idx, 2, -1):
        li = ~np.any([
            np.pad(mask[:, i//2:], ((0, 0), (0, i//2)), mode='constant'),
            np.pad(mask[:, :-((i+1)//2)], ((0, 0), ((i+1)//2, 0)), mode='constant'),
        ], axis=0)
        mask[li] = 0

    # Check which depths were removed for each timestamp
    removed_depths = np.tile(depths, (mask.shape[0], 1)).astype('float')
    removed_depths[~mask] = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN (slice|axis) encountered')
        # Top line is the smallest removed depth at each timepoint.
        # We offset by depth_intv / 2 to get a depth in between the last kept
        # value at the top and the first removed value.
        d_top = np.nanmin(removed_depths, axis=1) - depth_intv / 2
        # Bottom line is the largest removed depth at each timepoint,
        # offset similarly.
        d_bot = np.nanmax(removed_depths, axis=1) + depth_intv / 2

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

    # Ensure nans in the lines are replaced with prior values, if possible
    li = np.isnan(d_top_new)
    if d_top is not None:
        d_top_new[li] = d_top[li]
    elif np.any(~li):
        d_top_new[li] = np.interp(
            timestamps[li],
            timestamps[~li],
            d_top_new[~li],
        )
    li = np.isnan(d_bot_new)
    if d_bot is not None:
        d_bot_new[li] = d_bot[li]
    elif np.any(~li):
        d_bot_new[li] = np.interp(
            timestamps[li],
            timestamps[~li],
            d_bot_new[~li],
        )

    # Ensure that the lines cover at least as much material as they did before
    if d_top is not None:
        d_top_new = np.maximum(d_top, d_top_new)
    if d_bot is not None:
        d_bot_new = np.minimum(d_bot, d_bot_new)

    # This mask can't handle regions where all the data was removed.
    # Find those and replace them with the original lines, if they were
    # provided. If they weren't, interpolate to fill the holes (if there is
    # something to interpolate).
    all_removed = ~np.any(mask, axis=1)
    any_all_removed = np.any(all_removed)
    everything_removed = np.all(all_removed)
    if not any_all_removed:
        pass
    elif d_top is not None:
        d_top_new[all_removed] = d_top[all_removed]
    elif ~everything_removed:
        d_top_new[all_removed] = np.interp(
            timestamps[all_removed],
            timestamps[~all_removed],
            d_top_new[~all_removed],
        )
    else:
        d_top_new[all_removed] = np.nan

    if not any_all_removed:
        pass
    elif d_bot is not None:
        d_bot_new[all_removed] = d_bot[all_removed]
    elif ~everything_removed:
        d_bot_new[all_removed] = np.interp(
            timestamps[all_removed],
            timestamps[~all_removed],
            d_bot_new[~all_removed],
        )
    else:
        d_bot_new[all_removed] = np.nan

    return d_top_new, d_bot_new


def load_decomposed_transect_mask(
        sample_path,
        dataset='',
    ):
    '''
    Loads a raw and masked transect and decomposes the mask into top and bottom
    lines, and passive and removed regions.

    Parameters
    ----------
    sample_path : str
        Path to sample, without extension. The raw data should be located at
        `sample_path + '_Sv_raw.csv'`.
    dataset : {'mobile', 'MinasPassage', ''}, optional
        Name of dataset. Used to check integrity of the data loaded against
        what is expected for that dataset. Default is `''`, which has no
        dataset-specific expectations.

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
            - 'is_upward_facing' : bool
                Indicates whether the recording source is located at the
                deepest depth (i.e. the seabed), facing upwards. Otherwise, the
                recording source is at the shallowest depth (i.e. the surface),
                facing downwards.
    '''

    # Load raw data
    fname_raw = os.path.join(sample_path + '_Sv_raw.csv')
    fname_masked = os.path.join(sample_path + '_Sv.csv')

    ts_raw, depths_raw, signals_raw = loader.transect_loader(fname_raw)
    ts_mskd, depths_mskd, signals_mskd = loader.transect_loader(fname_masked)
    mask = ~np.isnan(signals_mskd)

    fname_top1 = os.path.join(sample_path + '_turbulence.evl')
    fname_top2 = os.path.join(sample_path + '_air.evl')
    fname_bot = os.path.join(sample_path + '_bottom.evl')
    fname_surf = os.path.join(sample_path + '_surface.evl')

    if os.path.isfile(fname_top1):
        fname_top = fname_top1
        if os.path.isfile(fname_top2):
            raise ValueError(
                'Only one of {} and {} should exist.'
                .format(fname_top1, fname_top2)
            )
    elif os.path.isfile(fname_top2):
        fname_top = fname_top2
    else:
        raise ValueError(
            'Neither {} nor {} were found.'
            .format(fname_top1, fname_top2)
        )
    t_top, d_top = loader.evl_loader(fname_top)

    if os.path.isfile(fname_bot):
        t_bot, d_bot = loader.evl_loader(fname_bot)
    elif dataset == 'mobile':
        raise ValueError(
            'Expected {} to exist when dateset is {}.'
            .format(fname_bot, dataset)
        )
    else:
        t_bot = d_bot = None

    if os.path.isfile(fname_surf):
        t_surf, d_surf = loader.evl_loader(fname_surf)
    elif dataset == "MinasPassage" or "GrandPassage" in dataset:
        raise ValueError(
            'Expected {} to exist when dateset is {}.'
            .format(fname_surf, dataset)
        )
    else:
        t_surf = d_surf = None

    # Generate new lines from mask
    d_top_new, d_bot_new = fixup_lines(
        ts_raw,
        depths_raw,
        signals_raw,
        mask,
        t_top=t_top,
        d_top=d_top,
        t_bot=t_bot,
        d_bot=d_bot,
    )
    passive_starts, passive_ends = find_passive_data(signals_raw)
    # Determine whether each timestamps is for a period of passive recording
    is_passive = np.zeros(ts_raw.shape, dtype=bool)
    for pass_start, pass_end in zip(passive_starts, passive_ends):
        is_passive[pass_start:pass_end] = True

    # Determine whether each timestamp is for recording which was completely
    # removed from analysis (but not because it is passive recording)
    mask = ~np.isnan(signals_mskd)
    if np.all(mask):
        print('No data points were masked out in {}'.format(fname_masked))
        # Use lines to create a mask
        ddepths = np.broadcast_to(depths_mskd, signals_mskd.shape)
        mask[ddepths < np.expand_dims(d_top_new, -1)] = 0
        mask[ddepths > np.expand_dims(d_bot_new, -1)] = 0
        mask[is_passive] = 0
    allnan = np.all(np.isnan(signals_mskd), axis=1)

    # Timestamp is entirely removed if everything is nan and it isn't passive
    is_removed_raw = allnan & ~is_passive
    # But we don't want to include removed segments which are marked as
    # removed just because the lines crossed each other.
    r_starts_raw, r_ends_raw = utils.get_indicator_onoffsets(is_removed_raw)
    r_starts = []
    r_ends = []
    is_removed = np.zeros_like(is_removed_raw)
    for r_start, r_end in zip(r_starts_raw, r_ends_raw):
        if not np.all(d_top_new[r_start : r_end + 1] >= d_bot_new[r_start : r_end + 1]):
            r_starts.append(r_start)
            r_ends.append(r_end)
            is_removed[r_start : r_end + 1] = 1

    # Determine whether depths are ascending or descending
    is_upward_facing = (depths_raw[-1] < depths_raw[0])
    # Ensure depth is always increasing (which corresponds to descending from
    # the air down the water column)
    if is_upward_facing:
        depths_raw = depths_raw[::-1].copy()
        signals_raw = signals_raw[:, ::-1].copy()
        mask = mask[:, ::-1].copy()

    # Offset by a small amount to catch pixels on the edge of the line
    depth_intv = abs(depths_raw[1] - depths_raw[0])
    if d_top is not None:
        d_top += depth_intv / 4
    if d_bot is not None:
        d_bot -= depth_intv / 4

    def tidy_up_line(t, d):
        if d is None:
            d = np.nan * np.ones_like(ts_raw)
        else:
            d = np.interp(ts_raw, t, d)
        return d

    d_top = tidy_up_line(t_top, d_top)
    d_bot = tidy_up_line(t_bot, d_bot)
    d_surf = tidy_up_line(t_surf, d_surf)

    # Make a mask indicating left-over patches. This is 0 everywhere,
    # except 1s wherever pixels in the overall mask are removed for
    # reasons not explained by the top and bottom lines, and is_passive and
    # is_removed indicators.
    mask_patches = ~mask
    mask_patches[is_passive] = 0
    mask_patches[is_removed] = 0
    mask_patches_og = mask_patches.copy()
    mask_patches_ntob = mask_patches.copy()
    ddepths = np.broadcast_to(depths_raw, signals_raw.shape)
    mask_patches[ddepths <= np.expand_dims(d_top_new, -1)] = 0
    mask_patches[ddepths >= np.expand_dims(d_bot_new, -1)] = 0
    mask_patches_og[ddepths <= np.expand_dims(d_top, -1)] = 0
    mask_patches_og[ddepths >= np.expand_dims(d_bot, -1)] = 0
    mask_patches_ntob[ddepths <= np.expand_dims(d_top_new, -1)] = 0
    mask_patches_ntob[ddepths >= np.expand_dims(d_bot, -1)] = 0
    # Remove trivial mask patches. If the pixel above and below are both empty,
    # delete a mask with a height of only one-pixel.
    mask_patches[~(
        np.concatenate((mask_patches[:, 2:], np.ones((mask_patches.shape[0], 2), dtype="bool")), axis=-1)
        |
        np.concatenate((np.ones((mask_patches.shape[0], 2), dtype="bool"), mask_patches[:, :-2]), axis=-1)
    )] = 0
    mask_patches_og[~(
        np.concatenate((mask_patches_og[:, 2:], np.ones((mask_patches_og.shape[0], 2), dtype="bool")), axis=-1)
        |
        np.concatenate((np.ones((mask_patches_og.shape[0], 2), dtype="bool"), mask_patches_og[:, :-2]), axis=-1)
    )] = 0
    mask_patches_ntob[~(
        np.concatenate((mask_patches_ntob[:, 2:], np.ones((mask_patches_ntob.shape[0], 2), dtype="bool")), axis=-1)
        |
        np.concatenate((np.ones((mask_patches_ntob.shape[0], 2), dtype="bool"), mask_patches_ntob[:, :-2]), axis=-1)
    )] = 0

    # Collate transect as a dictionary
    transect = {}
    transect['timestamps'] = ts_raw
    transect['depths'] = depths_raw
    transect['Sv'] = signals_raw
    transect['mask'] = mask
    transect['mask_patches'] = mask_patches
    transect['mask_patches-original'] = mask_patches_og
    transect['mask_patches-ntob'] = mask_patches_ntob
    transect['top'] = d_top_new
    transect['bottom'] = d_bot_new
    transect['surface'] = d_surf
    transect['top-original'] = d_top
    transect['bottom-original'] = d_bot
    transect['is_passive'] = is_passive
    transect['is_removed'] = is_removed
    transect['is_upward_facing'] = is_upward_facing

    return transect


def split_transect(timestamps=None, threshold=20, percentile=97.5, **transect):
    '''
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
        Containing segmented data, key/value pairs as per given in **kwargs
        in addition to `timestamps`.
    '''

    if timestamps is None:
        raise ValueError('The `timestamps` argument is required.')

    dt = np.diff(timestamps)
    break_indices = np.where(dt > np.percentile(dt, percentile) * threshold)[0]
    if len(break_indices) > 0:
        break_indices += 1

    for seg_start, seg_end in zip(
        np.r_[0, break_indices],
        np.r_[break_indices, len(timestamps)],
    ):
        segment = {}
        segment['timestamps'] = timestamps[seg_start:seg_end]
        for key in transect:
            if key in ('depths', ) or np.asarray(transect[key]).size <= 1:
                segment[key] = copy.deepcopy(transect[key])
            else:
                segment[key] = transect[key][seg_start:seg_end]
        yield segment


def join_transect(transects):
    '''
    Joins segmented transects together into a single dictionary.

    Parameters
    ----------
    transects : iterable of dict
        Transect segments, each with the same fields and compatible shapes.

    Yields
    ------
    dict
        Transect data.
    '''

    non_timelike_dims = ['depths']

    for i, transect in enumerate(transects):
        if 'depths' not in transect:
            raise ValueError(
                "'depths' is a required field, not found in transect {}".format(i)
            )
        if i == 0:
            output = {k: [] for k in transect}
            output['depths'] = transect['depths']
            for key in transect:
                if np.asarray(transect[key]).size <= 1:
                    output[key] = transect[key]
                    non_timelike_dims.append(key)
        if not np.allclose(output['depths'], transect['depths']):
            raise ValueError("'depths' must be the same for all segments.")
        if transect.keys() != output.keys():
            raise ValueError('Keys mismatch.')
        for key in output:
            if key in non_timelike_dims: continue
            output[key].append(transect[key])

    for key in output:
        if key in non_timelike_dims: continue
        output[key] = np.concatenate(output[key], axis=0)

    return output
