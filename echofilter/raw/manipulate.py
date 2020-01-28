'''
Manipulating lines and masks contained in echoview files.
'''

import os
import warnings

import numpy as np

from . import loader


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
    timestamps, d_top, d_bot = loader.make_lines_from_masked_csv(fname_mask)
    # Write the new lines to their output files.
    loader.evl_writer(fname_top, timestamps, d_top)
    loader.evl_writer(fname_bot, timestamps, d_bot)
