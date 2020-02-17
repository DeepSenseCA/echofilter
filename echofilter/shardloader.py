'''
Converting raw data into shards, and loading data from shards.
'''

import os
import numpy as np

from . import raw


ROOT_DATA_DIR = raw.loader.ROOT_DATA_DIR


def shard_transect(
        transect_pth,
        dataset='mobile',
        max_depth=100.,
        shard_len=128,
        root_data_dir=ROOT_DATA_DIR
        ):
    '''
    Creates a sharded copy of a transect, with the transect cut by timestamp
    and split across multiple files.

    Parameters
    ----------
    transect_pth : str
        Relative path to transect, excluding '_Sv_raw.csv'.
    dataset : str, optional
        Name of dataset. Default is `'mobile'`.
    max_depth : float, optional
        The maximum depth to include in the saved shard. Data corresponding
        to deeper locations is omitted to save on load time and memory when
        the shard is loaded. Default is `100`.
    shard_len : int, optional
        Number of timestamp samples to include in each shard. Default is 128.
    root_data_dir : str
        Path to root directory where data is located.

    Notes
    -----
    The output will be written to the directory
    <root_data_dir>_sharded/<dataset>/transect_path
    and will contain:

        - a file named `'shard_size.txt'`, which contains the sharding metadata:
          total number of samples, and shard size;
        - a directory for each shard, named 0, 1, ...
          Each shard directory will contain files:

            - depths.npy
            - timestamps.npy
            - Sv.npy
            - top.npy
            - bottom.npy

          which contain pickled numpy dumps of the matrices for each shard.
    '''
    # Define output destination
    root_data_dir = raw.loader.remove_trailing_slash(root_data_dir)
    root_shard_dir = os.path.join(root_data_dir + '_sharded', dataset)

    # Load the data, with mask decomposed into top, bottom, passive,
    # and removed regions.
    transect = raw.manipulate.load_decomposed_transect_mask(
        transect_pth,
        dataset,
        root_data_dir,
    )

    # Remove depths which are too deep for us to care about
    depth_mask = transect['depths'] <= max_depth
    transect['depths'] = transect['depths'][depth_mask]
    transect['Sv'] = transect['Sv'][:, depth_mask]
    transect['mask'] = transect['mask'][:, depth_mask]

    # Reduce floating point precision for some variables
    for key in ('Sv', 'top', 'bottom'):
        transect[key] = np.single(transect[key])

    # Prep output directory
    dirname = os.path.join(root_shard_dir, transect_pth)
    os.makedirs(dirname, exist_ok=True)

    # Save sharding metadata (total number of datapoints, shard size) to
    # make loading from the shards easier
    with open(os.path.join(dirname, 'shard_size.txt'), 'w') as hf:
        print('{},{}'.format(transect['Sv'].shape[0], shard_len), file=hf)

    # Work out where to split the arrays
    indices = range(shard_len, transect['Sv'].shape[0], shard_len)

    splits = {}
    for key in transect:
        if key in ('depths', 'is_source_bottom'):
            continue
        splits[key] = np.split(transect[key], indices)

    for key in splits:
        if len(splits[key]) != len(splits['Sv']):
            raise ValueError('Inconsistent split lengths')

    transect['is_source_bottom'] = np.array(transect['is_source_bottom'])

    # Save the data for each of the shards
    for i in range(len(splits['Sv'])):
        os.makedirs(os.path.join(dirname, str(i)), exist_ok=True)
        for key in ('depths', 'is_source_bottom'):
            transect[key].dump(os.path.join(dirname, str(i), key + '.npy'))
        for key in splits:
            splits[key][i].dump(os.path.join(dirname, str(i), key + '.npy'))


def load_transect_from_shards_abs(
        transect_abs_pth,
        i1=0,
        i2=None,
        ):
    '''
    Load transect data from shard files.

    Parameters
    ----------
    transect_abs_pth : str
        Absolute path to transect shard directory.
    i1 : int, optional
        Index of first sample to retrieve. Default is `0`, the first sample.
    i2 : int, optional
        Index of last sample to retrieve. As-per python convention, the range
        `i1` to `i2` is inclusive on the left and exclusive on the right, so
        datapoint `i2 - 1` is the right-most datapoint loaded. Default is
        `None`, which loads everything up to and including to the last sample.

    Returns
    -------
    dict
        A dictionary with keys:

            - 'timestamps' : numpy.ndarray
                Timestamps (in seconds since Unix epoch), for each recording
                timepoint. The number of entries, `num_timestamps`, is equal
                to `i2 - i1`.
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
    # Load the sharding metadata
    with open(os.path.join(transect_abs_pth, 'shard_size.txt'), 'r') as f:
        n_timestamps, shard_len = f.readline().strip().split(',')
        n_timestamps = int(n_timestamps)
        shard_len = int(shard_len)
    # Set the default value for i2
    if i2 is None: i2 = n_timestamps

    # Sanity check
    if i1 > n_timestamps:
        raise ValueError(
            'All requested datapoints out of range: {}, {} > {}'
            .format(i1, i2, n_timestamps)
        )
    if i2 < 0:
        raise ValueError(
            'All requested datapoints out of range: {}, {} < {}'
            .format(i1, i2, 0)
        )
    # Make indices safe
    i1_ = max(0, i1)
    i2_ = min(i2, n_timestamps)
    # Work out which shards we'll need to load to get this data
    j1 = max(0, int(i1 / shard_len))
    j2 = int(min(i2, n_timestamps - 1) / shard_len)

    transect = {}
    # Depths and is_source_bottom should all be the same. Only load one of
    # each of them.
    for key in ('depths', 'is_source_bottom'):
        transect[key] = np.load(
            os.path.join(transect_abs_pth, str(j1), key + '.npy'),
            allow_pickle=True,
        )

    # Load the rest, knitting the shards back together and cutting down to just
    # the necessary timestamps.
    def load_shard(fname):
        # Load necessary shards
        broad_data = np.concatenate([
            np.load(os.path.join(transect_abs_pth, str(j), fname + '.npy'), allow_pickle=True)
            for j in range(j1, j2+1)
        ])
        # Have to trim data down, and pad if requested indices out of range
        return np.concatenate([
            broad_data[[0] * (i1_ - i1)],
            broad_data[(i1_ - j1 * shard_len) : (i2_ - j1 * shard_len)],
            broad_data[[-1] * (i2 - i2_)],
        ])

    for key in ('timestamps', 'Sv', 'mask', 'top', 'bottom', 'is_passive',
                'is_removed'):
        transect[key] = load_shard(key)

    return transect


def load_transect_from_shards_rel(
        transect_rel_pth,
        i1=0,
        i2=None,
        dataset='mobile',
        root_data_dir=ROOT_DATA_DIR,
        ):
    '''
    Load transect data from shard files.

    Parameters
    ----------
    transect_rel_pth : str
        Relative path to transect.
    i1 : int, optional
        Index of first sample to retrieve. Default is `0`, the first sample.
    i2 : int, optional
        Index of last sample to retrieve. As-per python convention, the range
        `i1` to `i2` is inclusive on the left and exclusive on the right, so
        datapoint `i2 - 1` is the right-most datapoint loaded. Default is
        `None`, which loads everything up to and including to the last sample.
    dataset : str, optional
        Name of dataset. Default is `'mobile'`.
    root_data_dir : str
        Path to root directory where data is located.

    Returns
    -------
    timestamps : numpy.ndarray
        Timestamps (in seconds since Unix epoch), with each entry
        corresponding to each row in the `signals` data. The number of entries,
        `num_timestamps` is equal to `i2 - i1`.
    depths : numpy.ndarray
        Depths from the surface (in metres), with each entry corresponding
        to each column in the `signals` data.
    signals : numpy.ndarray
        Echogram Sv data, shaped `(num_timestamps, num_depths)`.
    top : numpy.ndarray
        Depth of top line, shaped `(num_timestamps, )`.
    bottom : numpy.ndarray
        Depth of bottom line, shaped `(num_timestamps, )`.
    '''
    root_data_dir = raw.loader.remove_trailing_slash(root_data_dir)
    root_shard_dir = os.path.join(root_data_dir + '_sharded', dataset)
    dirname = os.path.join(root_shard_dir, transect_rel_pth)
    return load_transect_from_shards_abs(
        dirname,
        i1=0,
        i2=None,
    )


# Backwards compatibility
load_transect_from_shards = load_transect_from_shards_rel
