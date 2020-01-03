import os
import numpy as np

from . import rawloader


ROOT_DATA_DIR = rawloader.ROOT_DATA_DIR


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
    <root_data_dir>/<dataset>_sharded/transect_path
    and will contain
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
    root_shard_dir = os.path.join(root_data_dir, dataset + '_sharded')
    # Load the raw data
    timestamps, depths, signals, d_top, d_bot = rawloader.load_transect_data(
        transect_pth, dataset, root_data_dir,
    )
    # Prep
    depth_mask = depths <= 100
    indices = range(shard_len, signals.shape[0], shard_len)
    dirname = os.path.join(root_shard_dir, transect_pth)
    os.makedirs(dirname, exist_ok=True)
    # Save sharding metadata (total number of datapoints, shard size) to
    # make loading from the shards easier
    with open(os.path.join(dirname, 'shard_size.txt'), 'w') as hf:
        print('{},{}'.format(len(timestamps), shard_len), file=hf)
    # Save the data for each of the shards
    for i, (ts_i, sig_i, top_i, bot_i) in enumerate(
            zip(
                np.split(timestamps, indices),
                np.split(np.single(signals[:, depth_mask]), indices),
                np.split(np.single(d_top), indices),
                np.split(np.single(d_bot), indices),
            )
            ):
        os.makedirs(os.path.join(dirname, str(i)), exist_ok=True)
        for obj, fname in (
                (depths[depth_mask], 'depths'), (ts_i, 'timestamps'),
                (sig_i, 'Sv'), (top_i, 'top'), (bot_i, 'bottom')):
            obj.dump(os.path.join(dirname, str(i), fname + '.npy'))


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

    # Depths should all be the same. Only load one of them.
    depths = np.load(os.path.join(transect_abs_pth, str(j1), 'depths.npy'), allow_pickle=True)

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

    timestamps = load_shard('timestamps')
    signals = load_shard('Sv')
    d_top = load_shard('top')
    d_bot = load_shard('bottom')

    return timestamps, depths, signals, d_top, d_bot


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
    root_shard_dir = os.path.join(root_data_dir, dataset + '_sharded')
    dirname = os.path.join(root_shard_dir, transect_rel_pth)
    return load_transect_from_shards_abs(
        dirname,
        i1=0,
        i2=None,
    )


# Backwards compatibility
load_transect_from_shards = load_transect_from_shards_rel
