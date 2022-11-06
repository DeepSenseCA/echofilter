"""
Converting raw data into shards, and loading data from shards.
"""

import os

import numpy as np

from . import loader, manipulate
from .utils import pad1d


ROOT_DATA_DIR = loader.ROOT_DATA_DIR


def segment_and_shard_transect(
    transect_pth,
    dataset="mobile",
    max_depth=None,
    shard_len=128,
    root_data_dir=ROOT_DATA_DIR,
):
    """
    Creates a sharded copy of a transect, with the transect cut into segments
    based on recording starts/stops. Each segment is split across multiple
    files (shards) for efficient loading.

    Parameters
    ----------
    transect_pth : str
        Relative path to transect, excluding `"_Sv_raw.csv"`.
    dataset : str, optional
        Name of dataset. Default is `"mobile"`.
    max_depth : float or None, optional
        The maximum depth to include in the saved shard. Data corresponding
        to deeper locations is omitted to save on load time and memory when
        the shard is loaded. If `None`, no cropping is applied.
        Default is `None`.
    shard_len : int, optional
        Number of timestamp samples to include in each shard. Default is `128`.
    root_data_dir : str
        Path to root directory where data is located.

    Notes
    -----
    The segments will be written to the directories
    `<root_data_dir>_sharded/<dataset>/transect_path/<segment>/`
    For the contents of each directory, see `write_transect_shards`.
    """
    # Define output destination
    root_data_dir = loader.remove_trailing_slash(root_data_dir)
    root_shard_dir = os.path.join(root_data_dir + "_sharded", dataset)

    # Load the data, with mask decomposed into turbulence, bottom, passive,
    # and removed regions.
    transect = manipulate.load_decomposed_transect_mask(
        os.path.join(root_data_dir, dataset, transect_pth)
    )

    segments = manipulate.split_transect(**transect)

    for i_segment, segment in enumerate(segments):
        dirname = os.path.join(root_shard_dir, transect_pth, str(i_segment))
        write_transect_shards(
            dirname, segment, max_depth=max_depth, shard_len=shard_len
        )

    n_segment = i_segment + 1

    # Save segmentation metadata
    with open(os.path.join(root_shard_dir, transect_pth, "n_segment.txt"), "w") as hf:
        print(str(n_segment), file=hf)


def write_transect_shards(dirname, transect, max_depth=None, shard_len=128):
    """
    Creates a sharded copy of a transect, with the transect cut by timestamp
    and split across multiple files.

    Parameters
    ----------
    dirname : str
        Path to output directory.
    transect : dict
        Observed values for the transect. Should already be segmented.
    max_depth : float or None, optional
        The maximum depth to include in the saved shard. Data corresponding
        to deeper locations is omitted to save on load time and memory when
        the shard is loaded. If `None`, no cropping is applied.
        Default is `None`.
    shard_len : int, optional
        Number of timestamp samples to include in each shard. Default is `128`.

    Notes
    -----
    The output will be written to the directory `dirname`, and will contain:

    - a file named `"shard_size.txt"`, which contains the sharding metadata:
      total number of samples, and shard size;
    - a directory for each shard, named 0, 1, ...
      Each shard directory will contain files:

        - depths.npy
        - timestamps.npy
        - Sv.npy
        - mask.npy
        - turbulence.npy
        - bottom.npy
        - is_passive.npy
        - is_removed.npy
        - is_upward_facing.npy

      which contain pickled numpy dumps of the matrices for each shard.
    """

    # Remove depths which are too deep for us to care about
    if max_depth is not None:
        depth_mask = transect["depths"] <= max_depth
        transect["depths"] = transect["depths"][depth_mask]
        transect["Sv"] = transect["Sv"][:, depth_mask]
        transect["mask"] = transect["mask"][:, depth_mask]

    # Reduce floating point precision for some variables
    for key in ("Sv", "turbulence", "bottom"):
        transect[key] = np.half(transect[key])

    # Ensure is_upward_facing is an array
    transect["is_upward_facing"] = np.array(transect["is_upward_facing"])

    # Prep output directory
    os.makedirs(dirname, exist_ok=True)

    # Save sharding metadata (total number of datapoints, shard size) to
    # make loading from the shards easier
    with open(os.path.join(dirname, "shard_size.txt"), "w") as hf:
        print("{},{}".format(transect["Sv"].shape[0], shard_len), file=hf)

    # Work out where to split the arrays
    indices = range(shard_len, transect["Sv"].shape[0], shard_len)

    # Split the transect into shards
    n_shards = len(indices) + 1
    shards = [{} for _ in range(n_shards)]
    for key in transect:
        if key in ("depths", "is_upward_facing") or not hasattr(
            transect[key], "__len__"
        ):
            for i_shards in range(n_shards):
                shards[i_shards][key] = transect[key]
        else:
            for i_split, split in enumerate(np.split(transect[key], indices)):
                shards[i_split][key] = split

    for shard in shards:
        if shard.keys() != shards[0].keys():
            raise ValueError("Inconsistent split lengths")

    # Save the data for each of the shards
    for i_shard, shard in enumerate(shards):
        fname = os.path.join(dirname, "{}.npz".format(i_shard))
        np.savez_compressed(fname, **shard)


def load_transect_from_shards_abs(
    transect_abs_pth,
    i1=0,
    i2=None,
    pad_mode="edge",
):
    """
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
    pad_mode : str, optional
        Padding method for out-of-bounds inputs. Must be supported by
        :meth:`numpy.pad`, such as `"contast"`, `"reflect"`, or `"edge"`. If the mode
        is `"contast"`, the array will be padded with zeros. Default is "edge".

    Returns
    -------
    dict
        A dictionary with keys:

            - "timestamps" : numpy.ndarray
                Timestamps (in seconds since Unix epoch), for each recording
                timepoint. The number of entries, `num_timestamps`, is equal
                to `i2 - i1`.
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
    # Load the sharding metadata
    with open(os.path.join(transect_abs_pth, "shard_size.txt"), "r") as f:
        n_timestamps, shard_len = f.readline().strip().split(",")
        n_timestamps = int(n_timestamps)
        shard_len = int(shard_len)
    # Set the default value for i2
    if i2 is None:
        i2 = n_timestamps

    # Sanity check
    if i1 > n_timestamps:
        raise ValueError(
            "All requested datapoints out of range: {}, {} > {}".format(
                i1, i2, n_timestamps
            )
        )
    if i2 < 0:
        raise ValueError(
            "All requested datapoints out of range: {}, {} < {}".format(i1, i2, 0)
        )
    # Make indices safe
    i1_ = max(0, i1)
    i2_ = min(i2, n_timestamps)
    # Work out which shards we'll need to load to get this data
    j1 = max(0, int(i1 / shard_len))
    j2 = int(min(i2, n_timestamps - 1) / shard_len)

    transect = {}

    shards = [
        np.load(os.path.join(transect_abs_pth, str(j) + ".npz"), allow_pickle=True)
        for j in range(j1, j2 + 1)
    ]
    # Depths and is_upward_facing should all be the same. Only load one of
    # each of them.
    for key in shards[0].keys():
        if key in ("depths", "is_upward_facing"):
            transect[key] = shards[0][key]
        else:
            broad_data = np.concatenate([shard[key] for shard in shards])
            # Have to trim data down, and pad if requested indices out of range
            transect[key] = pad1d(
                broad_data[(i1_ - j1 * shard_len) : (i2_ - j1 * shard_len)],
                (i1_ - i1, i2 - i2_),
                axis=0,
                mode=pad_mode,
            )

    return transect


def load_transect_from_shards_rel(
    transect_rel_pth,
    i1=0,
    i2=None,
    dataset="mobile",
    segment=0,
    root_data_dir=ROOT_DATA_DIR,
    **kwargs,
):
    """
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
        Name of dataset. Default is `"mobile"`.
    segment : int, optional
        Which segment to load. Default is `0`.
    root_data_dir : str
        Path to root directory where data is located.
    **kwargs
        As per :meth:`load_transect_from_shards_abs`.

    Returns
    -------
    dict
        See :meth:`load_transect_from_shards_abs`.
    """
    root_data_dir = loader.remove_trailing_slash(root_data_dir)
    root_shard_dir = os.path.join(root_data_dir + "_sharded", dataset)
    dirname = os.path.join(root_shard_dir, transect_rel_pth, str(segment))
    return load_transect_from_shards_abs(
        dirname,
        i1=i1,
        i2=i2,
        **kwargs,
    )


def load_transect_segments_from_shards_abs(
    transect_abs_pth,
    segments=None,
):
    """
    Load transect data from shard files.

    Parameters
    ----------
    transect_abs_pth : str
        Absolute path to transect shard segments directory.
    segments : iterable or None
        Which segments to load. If `None` (default), all segments are loaded.

    Returns
    -------
    dict
        See :meth:`load_transect_from_shards_abs`.
    """
    if segments is None:
        # Load the segmentation metadata
        with open(os.path.join(transect_abs_pth, "n_segment.txt"), "r") as f:
            n_segment = int(f.readline().strip())
        segments = range(n_segment)

    # Load each segment
    transects = []
    for segment in segments:
        dirname = os.path.join(transect_abs_pth, str(segment))
        transects.append(load_transect_from_shards_abs(dirname))

    # Join the segments together
    return manipulate.join_transect(transects)


def load_transect_segments_from_shards_rel(
    transect_rel_pth,
    dataset="mobile",
    segments=None,
    root_data_dir=ROOT_DATA_DIR,
):
    """
    Load transect data from shard files.

    Parameters
    ----------
    transect_rel_pth : str
        Relative path to transect.
    dataset : str, optional
        Name of dataset. Default is `"mobile"`.
    segments : iterable or None
        Which segments to load. If `None` (default), all segments are loaded.
    root_data_dir : str
        Path to root directory where data is located.
    **kwargs
        As per :meth:`load_transect_from_shards_abs`.

    Returns
    -------
    dict
        See :meth:`load_transect_from_shards_abs`.
    """
    root_data_dir = loader.remove_trailing_slash(root_data_dir)
    root_shard_dir = os.path.join(root_data_dir + "_sharded", dataset)
    dirname = os.path.join(root_shard_dir, transect_rel_pth)
    return load_transect_segments_from_shards_abs(dirname, segments=segments)


# Backwards compatibility
shard_transect = segment_and_shard_transect
load_transect_from_shards = load_transect_from_shards_rel
