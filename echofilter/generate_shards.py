#!/usr/bin/env python
# coding: utf-8

import functools
import multiprocessing
import os
import traceback
import sys

import echofilter.raw.loader
import echofilter.shardloader


ROOT_DATA_DIR = echofilter.raw.loader.ROOT_DATA_DIR


def generate_shard(
    transect_pth,
    verbose=False,
    fail_gracefully=True,
    **kwargs,
):
    '''
    Shard a single transect.

    Wrapper around echofilter.shardloader.segment_and_shard_transect which
    adds verboseness and graceful failure options.

    Parameters
    ----------
    transect_pth : str
        Relative path to transect.
    verbose : bool, optional
        Whether to print which transect is being processed. Default is `False`.
    fail_gracefully : bool, optional
        If `True`, any transect which triggers an errors during processing
        will be printed out, but processing the rest of the transects will
        continue. If `False`, the process will halt with an error as soon as
        any single transect hits an error. Default is `True`.
    *kwargs
        See `echofilter.shardloader.segment_and_shard_transect`.
    '''
    if verbose:
        print('Sharding {}'.format(transect_pth))
    try:
        echofilter.shardloader.segment_and_shard_transect(
            transect_pth,
            **kwargs,
        )
    except Exception as ex:
        if not fail_gracefully:
            raise ex
        print('Error sharding {}'.format(transect_pth))
        print("".join(traceback.TracebackException.from_exception(ex).format()))


def generate_shards(
    partition,
    dataset,
    partitioning_version='firstpass',
    progress_bar=False,
    ncores=None,
    verbose=False,
    fail_gracefully=True,
    root_data_dir=ROOT_DATA_DIR,
    **kwargs,
):
    '''
    Shard all transections in one partition of a dataset.

    Wrapper around echofilter.shardloader.segment_and_shard_transect which
    adds verboseness and graceful failure options.

    Parameters
    ----------
    partition : str
        Name of the partition to process (`'train'`, `'validate'`, `'test'`,
        etc).
    dataset : str
        Name of the dataset to process (`'mobile'`, `'stationary'`, etc).
    partitioning_version : str, optional
        Name of the partition version to use process. Default is `'firstpass'`.
    progress_bar : bool, optional
        Whether to output a progress bar using `tqdm`. Default is `False`.
    ncores : int, optional
        Number of cores to use for multiprocessing. To disable multiprocessing,
        set to `1`. Set to `None` to use all available cores.
        Default is `None`.
    verbose : bool, optional
        Whether to print which transect is being processed. Default is `False`.
    fail_gracefully : bool, optional
        If `True`, any transect which triggers an errors during processing
        will be printed out, but processing the rest of the transects will
        continue. If `False`, the process will halt with an error as soon as
        any single transect hits an error. Default is `True`.
    *kwargs
        See `echofilter.shardloader.segment_and_shard_transect`.
    '''
    if verbose:
        print('Getting partition list "{}" for "{}"'.format(partition, dataset))
    transect_pths = echofilter.raw.loader.get_partition_list(
        partition,
        dataset=dataset,
        full_path=False,
        partitioning_version=partitioning_version,
        root_data_dir=root_data_dir,
    )
    if verbose:
        print('Will process {} transects'.format(len(transect_pths)))
        print()

    if progress_bar:
        from tqdm.autonotebook import tqdm
        maybe_tqdm = lambda x: tqdm(x, total=len(session_paths))
    else:
        maybe_tqdm = lambda x: x

    fn = functools.partial(
        generate_shard,
        dataset=dataset,
        verbose=verbose,
        fail_gracefully=fail_gracefully,
        root_data_dir=root_data_dir,
        **kwargs,
    )
    if ncores == 1:
        for transect_pth in maybe_tqdm(transect_pths):
            fn(transect_pth)
    else:
        with multiprocessing.Pool(ncores) as pool:
            for _ in maybe_tqdm(pool.imap_unordered(fn, transect_pths)):
                pass


def main():
    import argparse

    # Create parser

    prog = os.path.split(sys.argv[0])[1]
    if prog == '__main__.py' or prog == '__main__':
        prog = os.path.split(__file__)[1]
    parser = argparse.ArgumentParser(
        prog=prog,
        description='Generate dataset shards',
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s {version}'.format(version=echofilter.__version__)
    )
    parser.add_argument(
        'partition',
        type=str,
        help='partition to shard',
    )
    parser.add_argument(
        'dataset',
        type=str,
        help='dataset to shard',
    )
    parser.add_argument(
        '--root',
        dest='root_data_dir',
        type=str,
        default=ROOT_DATA_DIR,
        help='root data directory',
    )
    parser.add_argument(
        '--partitioning_version',
        type=str,
        default='firstpass',
        help='partitioning version',
    )
    parser.add_argument(
        '--max_depth',
        type=float,
        default=100.,
        help='maximum depth to include in sharded data',
    )
    parser.add_argument(
        '--shard_len',
        type=int,
        default=128,
        help='number of samples in each shard',
    )
    parser.add_argument(
        '--ncores',
        type=int,
        default=None,
        help=
            'number of cores to use (default: all). Set to 1 to disable'
            ' multiprocessing.',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=0,
        help='increase verbosity',
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Check the input directory exists
    print("Sharding {} partition of {}".format(args.partition, args.dataset))

    # Run command with these arguments
    generate_shards(**vars(args))


if __name__ == '__main__':
    main()