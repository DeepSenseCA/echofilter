#!/usr/bin/env python
# coding: utf-8

import functools
import multiprocessing
import os
import traceback

import tqdm

import echofilter.raw.loader
import echofilter.shardloader


ROOT_DATA_DIR = echofilter.raw.loader.ROOT_DATA_DIR


def single(
    transect_pth,
    verbose=False,
    fail_gracefully=True,
    **kwargs,
):
    '''
    Shard a single transect.

    Wrapper around echofilter.shardloader.segment_and_shard_transect which
    adds verboseness and graceful failure options.
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


def main(
        partition,
        dataset,
        partitioning_version='firstpass',
        max_depth=100.,
        shard_len=128,
        root_data_dir=ROOT_DATA_DIR,
        progress_bar=False,
        ncores=None,
        verbose=False,
        fail_gracefully=True,
    ):
    '''
    Shard all transections in one partition of a dataset.
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
        maybe_tqdm = lambda x: tqdm.tqdm(x, total=len(session_paths))
    else:
        maybe_tqdm = lambda x: x

    fn = functools.partial(
        single,
        dataset=dataset,
        max_depth=max_depth,
        shard_len=shard_len,
        root_data_dir=root_data_dir,
        verbose=verbose,
        fail_gracefully=fail_gracefully,
    )
    if ncores == 1:
        for transect_pth in maybe_tqdm(transect_pths):
            fn(transect_pth)
    else:
        with multiprocessing.Pool(ncores) as pool:
            for _ in maybe_tqdm(pool.imap_unordered(fn, transect_pths)):
                pass


if __name__ == '__main__':
    import argparse

    # Create parser
    parser = argparse.ArgumentParser(
        description='Generate dataset shards',
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
    main(**vars(args))
