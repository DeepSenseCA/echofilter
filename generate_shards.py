#!/usr/bin/env python
# coding: utf-8

import os
import traceback

import tqdm

import echofilter.raw.loader
import echofilter.shardloader


ROOT_DATA_DIR = echofilter.raw.loader.ROOT_DATA_DIR


def main(
        partition,
        dataset,
        partitioning_version='firstpass',
        max_depth=100.,
        shard_len=64,
        root_data_dir=ROOT_DATA_DIR,
        progress_bar=False,
        verbose=False,
    ):
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
    for transect_pth in (tqdm.tqdm(transect_pths) if progress_bar else transect_pths):
        if verbose:
            print('Sharding {}'.format(transect_pth))
        try:
            echofilter.shardloader.segment_and_shard_transect(
                transect_pth,
                dataset=dataset,
                max_depth=max_depth,
                shard_len=shard_len,
                root_data_dir=root_data_dir,
            )
        except Exception as ex:
            print('Error sharding {}'.format(transect_pth))
            print("".join(traceback.TracebackException.from_exception(ex).format()))


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
        default=64,
        help='number of samples in each shard',
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
    main(
        args.partition,
        args.dataset,
        partitioning_version=args.partitioning_version,
        max_depth=args.max_depth,
        shard_len=args.shard_len,
        root_data_dir=args.root,
        verbose=args.verbose,
    )
