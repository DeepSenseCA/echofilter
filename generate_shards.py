#!/usr/bin/env python
# coding: utf-8

import os

import tqdm

from echofilter import rawloader, shardloader


ROOT_DATA_DIR = rawloader.ROOT_DATA_DIR


def main(
        partition,
        dataset,
        partitioning_version='firstpass',
        max_depth=100.,
        shard_len=64,
        root_data_dir=ROOT_DATA_DIR,
        progress_bar=False,
    ):
    transect_pths = rawloader.get_partition_list(
        partition,
        dataset=dataset,
        full_path=False,
        partitioning_version=partitioning_version,
        root_data_dir=root_data_dir,
    )
    for transect_pth in (tqdm.tqdm(transect_pths) if progress_bar else transect_pths):
        try:
            shardloader.shard_transect(
                transect_pth,
                dataset=dataset,
                max_depth=max_depth,
                shard_len=shard_len,
                root_data_dir=root_data_dir,
            )
        except Exception as ex:
            print('Error sharding {}'.format(transect_pth))
            print(ex)


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
        default=rawloader.ROOT_DATA_DIR,
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
        type=int,
        default=100.,
        help='maximum depth to include in sharded data',
    )
    parser.add_argument(
        '--shard_len',
        type=float,
        default=64,
        help='number of samples in each shard',
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
    )
