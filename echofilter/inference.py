#!/usr/bin/env python

from collections import OrderedDict
import os
import datetime
import pprint
import shutil
import sys
import time

import appdirs
import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.utils.data
import torchvision.transforms
from torchvision.datasets.utils import download_url, download_file_from_google_drive
from torchutils.utils import count_parameters
from torchutils.device import cuda_is_really_available

import echofilter.raw.loader
import echofilter.transforms
import echofilter.utils
from echofilter.unet import UNet
from echofilter.wrapper import Echofilter


DATA_MEAN = -80.
DATA_STDEV = 20.

CHECKPOINT_RESOURCES = OrderedDict([
    ('stationary_effunet_block6.xb.2-1_lc32_se2.ckpt.tar', ('gdrive', '114vL-pAxrn9UDhaNG5HxZwjxNy7WMfW_')),
])
DEFAULT_CHECKPOINT = next(iter(CHECKPOINT_RESOURCES))


def inference(
        files,
        checkpoint=DEFAULT_CHECKPOINT,
        data_dir='.',
        output_dir='processed',
        image_height=None,
        crop_depth=None,
        device=None,
        cache_dir=None,
    ):

    if device is None:
        device = 'cuda' if cuda_is_really_available() else 'cpu'
    device = torch.device(device)

    dtype = torch.float

    if checkpoint is None:
        # Use the first item from the list of checkpoints
        checkpoint = DEFAULT_CHECKPOINT

    if os.path.isfile(checkpoint):
        checkpoint_path = checkpoint
    elif checkpoint in CHECKPOINT_RESOURCES:
        checkpoint_path = download_checkpoint(checkpoint, cache_dir=cache_dir)
    else:
        raise ValueError(
            'The checkpoint parameter should either be a path to a file or '
            'one of \n{},\nbut {} was provided.'
            .format(list(CHECKPOINT_RESOURCES.keys()), checkpoint)
        )

    # Preprocessing transforms
    transform = torchvision.transforms.Compose([
        echofilter.transforms.Normalize(DATA_MEAN, DATA_STDEV),
        echofilter.transforms.ReplaceNan(-3),
    ])

    if not os.path.isfile(checkpoint_path):
        raise EnvironmentError("No checkpoint found at '{}'".format(checkpoint_path))
    print("Loading checkpoint '{}'".format(checkpoint_path))
    if device is None:
        checkpoint = torch.load(checkpoint_path)
    else:
        # Map model to be loaded to specified single gpu.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if image_height is None:
        image_height = checkpoint.get('sample_shape', (128, 512))[1]

    print('Constructing U-Net model, with parameters:')
    pprint.pprint(checkpoint['model_parameters'])
    model = Echofilter(
        UNet(**checkpoint['model_parameters']),
        top='boundary',
        bottom='boundary',
    )
    print(
        'Built model with {} trainable parameters'
        .format(count_parameters(model, only_trainable=True))
    )
    model.load_state_dict(checkpoint['state_dict'])
    print(
        "Loaded checkpoint '{}' (epoch {})"
        .format(checkpoint_path, checkpoint['epoch'])
    )
    # Ensure model is on correct device
    model.to(device)
    # Put model in evaluation mode
    model.eval()

    files = list(parse_files_in_folders(files, data_dir))
    print('Processing {} file{}'.format(len(files), '' if len(files) == 1 else 's'))

    for fname in files:
        # Check what the full path should be
        if os.path.isfile(fname):
            fname_full = fname
        elif os.path.isfile(os.path.join(data_dir, fname)):
            fname_full = os.path.join(data_dir, fname)
        else:
            raise EnvironmentError('Could not locate file {}'.format(fname))
        # Load the data
        timestamps, depths, signals = echofilter.raw.loader.transect_loader(fname_full)
        data = {
            'timestamps': timestamps,
            'depths': depths,
            'signals': signals,
        }
        if crop_depth is not None:
            # Apply depth crop
            depth_crop_mask = data['depths'] <= crop_depth
            data['depths'] = data['depths'][depth_crop_mask]
            data['signals'] = data['signals'][:, depth_crop_mask]

        # Configure data to match what the model expects to see
        # Determine whether depths are ascending or descending
        is_upward_facing = (data['depths'][-1] < data['depths'][0])
        # Ensure depth is always increasing (which corresponds to descending from
        # the air down the water column)
        if is_upward_facing:
            data['depths'] = data['depths'][::-1].copy()
            data['signals'] = data['signals'][:, ::-1].copy()
        # Apply transforms
        data = transform(data)
        data = echofilter.transforms.Rescale((signals.shape[0], image_height))(data)
        input = torch.tensor(data['signals']).unsqueeze(0).unsqueeze(0)
        input = input.to(device, dtype)
        # Put data through model
        with torch.no_grad():
            output = model(input)
            output = {k: v.squeeze(0).cpu().numpy() for k, v in output.items()}
        # Convert output into lines
        top_depths = data['depths'][echofilter.utils.last_nonzero(output['p_is_above_top'] > 0.5, -1)]
        bottom_depths = data['depths'][echofilter.utils.first_nonzero(output['p_is_below_bottom'] > 0.5, -1)]
        # Export evl files
        if output_dir is None or output_dir == '':
            destination = fname
        else:
            destination = os.path.join(output_dir, fname)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        print(destination + '.top.evl')
        echofilter.raw.loader.evl_writer(destination + '.top.evl', timestamps, top_depths)
        print(destination + '.bottom.evl')
        echofilter.raw.loader.evl_writer(destination + '.bottom.evl', timestamps, bottom_depths)


def parse_files_in_folders(files_or_folders, data_dir, extension='csv'):
    '''
    Walk through folders and find suitable files.

    Parameters
    ----------
    files_or_folders : iterable
        List of files and folders.
    data_dir : str
        Root directory within which elements of `files_or_folders` may
        be found.
    extension : str, optional
        Extension which files within directories must bear to be included.
        Explicitly given files are always used. Default is `'csv'`.

    Yields
    ------
    str
        Paths to explicitly given files and files within directories with
        extension `extension`.
    '''
    for path in files_or_folders:
        if os.path.isfile(path) or os.path.isfile(os.path.join(data_dir, path)):
            yield path
            continue
        elif os.path.isdir(path):
            folder = path
        elif os.path.isdir(os.path.join(data_dir, path)):
            folder = os.path.join(data_dir, path)
        else:
            raise EnvironmentError('Missing file or directory: {}'.format(path))
        for dirpath, dirnames, filenames in os.walk(folder):
            for filename in filenames:
                if not os.path.isfile(os.path.join(folder, dirpath, filename)):
                    continue
                if extension is None or os.path.splitext(filename)[1][1:] == extension:
                    yield os.path.join(path, dirpath, filename)


def get_default_cache_dir():
    '''Determine the default cache directory.'''
    return appdirs.user_cache_dir('echofilter', 'DeepSense')


def download_checkpoint(checkpoint_name, cache_dir=None):
    '''
    Download a checkpoint if it isn't already cached.

    Parameters
    ----------
    checkpoint_name : str
        Name of checkpoint to download.
    cache_dir : str or None, optional
        Path to local cache directory. If `None` (default), an OS-appropriate
        default cache directory is used.

    Returns
    -------
    str
        Path to downloaded checkpoint file.
    '''
    if cache_dir is None:
        cache_dir = get_default_cache_dir()

    destination = os.path.join(cache_dir, checkpoint_name)

    if os.path.exists(destination):
        return destination

    os.makedirs(cache_dir, exist_ok=True)

    type, url_or_id = CHECKPOINT_RESOURCES[checkpoint_name]

    if type == 'gdrive':
        print('Downloading checkpoint {} from GDrive...'.format(checkpoint_name))
        download_file_from_google_drive(url_or_id, cache_dir, filename=checkpoint_name)
    else:
        print('Downloading checkpoint {} from {}...'.format(checkpoint_name, url_or_id))
        download_url(url_or_id, cache_dir, filename=checkpoint_name)

    print('Downloaded checkpoint to {}'.format(destination))

    return destination


def main():
    import argparse

    prog = os.path.split(sys.argv[0])[1]
    if prog == '__main__.py':
        prog = 'echofilter'
    parser = argparse.ArgumentParser(
        prog=prog,
        description=echofilter.__meta__.description,
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s {version}'.format(version=echofilter.__version__)
    )
    parser.add_argument(
        'files',
        type=str,
        nargs='+',
        default=[],
        metavar='FILE',
        help=
            'file(s) to process. For each directory given, all csv files'
            ' within that directory and its subdirectories will be processed.'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='.',
        metavar='DIR',
        help='path to directory containing FILE (default: ".")',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='processed',
        metavar='DIR',
        help=
            'path to output directory. If empty, output is placed in the same'
            ' directory as the input file. (default: "processed")',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=DEFAULT_CHECKPOINT,
        metavar='PATH',
        help=
            'path to checkpoint to load, or name of checkpoint available to'
            ' download (default: "{}")'.format(DEFAULT_CHECKPOINT),
    )
    DEFAULT_CACHE_DIR = get_default_cache_dir()
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=
            'path to checkpoint cache directory (default: "{}")'
            .format(DEFAULT_CACHE_DIR),
    )
    parser.add_argument(
        '--image-height', '--height',
        dest='image_height',
        type=float,
        default=None,
        help=
            'input image height, in pixels. The echogram will be resized to'
            ' have this height, and its width will be kept. (default: same'
            ' as using during training)',
    )
    parser.add_argument(
        '--crop-depth',
        type=float,
        default=None,
        help='depth, in metres, at which data should be truncated (default: None)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='device to use (default: use first gpu if available, otherwise cpu)',
    )

    inference(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
