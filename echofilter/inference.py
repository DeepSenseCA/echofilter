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
from tqdm.auto import tqdm

import echofilter.raw
from echofilter.raw.manipulate import join_transect, split_transect
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
        row_len_selector='mode',
        crop_depth=None,
        device=None,
        cache_dir=None,
        keep_ext=False,
        verbose=1,
    ):

    if device is None:
        device = 'cuda' if cuda_is_really_available() else 'cpu'
    device = torch.device(device)

    if checkpoint is None:
        # Use the first item from the list of checkpoints
        checkpoint = DEFAULT_CHECKPOINT

    if os.path.isfile(checkpoint):
        checkpoint_path = checkpoint
    elif checkpoint in CHECKPOINT_RESOURCES:
        checkpoint_path = download_checkpoint(checkpoint, cache_dir=cache_dir, verbose=verbose)
    else:
        raise ValueError(
            'The checkpoint parameter should either be a path to a file or '
            'one of \n{},\nbut {} was provided.'
            .format(list(CHECKPOINT_RESOURCES.keys()), checkpoint)
        )

    if not os.path.isfile(checkpoint_path):
        raise EnvironmentError("No checkpoint found at '{}'".format(checkpoint_path))
    if verbose >= 1:
        print("Loading checkpoint '{}'".format(checkpoint_path))
    if device is None:
        checkpoint = torch.load(checkpoint_path)
    else:
        # Map model to be loaded to specified single gpu.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if image_height is None:
        image_height = checkpoint.get('sample_shape', (128, 512))[1]

    if verbose >= 2:
        print('Constructing U-Net model, with arguments:')
        pprint.pprint(checkpoint['model_parameters'])
    model = Echofilter(
        UNet(**checkpoint['model_parameters']),
        top='boundary',
        bottom='boundary',
    )
    if verbose >= 1:
        print(
            'Built model with {} trainable parameters'
            .format(count_parameters(model, only_trainable=True))
        )
    model.load_state_dict(checkpoint['state_dict'])
    if verbose >= 1:
        print(
            "Loaded checkpoint state from '{}' (epoch {})"
            .format(checkpoint_path, checkpoint['epoch'])
        )
    # Ensure model is on correct device
    model.to(device)
    # Put model in evaluation mode
    model.eval()

    files = list(parse_files_in_folders(files, data_dir))
    if verbose >= 1:
        print('Processing {} file{}'.format(len(files), '' if len(files) == 1 else 's'))

    if len(files) == 1:
        maybe_tqdm = lambda x: x
    else:
        maybe_tqdm = tqdm

    for fname in maybe_tqdm(files):
        if verbose >= 2:
            print('Processing {}'.format(fname))
        # Check what the full path should be
        if os.path.isfile(fname):
            fname_full = fname
        elif os.path.isfile(os.path.join(data_dir, fname)):
            fname_full = os.path.join(data_dir, fname)
        else:
            raise EnvironmentError('Could not locate file {}'.format(fname))
        # Load the data
        if verbose >= 4:
            warn_row_overflow = np.inf
        elif verbose >= 3:
            warn_row_overflow = None
        else:
            warn_row_overflow = 0
        timestamps, depths, signals = echofilter.raw.loader.transect_loader(
            fname_full,
            warn_row_overflow=warn_row_overflow,
            row_len_selector=row_len_selector,
        )
        output = inference_transect(
            model,
            timestamps,
            depths,
            signals,
            device,
            image_height,
            crop_depth=crop_depth,
        )
        # Convert output into lines
        top_depths = output['depths'][echofilter.utils.last_nonzero(output['p_is_above_top'] > 0.5, -1)]
        bottom_depths = output['depths'][echofilter.utils.first_nonzero(output['p_is_below_bottom'] > 0.5, -1)]
        # Export evl files
        if output_dir is None or output_dir == '':
            destination = fname_full
        elif os.path.isabs(fname):
            destination = os.path.join(output_dir, os.path.split(fname)[1])
        elif os.path.abspath(fname).startswith(os.path.abspath(os.path.join(data_dir, ''))):
            destination = os.path.join(
                output_dir,
                os.path.abspath(fname)[len(os.path.abspath(os.path.join(data_dir, ''))):],
            )
        else:
            destination = os.path.join(output_dir, fname)
        if not keep_ext:
            destination = os.path.splitext(destination)[0]
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if verbose >= 2:
            print('Writing output {}'.format(destination + '.top.evl'))
        echofilter.raw.loader.evl_writer(destination + '.top.evl', timestamps, top_depths)
        if verbose >= 2:
            print('Writing output {}'.format(destination + '.bottom.evl'))
        echofilter.raw.loader.evl_writer(destination + '.bottom.evl', timestamps, bottom_depths)

    if verbose >= 1:
        print('Finished processing {} file{}'.format(len(files), '' if len(files) == 1 else 's'))


def inference_transect(
    model,
    timestamps,
    depths,
    signals,
    device,
    image_height,
    crop_depth=None,
    dtype=torch.float,
):
    '''
    Run inference on a single transect.

    Parameters
    ----------
    model : echofilter.wrapper.Echofilter
        A pytorch Module wrapped in an Echofilter UI layer.
    timestamps : array_like
        Sample recording timestamps (in seconds since Unix epoch). Must be a
        vector.
    depths : array_like
        Recording depths from the surface (in metres). Must be a vector.
    signals : array_like
        Echogram Sv data. Must be a matrix shaped
        `(len(timestamps), len(depths))`.
    image_height : int
        Height to resize echogram before passing through model.
    crop_depth : float or None, optional
        Maximum depth at which to crop input.
    dtype : torch.dtype, optional
        Datatype to use for model input. Default is `torch.float`.

    Returns
    -------
    dict
        Dictionary with fields as output by `echofilter.wrapper.Echofilter`,
        plus `timestamps` and `depths`.
    '''
    timestamps = np.asarray(timestamps)
    depths = np.asarray(depths)
    signals = np.asarray(signals)
    transect = {
        'timestamps': timestamps,
        'depths': depths,
        'signals': signals,
    }
    if crop_depth is not None:
        # Apply depth crop
        depth_crop_mask = transect['depths'] <= crop_depth
        transect['depths'] = transect['depths'][depth_crop_mask]
        transect['signals'] = transect['signals'][:, depth_crop_mask]

    # Configure data to match what the model expects to see
    # Determine whether depths are ascending or descending
    is_upward_facing = (transect['depths'][-1] < transect['depths'][0])
    # Ensure depth is always increasing (which corresponds to descending from
    # the air down the water column)
    if is_upward_facing:
        transect['depths'] = transect['depths'][::-1].copy()
        transect['signals'] = transect['signals'][:, ::-1].copy()

    # To reduce memory consumption, split into segments whenever the recording
    # interval is longer than normal
    outputs = []
    for segment in split_transect(threshold=20, **transect):
        # Preprocessing transform
        transform = torchvision.transforms.Compose([
            echofilter.transforms.Normalize(DATA_MEAN, DATA_STDEV),
            echofilter.transforms.ReplaceNan(-3),
            echofilter.transforms.Rescale((segment['signals'].shape[0], image_height)),
        ])
        segment = transform(segment)
        input = torch.tensor(segment['signals']).unsqueeze(0).unsqueeze(0)
        input = input.to(device, dtype)
        # Put data through model
        with torch.no_grad():
            output = model(input)
            output = {k: v.squeeze(0).cpu().numpy() for k, v in output.items()}
        output['timestamps'] = segment['timestamps']
        output['depths'] = segment['depths']
        outputs.append(output)

    return join_transect(outputs)


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


def download_checkpoint(checkpoint_name, cache_dir=None, verbose=1):
    '''
    Download a checkpoint if it isn't already cached.

    Parameters
    ----------
    checkpoint_name : str
        Name of checkpoint to download.
    cache_dir : str or None, optional
        Path to local cache directory. If `None` (default), an OS-appropriate
        default cache directory is used.
    verbose : int, optional
        Verbosity level. Default is `1`. Set to `0` to disable print
        statements.

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
        if verbose > 0:
            print('Downloading checkpoint {} from GDrive...'.format(checkpoint_name))
        download_file_from_google_drive(url_or_id, cache_dir, filename=checkpoint_name)
    else:
        if verbose > 0:
            print('Downloading checkpoint {} from {}...'.format(checkpoint_name, url_or_id))
        download_url(url_or_id, cache_dir, filename=checkpoint_name)

    if verbose > 0:
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
        metavar='PATH',
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
        '--keep-ext',
        action='store_true',
        help='keep the input file extension in output file names',
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
        '--row-len-selector',
        type=str,
        choices=['init', 'min', 'max', 'median', 'mode'],
        default='mode',
        help='how to handle inputs with differing number of samples across time (default: "mode")',
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
    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=1,
        help='increase verbosity, print more progress details',
    )
    parser.add_argument(
        '--quiet', '-q',
        action='count',
        default=0,
        help='decrease verbosity, print fewer progress details',
    )
    kwargs = vars(parser.parse_args())
    kwargs['verbose'] -= kwargs.pop('quiet', 0)

    inference(**kwargs)


if __name__ == '__main__':
    main()
