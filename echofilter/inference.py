#!/usr/bin/env python

from collections import OrderedDict
import os
import datetime
import pickle
import pprint
import shutil
import sys
import tempfile
import time
import urllib

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

import echofilter.ev
import echofilter.path
import echofilter.raw
from echofilter.raw.manipulate import join_transect, split_transect
import echofilter.transforms
import echofilter.utils
from echofilter.unet import UNet
from echofilter.wrapper import Echofilter


DATA_MEAN = -80.
DATA_STDEV = 20.

CHECKPOINT_RESOURCES = OrderedDict([
    ('stationary_effunet_block6.xb.2-1_lc32_se2_v2.ckpt.tar', {'gdrive': '1Rgr6y7SYEYrAq6tSF7tjqbKoZthKpMCb'}),
    ('stationary_effunet_block6.xb.2-1_lc32_se2.ckpt.tar', {'gdrive': '114vL-pAxrn9UDhaNG5HxZwjxNy7WMfW_'}),
])
DEFAULT_CHECKPOINT = next(iter(CHECKPOINT_RESOURCES))

DEFAULT_VARNAME = "Fileset1: Sv pings T1"


def run_inference(
    files,
    data_dir='.',
    checkpoint=None,
    output_dir='',
    variable_name=DEFAULT_VARNAME,
    image_height=None,
    facing="auto",
    row_len_selector='mode',
    crop_depth_min=None,
    crop_depth_max=None,
    extensions='csv',
    keep_ext=False,
    overwrite_existing=False,
    skip_existing=False,
    skip_incompatible=False,
    minimize_echoview=False,
    hide_echoview="new",
    device=None,
    cache_dir=None,
    cache_csv=None,
    csv_suffix='.csv',
    verbose=1,
    dry_run=False,
):
    '''
    Perform inference on input files, and write output lines in evl format.

    Parameters
    ----------
    files : iterable
        Files and folders to be processed. These may be full paths or paths
        relative to `data_dir`. For each folder specified, any files with
        extension `'csv'` within the folder and all its tree of subdirectories
        will be processed.
    data_dir : str, optional
        Path to directory where files are found. Default is `'.'`.
    checkpoint : str or None, optional
        A path to a checkpoint file, or name of a checkpoint known to this
        package (listed in `CHECKPOINT_RESOURCES`). If `None` (default),
        the first checkpoint in `CHECKPOINT_RESOURCES` is used.
    output_dir : str, optional
        Directory where output files will be written. If this is `''`, outputs
        are written to the same directory as each input file. Otherwise, they
        are written to `output_dir`, preserving their path relative to
        `data_dir` if relative paths were used. Default is `''`.
    variable_name : str, optional
        Name of the EchoView acoustic variable to load from EV files. Default
        is `'Fileset1: Sv pings T1'`.
    image_height : int or None, optional
        Height in pixels of input to model. The data loaded from the csv will
        be resized to this height (the width of the image is unchanged).
        If `None` (default), the height matches that used when the model was
        trained.
    facing : {"downward", "upward", "auto"}, optional
        Orientation in which the echosounder is facing. Default is `"auto"`,
        in which case the orientation is determined from the ordering of the
        depth values in the data (increasing = `"upward"`,
        decreasing = `"downward"`).
    row_len_selector : str, optional
        Method used to handle input csv files with different number of Sv
        values across time (i.e. a non-rectangular input). Default is `'mode'`.
        See `echofilter.raw.loader.transect_loader` for options.
    crop_depth_min : float or None, optional
        Minimum depth to include in input. If `None` (default), there is no
        minimum depth.
    crop_depth_max : float or None, optional
        Maxmimum depth to include in input. If `None` (default), there is no
        maximum depth.
    extensions : iterable or str, optional
        File extensions to detect when running on a directory. Default is
        `'csv'`.
    keep_ext : bool, optional
        Whether to preserve the file extension in the input file name when
        generating output file name. Default is `False`, removing the
        extension.
    overwrite_existing : bool, optional
        Overwrite existing outputs without producing a warning message. If
        `False`, an error is generated if files would be overwritten.
        Default is `False`.
    skip_existing : bool, optional
        Skip processing files which already have all outputs present. Default
        is `False`.
    skip_incompatible : bool, optional
        Skip processing CSV files which do not seem to contain an exported
        echoview transect. If `False`, an error is raised. Default is `False`.
    minimize_echoview : bool, optional
        If `True`, the Echoview window being used will be minimized while this
        function is running. Default is `False`.
    hide_echoview : {"never", "new", "always"}, optional
        Whether to hide the Echoview window entirely while the code runs.
        If `hide_echoview="new"`, the application is only hidden if it
        was created by this function, and not if it was already running.
        If `hide_echoview="always"`, the application is hidden even if it was
        already running. In the latter case, the window will be revealed again
        when this function is completed. Default is `"new"`.
    device : str or torch.device or None, optional
        Name of device on which the model will be run. If `None`, the first
        available CUDA GPU is used if any are found, and otherwise the CPU is
        used. Set to `'cpu'` to use the CPU even if a CUDA GPU is available.
    cache_dir : str or None, optional
        Path to directory where downloaded checkpoint files should be cached.
        If `None` (default), an OS-appropriate application-specific default
        cache directory is used.
    cache_csv : str or None, optional
        Path to directory where CSV files generated from EV inputs should be
        cached. If `None` (default), EV files which are exported to CSV files
        are temporary files, deleted after this program has completed. If
        `cache_csv=''`, the CSV files are cached in the same directory as the
        input EV files.
    csv_suffix : str, optional
        Suffix used for cached CSV files which are exported from EV files.
        Default is `'.csv'` (only the file extension is changed).
    verbose : int, optional
        Verbosity level. Default is `1`. Set to `0` to disable print
        statements, or elevate to a higher number to increase verbosity.
    dry_run : bool, optional
        If `True`, perform a trial run with no changes made. Default is
        `False`.
    '''

    if device is None:
        device = 'cuda' if cuda_is_really_available() else 'cpu'
    device = torch.device(device)

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

    if not os.path.isfile(checkpoint_path):
        raise EnvironmentError("No checkpoint found at '{}'".format(checkpoint_path))
    if verbose >= 1:
        print("Loading checkpoint '{}'".format(checkpoint_path))

    load_args = {}
    if device is not None:
        # Map model to be loaded to specified single gpu.
        load_args = dict(map_location=device)
    try:
        checkpoint = torch.load(checkpoint_path, **load_args)
    except pickle.UnpicklingError:
        if checkpoint not in CHECKPOINT_RESOURCES or checkpoint == checkpoint_path:
            # Direct path to checkpoint was given, so we shouldn't delete
            # the user's file
            print('Error: Unable to load checkpoint {}'.format(os.path.abspath(checkpoint_path)))
            raise
        # Delete the checkpoint and try again, in case it is just a
        # malformed download (interrupted download, etc)
        os.remove(checkpoint_path)
        checkpoint_path = download_checkpoint(checkpoint, cache_dir=cache_dir)
        checkpoint = torch.load(checkpoint_path, **load_args)

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

    files_input = files
    files = list(echofilter.path.parse_files_in_folders(files, data_dir, extensions))
    if verbose >= 1:
        print('Processing {} file{}'.format(len(files), '' if len(files) == 1 else 's'))

    if len(extensions) == 1 and 'ev' in extensions:
        do_open = True
    else:
        do_open = False
        for file in files:
            if os.path.splitext(file)[1].lower() == '.ev':
                do_open = True
                break

    if dry_run:
        if verbose >= 3:
            print(
                'Echoview application would{} be opened {}.'
                .format(
                    '' if do_open else ' not',
                    'to convert EV files to CSV' if do_open else '(no EV files to process)',
                )
            )
        do_open = False

    if len(files) == 1 or verbose <= 0:
        maybe_tqdm = lambda x: x
    else:
        maybe_tqdm = lambda x: tqdm(x, desc='Files')

    skip_count = 0
    incompatible_count = 0

    # Open EchoView connection
    with echofilter.ev.maybe_open_echoview(
        do_open=do_open,
        minimize=minimize_echoview,
        hide=hide_echoview,
    ) as ev_app:
        for fname in maybe_tqdm(files):
            if verbose >= 2:
                print('Processing {}'.format(fname))

            # Check what the full path should be
            fname_full = echofilter.path.determine_file_path(fname, data_dir)

            # Determine where destination should be placed
            destination = echofilter.path.determine_destination(fname, fname_full, data_dir, output_dir)
            if not keep_ext:
                destination = os.path.splitext(destination)[0]

            # Check whether to skip processing this file
            if skip_existing:
                any_missing = False
                for name in ('top', 'bottom'):
                    dest_file = '{}.{}.evl'.format(destination, name)
                    if not os.path.isfile(dest_file):
                        any_missing = True
                        break
                if not any_missing:
                    if verbose >= 2:
                        print('  Skipping {}'.format(fname))
                    skip_count += 1
                    continue

            # Determine whether we need to run ev2csv on this file
            ext = os.path.splitext(fname)[1]
            if len(ext) > 0:
                ext = ext[1:].lower()
            if ext == 'csv':
                export_to_csv = False
            elif ext == 'ev':
                export_to_csv = True
            elif len(extensions) == 1 and 'csv' in extensions:
                export_to_csv = False
            elif len(extensions) == 1 and 'ev' in extensions:
                export_to_csv = True
            else:
                error_str = (
                    'Unsure how to process file {} with unrecognised extension {}'
                    .format(fname, ext)
                )
                if not skip_incompatible:
                    raise EnvironmentError(error_str)
                if verbose >= 2:
                    print('  Skipping incompatible file {}'.format(fname))
                incompatible_count += 1
                continue

            # Make a temporary directory in case we are not caching generated csvs
            # Directory and all its contents are deleted when we leave this context
            with tempfile.TemporaryDirectory() as tmpdirname:

                # Convert ev file to csv, if necessary
                ev2csv_dir = cache_csv
                if ev2csv_dir is None:
                    ev2csv_dir = tmpdirname

                if not export_to_csv:
                    csv_fname = fname_full
                else:
                    # Determine where exported CSV file should be placed
                    csv_fname = echofilter.path.determine_destination(
                        fname, fname_full, data_dir, ev2csv_dir
                    )
                    if not keep_ext:
                        csv_fname = os.path.splitext(csv_fname)[0]
                    csv_fname += csv_suffix

                if os.path.isfile(csv_fname):
                    # If CSV file is already cached, no need to re-export it
                    export_to_csv = False

                if not export_to_csv:
                    pass
                elif dry_run:
                    if verbose >= 1:
                        print('  Would export {} as CSV file {}'.format(fname_full, csv_fname))
                else:
                    # Import ev2csv now. We delay this import so Linux users
                    # without pywin32 can run on CSV files.
                    from echofilter.ev2csv import ev2csv

                    # Export the CSV file
                    fname_full = os.path.abspath(fname_full)
                    csv_fname = os.path.abspath(csv_fname)
                    ev2csv(
                        fname_full,
                        csv_fname,
                        variable_name=variable_name,
                        ev_app=ev_app,
                        verbose=verbose-1,
                    )

                if dry_run:
                    if verbose >= 1:
                        print('  Would write files to {}.SUFFIX'.format(destination))
                    continue

                # Load the data
                if verbose >= 4:
                    warn_row_overflow = np.inf
                elif verbose >= 3:
                    warn_row_overflow = None
                else:
                    warn_row_overflow = 0
                try:
                    timestamps, depths, signals = echofilter.raw.loader.transect_loader(
                        csv_fname,
                        warn_row_overflow=warn_row_overflow,
                        row_len_selector=row_len_selector,
                    )
                except KeyError:
                    if skip_incompatible and fname not in files_input:
                        if verbose >= 2:
                            print('  Skipping incompatible file {}'.format(fname))
                        incompatible_count += 1
                        continue
                    print('CSV file {} could not be loaded.'.format(fname))
                    raise

            output = inference_transect(
                model,
                timestamps,
                depths,
                signals,
                device,
                image_height,
                facing=facing,
                crop_depth_min=crop_depth_min,
                crop_depth_max=crop_depth_max,
                verbose=verbose-1,
            )

            # Convert output into lines
            top_depths = output['depths'][echofilter.utils.last_nonzero(output['p_is_above_top'] > 0.5, -1)]
            bottom_depths = output['depths'][echofilter.utils.first_nonzero(output['p_is_below_bottom'] > 0.5, -1)]

            # Export evl files
            destination_dir = os.path.dirname(destination)
            if destination_dir != '':
                os.makedirs(destination_dir, exist_ok=True)
            for name, depths in (('top', top_depths), ('bottom', bottom_depths)):
                dest_file = '{}.{}.evl'.format(destination, name)
                if verbose >= 2:
                    print('Writing output {}'.format(dest_file))
                if os.path.exists(dest_file) and not overwrite_existing:
                    raise EnvironmentError(
                        'Output {} already exists.\n'
                        ' Run with overwrite_existing=True (with the command line'
                        ' interface, use the --force flag) to overwrite existing'
                        ' outputs.'
                        .format(dest_file)
                    )
                echofilter.raw.loader.evl_writer(dest_file, timestamps, depths)

    if verbose >= 1:
        s = 'Finished {}processing {} file{}.'.format(
            'simulating ' if dry_run else '',
            len(files),
            '' if len(files) == 1 else 's',
        )
        skip_total = skip_count + incompatible_count
        if skip_total > 0:
            s += (
                ' Of these, {} file{} skipped: {} already processed'
                .format(
                    skip_total,
                    ' was' if skip_total == 1 else 's were',
                    skip_count,
                )
            )
            if not dry_run:
                s += ', {} incompatible.'.format(incompatible_count)
            s += '.'
        print(s)


def inference_transect(
    model,
    timestamps,
    depths,
    signals,
    device,
    image_height,
    facing="auto",
    crop_depth_min=None,
    crop_depth_max=None,
    dtype=torch.float,
    verbose=0,
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
    facing : {"downward", "upward", "auto"}, optional
        Orientation in which the echosounder is facing. Default is `"auto"`,
        in which case the orientation is determined from the ordering of the
        depth values in the data (increasing = `"upward"`,
        decreasing = `"downward"`).
    crop_depth_min : float or None, optional
        Minimum depth to include in input. If `None` (default), there is no
        minimum depth.
    crop_depth_max : float or None, optional
        Maxmimum depth to include in input. If `None` (default), there is no
        maximum depth.
    dtype : torch.dtype, optional
        Datatype to use for model input. Default is `torch.float`.
    verbose : int, optional
        Level of verbosity. Default is `1`.

    Returns
    -------
    dict
        Dictionary with fields as output by `echofilter.wrapper.Echofilter`,
        plus `timestamps` and `depths`.
    '''
    facing = facing.lower()
    timestamps = np.asarray(timestamps)
    depths = np.asarray(depths)
    signals = np.asarray(signals)
    transect = {
        'timestamps': timestamps,
        'depths': depths,
        'signals': signals,
    }
    if crop_depth_min is not None:
        # Apply minimum depth crop
        depth_crop_mask = transect['depths'] >= crop_depth_min
        transect['depths'] = transect['depths'][depth_crop_mask]
        transect['signals'] = transect['signals'][:, depth_crop_mask]
    if crop_depth_max is not None:
        # Apply maximum depth crop
        depth_crop_mask = transect['depths'] <= crop_depth_max
        transect['depths'] = transect['depths'][depth_crop_mask]
        transect['signals'] = transect['signals'][:, depth_crop_mask]

    # Configure data to match what the model expects to see
    # Determine whether depths are ascending or descending
    is_upward_facing = (transect['depths'][-1] < transect['depths'][0])
    # Ensure depth is always increasing (which corresponds to descending from
    # the air down the water column)
    if facing[:2] == "up" or (facing == "auto" and is_upward_facing):
        transect['depths'] = transect['depths'][::-1].copy()
        transect['signals'] = transect['signals'][:, ::-1].copy()
        if facing == "auto" and verbose >= 1:
            print(
                'Data was autodetected as upward facing, and was flipped'
                ' vertically before being input into the model.'
            )
    elif facing[:4] != "down" and facing[:4] != "auto":
        raise ValueError('facing should be one of "downward", "upward", and "auto"')
    elif facing[:4] == "down" and is_upward_facing:
        print('Warning: facing = "{}" was provided, but data appears to be upward facing'.format(facing))

    # To reduce memory consumption, split into segments whenever the recording
    # interval is longer than normal
    segments = split_transect(**transect)
    if verbose >= 1:
        segments = tqdm(list(segments), desc='Segments')
    outputs = []
    for segment in segments:
        # Preprocessing transform
        transform = torchvision.transforms.Compose([
            echofilter.transforms.Normalize(DATA_MEAN, DATA_STDEV),
            echofilter.transforms.ReplaceNan(-3),
            echofilter.transforms.Rescale(
                (segment['signals'].shape[0], image_height),
                order=1,
            ),
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

    if verbose >= 1:
        print()

    return join_transect(outputs)


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
        application-specific default cache directory is used.
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

    sources = CHECKPOINT_RESOURCES[checkpoint_name]
    success = False
    for key, url_or_id in sources.items():
        if key == 'gdrive':
            if verbose > 0:
                print('Downloading checkpoint {} from GDrive...'.format(checkpoint_name))
            try:
                download_file_from_google_drive(url_or_id, cache_dir, filename=checkpoint_name)
                success = True
                continue
            except (pickle.UnpicklingError, urllib.error.URLError):
                if verbose > 0:
                    print('\nCould not download checkpoint {} from GDrive!'.format(checkpoint_name))
        else:
            if verbose > 0:
                print('Downloading checkpoint {} from {}...'.format(checkpoint_name, url_or_id))
            try:
                download_url(url_or_id, cache_dir, filename=checkpoint_name)
                success = True
                continue
            except (pickle.UnpicklingError, urllib.error.URLError):
                if verbose > 0:
                    print('\nCould not download checkpoint {} from {}'.format(checkpoint_name, url_or_id))

    if not success:
        raise OSError('Unable to download {} from {}'.format(checkpoint_name, sources))

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
        '--data-dir', '-d',
        dest='data_dir',
        type=str,
        default='.',
        metavar='DIR',
        help='path to directory containing FILE (default: ".")',
    )
    default_extensions = ['csv']
    if echofilter.path.check_if_windows():
        default_extensions.append('ev')
    parser.add_argument(
        '--extension', '-x',
        dest='extensions',
        type=str,
        nargs='+',
        default=default_extensions,
        help='file extensions to process (default: {})'.format(default_extensions),
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='',
        metavar='DIR',
        help=
            'path to output directory. If empty, output is placed in the same'
            ' directory as the input file. (default: "")',
    )
    parser.add_argument(
        '--keep-ext',
        action='store_true',
        help='keep the input file extension in output file names',
    )
    parser.add_argument(
        "--variable-name",
        "--vn",
        dest="variable_name",
        type=str,
        default=DEFAULT_VARNAME,
        help="Name of the EchoView acoustic variable to load from EV files."
        " Default is {}.".format(
            DEFAULT_VARNAME
        ),
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
        '--cache-csv',
        nargs='?',
        type=str,
        default=None,
        const='',
        metavar='DIR',
        help=
            'path to directory where CSV files generated from EV inputs should'
            ' be cached. If an empty string is given, exported CSV files will'
            ' be placed in the same directory as the input EV file.'
            ' (default: do not cache)'
    )
    parser.add_argument(
        '--csv-suffix',
        type=str,
        default='.csv',
        help=
            'suffix used for cached CSV files which are exported from EV files'
            ' (default: ".csv")'
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
        '--facing',
        type=str,
        choices=['downward', 'upward', 'auto'],
        default='auto',
        help='orientation of echosounder (default: "auto")',
    )
    parser.add_argument(
        '--row-len-selector',
        type=str,
        choices=['init', 'min', 'max', 'median', 'mode'],
        default='mode',
        help='how to handle inputs with differing number of samples across time (default: "mode")',
    )
    parser.add_argument(
        '--crop-depth-min',
        type=float,
        default=None,
        help=
            'shallowest depth, in metres, to analyse after. Data will be'
            ' truncated at this depth with shallower data removed.'
            ' (default: None, do not truncate)',
    )
    parser.add_argument(
        '--crop-depth-max',
        type=float,
        default=None,
        help=
            'deepest depth, in metres, to analyse after. Data will be'
            ' truncated at this depth with deeper data removed.'
            ' (default: None, do not truncate)',
    )
    parser.add_argument(
        '--force', '-f',
        dest='overwrite_existing',
        action='store_true',
        help='overwrite existing files without warning',
    )
    parser.add_argument(
        '--skip-existing', '--skip',
        dest='skip_existing',
        action='store_true',
        help='skip processing files for which all outputs already exist',
    )
    parser.add_argument(
        '--skip-incompatible',
        dest='skip_incompatible',
        action='store_true',
        help='skip incompatible files without raising an error',
    )
    parser.add_argument(
        "--minimize-echoview",
        dest="minimize_echoview",
        action="store_true",
        help="minimize the Echoview window while this code runs",
    )
    parser.add_argument(
        "--show-echoview",
        dest="hide_echoview",
        action="store_const",
        const="never",
        default=None,
        help="don't hide an Echoview window created to run this code",
    )
    parser.add_argument(
        "--hide-echoview",
        dest="hide_echoview",
        action="store_const",
        const="new",
        help="hide Echoview window, but only if it was not already open (default behaviour)",
    )
    parser.add_argument(
        "--always-hide-echoview",
        "--always-hide",
        dest="hide_echoview",
        action="store_const",
        const="always",
        help="hide the Echoview window while this code runs, even if it was already open",
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='perform a trial run with no changes made',
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

    if kwargs["hide_echoview"] is None:
        kwargs["hide_echoview"] = "never" if kwargs["minimize_echoview"] else "new"

    run_inference(**kwargs)


if __name__ == '__main__':
    main()
