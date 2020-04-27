#!/usr/bin/env python

from collections import OrderedDict
import os
import shutil
import datetime
import time

import pandas as pd
import numpy as np
import torch
import torch.nn
import torch.utils.data
import torchvision.transforms
from torchutils.utils import count_parameters

import echofilter.raw.loader
import echofilter.transforms
import echofilter.utils
from echofilter.unet import UNet
from echofilter.wrapper import Echofilter


DATA_MEAN = -80.
DATA_STDEV = 20.


def main(
        files=[],
        checkpoint_path='model_best.pth.tar',
        data_dir='/data/dsforce/surveyExports',
        output_dir='processed',
        dataset_name='mobile',
        sample_shape=(128, 512),
        crop_depth=70,
        n_steps=4,
        latent_channels=64,
        expansion_factor=2,
        device='cuda',
        n_worker=4,
        batch_size=64,
        print_freq=10,
    ):

    dtype = torch.float

    # Preprocessing transforms
    transform = torchvision.transforms.Compose([
        echofilter.transforms.Normalize(DATA_MEAN, DATA_STDEV),
        echofilter.transforms.ReplaceNan(-3),
    ])

    print()
    print(
        'Constructing U-Net model with '
        '{} steps, '
        'initial latent channels {}, '
        'expansion_factor {}'
        .format(n_steps, latent_channels, expansion_factor)
    )
    model = Echofilter(
        UNet(1, 5, n_steps=n_steps, latent_channels=latent_channels, expansion_factor=expansion_factor),
        top='boundary',
        bottom='boundary',
    )

    if not os.path.isfile(checkpoint_path):
        raise EnvironmentError("No checkpoint found at '{}'".format(checkpoint_path))
    print("Loading checkpoint '{}'".format(checkpoint_path))
    if device is None:
        checkpoint = torch.load(checkpoint_path)
    else:
        # Map model to be loaded to specified single gpu.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    best_loss = checkpoint['best_loss']
    model.load_state_dict(checkpoint['state_dict'])
    print(
        "Loaded checkpoint '{}' (epoch {})"
        .format(checkpoint_path, checkpoint['epoch'])
    )
    # Ensure model is on correct device
    model.to(device)
    print(
        'Built model with {} trainable parameters'
        .format(count_parameters(model, only_trainable=True))
    )
    # Put model in evaluation mode
    model.eval()

    for fname in files:
        # Check what the full path should be
        if os.path.isfile(fname):
            fname_full = fname
        elif os.path.isfile(os.path.join(data_dir, fname)):
            fname_full = os.path.join(data_dir, fname)
        elif os.path.isfile(os.path.join(data_dir, dataset_name, fname)):
            fname_full = os.path.join(data_dir, dataset_name, fname)
        else:
            raise EnvironmentError('Could not locate file {}'.format(fname))
        # Load the data
        timestamps, depths, signals = echofilter.raw.loader.transect_loader(fname_full)
        data = {
            'timestamps': timestamps,
            'depths': depths,
            'signals': signals,
        }
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
        data = echofilter.transforms.Rescale((signals.shape[0], sample_shape[1]))(data)
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
        os.makedirs(os.path.dirname(os.path.join(output_dir, fname)), exist_ok=True)
        print(os.path.join(output_dir, fname + '.top.evl'))
        print(os.path.join(output_dir, fname + '.bottom.evl'))
        echofilter.raw.loader.evl_writer(os.path.join(output_dir, fname + '.top.evl'), timestamps, top_depths)
        echofilter.raw.loader.evl_writer(os.path.join(output_dir, fname + '.bottom.evl'), timestamps, bottom_depths)


if __name__ == '__main__':

    import argparse

    # Data parameters
    parser = argparse.ArgumentParser(description='Echofilter')
    parser.add_argument(
        'files',
        type=str,
        nargs='+',
        default=[],
        metavar='FILE',
        help='file(s) to process',
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/data/dsforce/surveyExports',
        metavar='DIR',
        help='path to root data directory',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='processed',
        metavar='DIR',
        help='path to output directory',
    )
    parser.add_argument(
        '--checkpoint',
        dest='checkpoint_path',
        type=str,
        default='model_best.pth.tar',
        metavar='PATH',
        help='path to checkpoint to load (default: "model_best.pth.tar")',
    )
    parser.add_argument(
        '--crop-depth',
        type=float,
        default=70,
        help='depth, in metres, at which data should be truncated (default: 70)',
    )

    # Model parameters
    parser.add_argument(
        '--n-steps',
        type=int,
        default=4,
        help='number of steps down and up in the UNet (default: 4)',
    )
    parser.add_argument(
        '--latent-channels',
        type=int,
        default=64,
        help='number of initial/final latent channels to use in the model (default: 64)',
    )
    parser.add_argument(
        '--expansion-factor',
        type=float,
        default=2.0,
        help='expansion for number of channels as model becomes deeper (default: 2.0)',
    )

    # Training methodology parameters
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='device to use (default: "cuda", using first gpu)',
    )
    parser.add_argument(
        '-j', '--workers',
        dest='n_worker',
        type=int,
        default=4,
        metavar='N',
        help='number of data loading workers (default: 4)',
    )
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=64,
        help='mini-batch size (default: 64)',
    )

    main(**vars(parser.parse_args()))
