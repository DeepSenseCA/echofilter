#!/usr/bin/env python

"""
Provides a command line interface for the training routine.

This is separated out from train.py so the documentation can be accessed without
having all the training dependencies installed.
"""

# This file is part of Echofilter.
#
# Copyright (C) 2020-2022  Scott C. Lowe and Offshore Energy Research Association (OERA)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import os
import sys

from .. import __meta__
from . import formatters


def get_parser():
    """
    Build parser for training command line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        CLI argument parser for training.
    """

    import argparse

    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        prog = os.path.split(__file__)[1]
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Echofilter model training",
        add_help=False,
    )

    # Actions
    group_action = parser.add_argument_group(
        "Actions",
        "These arguments specify special actions to perform. The main action"
        " of this program is supressed if any of these are given.",
    )
    group_action.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit.",
    )
    group_action.add_argument(
        "--version",
        "-V",
        action="version",
        version="%(prog)s {version}".format(version=__meta__.version),
        help="Show program's version number and exit.",
    )

    # Data parameters
    group_data = parser.add_argument_group("Data parameters")
    group_data.add_argument(
        "--data-dir",
        type=str,
        default="/data/dsforce/surveyExports",
        metavar="DIR",
        help="path to root data directory",
    )
    group_data.add_argument(
        "--dataset",
        dest="dataset_name",
        type=str,
        default="mobile",
        help="which dataset to use",
    )
    group_data.add_argument(
        "--train-partition",
        type=str,
        default=None,
        help="which partition to train on (default depends on dataset)",
    )
    group_data.add_argument(
        "--val-partition",
        type=str,
        default=None,
        help="which partition to validate on (default depends on dataset)",
    )
    group_data.add_argument(
        "--shape",
        dest="sample_shape",
        nargs=2,
        type=int,
        default=(128, 512),
        help="input shape [W, H] (default: %(default)s)",
    )
    group_data.add_argument(
        "--crop-depth",
        type=float,
        default=None,
        help="depth, in metres, at which data should be truncated (default: %(default)s)",
    )
    group_data.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help='path to latest checkpoint (default: "%(default)s")',
    )
    group_data.add_argument(
        "--cold-restart",
        dest="restart",
        action="store_const",
        const="cold",
        default="",
        help="when resuming from a checkpoint, use this only for initial weights",
    )
    group_data.add_argument(
        "--warm-restart",
        dest="restart",
        action="store_const",
        const="warm",
        help="""
            when resuming from a checkpoint, use the existing weights and
            optimizer state but start a new LR schedule
        """,
    )
    group_data.add_argument(
        "--log",
        dest="log_name",
        default=None,
        type=str,
        help="output directory name (default: DATE_TIME)",
    )
    group_data.add_argument(
        "--log-append",
        dest="log_name_append",
        default=None,
        type=str,
        help="string to append to output directory name (default: HOSTNAME)",
    )

    # Model parameters
    group_model = parser.add_argument_group("Model parameters")
    group_model.add_argument(
        "--conditional",
        action="store_true",
        help=(
            "train a model conditioned on the direction the sounder is facing"
            " (in addition to an unconditional model)"
        ),
    )
    group_model.add_argument(
        "--nblock",
        "--num-blocks",
        dest="n_block",
        type=int,
        default=6,
        help="number of blocks down and up in the UNet (default: %(default)s)",
    )
    group_model.add_argument(
        "--latent-channels",
        type=int,
        default=32,
        help="number of initial/final latent channels to use in the model (default: %(default)s)",
    )
    group_model.add_argument(
        "--expansion-factor",
        type=float,
        default=1.0,
        help="expansion for number of channels as model becomes deeper"
        " (default: %(default)s, constant number of channels)",
    )
    group_model.add_argument(
        "--expand-only-on-down",
        action="store_true",
        help="only expand channels on dowsampling blocks",
    )
    group_model.add_argument(
        "--blocks-per-downsample",
        nargs="+",
        type=int,
        default=(2, 1),
        help="for each dim (time, depth), number of blocks between downsample"
        " steps (default: %(default)s)",
    )
    group_model.add_argument(
        "--blocks-before-first-downsample",
        nargs="+",
        type=int,
        default=(2, 1),
        help="for each dim (time, depth), number of blocks before first"
        " downsample step (default: %(default)s)",
    )
    group_model.add_argument(
        "--only-skip-connection-on-downsample",
        dest="always_include_skip_connection",
        action="store_false",
        help="only include skip connections when downsampling",
    )
    group_model.add_argument(
        "--deepest-inner",
        type=str,
        default="horizontal_block",
        help="layer to include at the deepest point of the UNet"
        ' (default: "horizontal_block"). Set to "identity" to disable.',
    )
    group_model.add_argument(
        "--intrablock-expansion",
        type=float,
        default=6.0,
        help="expansion within inverse residual blocks (default: %(default)s)",
    )
    group_model.add_argument(
        "--se-reduction",
        "--se",
        dest="se_reduction",
        type=float,
        default=4.0,
        help="reduction within squeeze-and-excite blocks (default: %(default)s)",
    )
    group_model.add_argument(
        "--downsampling-modes",
        nargs="+",
        type=str,
        default="max",
        help='for each downsampling step, the method to use (default: "%(default)s")',
    )
    group_model.add_argument(
        "--upsampling-modes",
        nargs="+",
        type=str,
        default="bilinear",
        help='for each upsampling step, the method to use (default: "%(default)s")',
    )
    group_model.add_argument(
        "--fused-conv",
        dest="depthwise_separable_conv",
        action="store_false",
        help="use fused instead of depthwise separable convolutions",
    )
    group_model.add_argument(
        "--no-residual",
        dest="residual",
        action="store_false",
        help="don't use residual blocks",
    )
    group_model.add_argument(
        "--actfn",
        type=str,
        default="InplaceReLU",
        help="activation function to use",
    )
    group_model.add_argument(
        "--kernel",
        dest="kernel_size",
        type=int,
        default=5,
        help="convolution kernel size (default: %(default)s)",
    )

    # Training methodology parameters
    group_training = parser.add_argument_group("Training parameters")
    group_training.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='device to use (default: "%(default)s", using first gpu)',
    )
    group_training.add_argument(
        "--multigpu",
        action="store_true",
        help="train on multiple GPUs",
    )
    group_training.add_argument(
        "--no-amp",
        dest="use_mixed_precision",
        action="store_false",
        default=None,
        help="use fp32 instead of mixed precision (default: use mixed precision on gpu)",
    )
    group_training.add_argument(
        "--amp-opt",
        type=str,
        default="O1",
        help='optimizer level for apex automatic mixed precision (default: "%(default)s")',
    )
    group_training.add_argument(
        "-j",
        "--workers",
        dest="n_worker",
        type=int,
        default=8,
        metavar="N",
        help="number of data loading workers (default: %(default)s)",
    )
    group_training.add_argument(
        "-p",
        "--print-freq",
        type=int,
        default=50,
        help="print frequency (default: %(default)s)",
    )
    group_training.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=16,
        help="mini-batch size (default: %(default)s)",
    )
    group_training.add_argument(
        "--no-stratify",
        dest="stratify",
        action="store_false",
        help="disable stratified sampling; use fully random sampling instead",
    )
    group_training.add_argument(
        "--epochs",
        dest="n_epoch",
        type=int,
        default=20,
        help="number of total epochs to run (default: %(default)s)",
    )
    group_training.add_argument(
        "--seed",
        default=None,
        type=int,
        help="seed for initializing training.",
    )

    # Optimiser parameters
    group_optim = parser.add_argument_group("Optimizer parameters")
    group_optim.add_argument(
        "--optim",
        "--optimiser",
        "--optimizer",
        dest="optimizer",
        type=str,
        default="rangerva",
        help='optimizer name (default: "%(default)s")',
    )
    group_optim.add_argument(
        "--schedule",
        type=str,
        default="constant",
        help='LR schedule (default: "%(default)s")',
    )
    group_optim.add_argument(
        "--lr",
        "--learning-rate",
        dest="lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="initial learning rate (default: %(default)s)",
    )
    group_optim.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum (default: %(default)s)",
    )
    group_optim.add_argument(
        "--base-momentum",
        type=float,
        default=None,
        help="base momentum; only used for OneCycle schedule (default: same as momentum)",
    )
    group_optim.add_argument(
        "--wd",
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=1e-5,
        help="weight decay (default: %(default)s)",
    )
    group_optim.add_argument(
        "--warmup-pct",
        type=float,
        default=0.2,
        help="fraction of training to spend warming up LR; only used for"
        " OneCycle MesaOneCycle schedules (default: %(default)s)",
    )
    group_optim.add_argument(
        "--warmdown-pct",
        type=float,
        default=0.7,
        help="fraction of training before warming down LR; only used for"
        " MesaOneCycle schedule (default: %(default)s)",
    )
    group_optim.add_argument(
        "--anneal-strategy",
        type=str,
        default="cos",
        help='annealing strategy; only used for OneCycle schedule (default: "%(default)s")',
    )
    group_optim.add_argument(
        "--overall-loss-weight",
        type=float,
        default=0.0,
        help="weighting for overall loss term (default: %(default)s)",
    )

    return parser


def _get_parser_sphinx():
    """
    Pre-format parser help for sphinx-argparse processing.
    """
    return formatters.format_parser_for_sphinx(get_parser())


def main():
    """
    Run command line interface for model training.
    """
    parser = get_parser()

    kwargs = vars(parser.parse_args())

    for k in ["blocks_per_downsample", "blocks_before_first_downsample"]:
        if len(kwargs[k]) == 1:
            kwargs[k] = kwargs[k][0]

    print("CLI arguments:")
    print(kwargs)
    print()

    # Use seaborn to set matplotlib plotting defaults
    import seaborn as sns

    sns.set()

    from ..train import train

    train(**kwargs)


if __name__ == "__main__":
    main()
