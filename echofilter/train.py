#!/usr/bin/env python

"""
Model training routine.
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

import copy
import datetime
import os
import pprint
import shutil
import sys
import time
import traceback
from collections import OrderedDict

try:
    import apex
except ImportError:
    apex = None

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.optim
import torch.utils.data
import torchvision.transforms

import echofilter.data
import echofilter.optim.utils
import echofilter.raw.shardloader
from echofilter.nn.unet import UNet
from echofilter.nn.utils import count_parameters, seed_all
from echofilter.nn.wrapper import Echofilter, EchofilterLoss
from echofilter.optim import criterions, schedulers
from echofilter.optim.meters import AverageMeter, ProgressMeter
from echofilter.plotting import plot_transect_predictions
from echofilter.raw.loader import get_partition_list
from echofilter.raw.manipulate import load_decomposed_transect_mask
from echofilter.ui.train_cli import main

# --- For mobile dataset,
# DATA_CENTER = -81.5
# DATA_DEVIATION = 21.9
# CENTER_METHOD = "mean"
# DEVIATION_METHOD = "stdev"

# --- For stationary dataset,
# DATA_CENTER = -78.7
# DATA_DEVIATION = 19.2
# CENTER_METHOD = "mean"
# DEVIATION_METHOD = "stdev"

# --- For intermediate values between both datasets
# DATA_CENTER = -80.
# DATA_DEVIATION = 20.
# CENTER_METHOD = "mean"
# DEVIATION_METHOD = "stdev"
# NAN_VALUE = -3

# --- Overall values to use
# DATA_CENTER = -97.5
# DATA_DEVIATION = 16.5
# CENTER_METHOD = "pc10"
# DEVIATION_METHOD = "idr"
# NAN_VALUE = -1

# Normalise each sample independently, based on its own intensity distribution
CENTER_METHOD = "median"
DEVIATION_METHOD = "idr"
DATA_CENTER = CENTER_METHOD
DATA_DEVIATION = DEVIATION_METHOD
NAN_VALUE = -3

# Transects to plot for debugging
PLOT_TRANSECTS = {
    "mobile": [
        "mobile/Survey07/Survey07_GR4_N5W_survey7",
        "mobile/Survey14/Survey14_GR4_N0W_E",
        "mobile/Survey16/Survey16_GR4_N5W_E",
        "mobile/Survey17/Survey17_GR4_N5W_E",
    ],
    "MinasPassage": [
        "MinasPassage/december2017/december2017_D20180213-T115216_D20180213-T172216",
        "MinasPassage/march2018/march2018_D20180513-T195216_D20180514-T012216",
        "MinasPassage/september2018/september2018_D20181027-T202217_D20181028-T015217",
        "MinasPassage/september2018/september2018_D20181107-T122220_D20181107-T175217",
    ],
    "GrandPassage": [
        "GrandPassage/phase2/GrandPassage_WBAT_2B_20200125_UTC160020_ebblow",
        "GrandPassage/phase2/GrandPassage_WBAT_2B_20200202_UTC040019_floodhigh",
    ],
}

DEFAULT_CROP_DEPTH_PLOTS = 100
MAX_INPUT_LEN = 3500


def train(
    data_dir="/data/dsforce/surveyExports",
    dataset_name="mobile",
    train_partition=None,
    val_partition=None,
    sample_shape=(128, 512),
    crop_depth=None,
    resume="",
    restart="",
    log_name=None,
    log_name_append=None,
    conditional=False,
    n_block=6,
    latent_channels=32,
    expansion_factor=1,
    expand_only_on_down=False,
    blocks_per_downsample=(2, 1),
    blocks_before_first_downsample=(2, 1),
    always_include_skip_connection=True,
    deepest_inner="horizontal_block",
    intrablock_expansion=6,
    se_reduction=4,
    downsampling_modes="max",
    upsampling_modes="bilinear",
    depthwise_separable_conv=True,
    residual=True,
    actfn="InplaceReLU",
    kernel_size=5,
    use_mixed_precision=None,
    amp_opt="O1",
    device="cuda",
    multigpu=False,
    n_worker=8,
    batch_size=16,
    stratify=True,
    n_epoch=20,
    seed=None,
    print_freq=50,
    optimizer="adam",
    schedule="constant",
    lr=0.1,
    momentum=0.9,
    base_momentum=None,
    weight_decay=1e-5,
    warmup_pct=0.2,
    warmdown_pct=0.7,
    anneal_strategy="cos",
    overall_loss_weight=0.0,
):
    """
    Train a model.
    """
    if restart and not resume:
        raise ValueError(
            "A checkpoint must be provided to restart from when doing a cold restart"
        )

    # Lazy import of tensorboard, so training-only requirements are not needed
    # for automated documentation building.
    from torch.utils.tensorboard import SummaryWriter

    seed_all(seed)
    # Can't get this to be deterministic anyway, so may as well keep the
    # non-deterministic optimization enabled
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Input handling
    schedule = schedule.lower()

    if base_momentum is None:
        base_momentum = momentum

    if log_name is None or log_name == "":
        log_name = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    if log_name_append is None:
        log_name_append = os.uname()[1]
    if len(log_name_append) > 0:
        log_name += "_" + log_name_append

    print("Output will be written to {}/{}".format(dataset_name, log_name))

    if use_mixed_precision is None:
        use_mixed_precision = "cpu" not in device
    if use_mixed_precision and apex is None:
        print("NVIDIA apex must be installed to use mixed precision.")
        use_mixed_precision = False

    # Need to set the default device for apex.amp
    if device is not None and device != "cpu":
        torch.cuda.set_device(torch.device(device))

    # Build dataset
    dataset_train, dataset_val, dataset_augval = build_dataset(
        dataset_name,
        data_dir,
        sample_shape,
        train_partition=train_partition,
        val_partition=val_partition,
    )

    print("Train dataset has {:4d} samples".format(len(dataset_train)))
    print("Val   dataset has {:4d} samples".format(len(dataset_val)))

    stratify_allowed = stratify
    if stratify_allowed:
        stratify = isinstance(dataset_train, echofilter.data.dataset.ConcatDataset)
    if stratify:
        # Use custom stratified sampler to handle multiple datasets
        sampler = echofilter.data.dataset.StratifiedRandomSampler(dataset_train)
    else:
        sampler = None

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=not stratify,
        sampler=sampler,
        num_workers=n_worker,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=echofilter.data.utils.worker_seed_fn,
    )
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_worker,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=echofilter.data.utils.worker_staticseed_fn,
    )
    loader_augval = torch.utils.data.DataLoader(
        dataset_augval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_worker,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=echofilter.data.utils.worker_staticseed_fn,
    )
    print("Train loader has {:3d} batches".format(len(loader_train)))
    print("Val   loader has {:3d} batches".format(len(loader_val)))

    print()
    print(
        "Constructing U-Net model with "
        "{} blocks, "
        "initial latent channels {}, "
        "expansion_factor {}".format(n_block, latent_channels, expansion_factor)
    )
    model_parameters = dict(
        in_channels=1,
        out_channels=10,
        initial_channels=latent_channels,
        bottleneck_channels=latent_channels,
        n_block=n_block,
        unet_expansion_factor=expansion_factor,
        expand_only_on_down=expand_only_on_down,
        blocks_per_downsample=blocks_per_downsample,
        blocks_before_first_downsample=blocks_before_first_downsample,
        always_include_skip_connection=always_include_skip_connection,
        deepest_inner=deepest_inner,
        intrablock_expansion=intrablock_expansion,
        se_reduction=se_reduction,
        downsampling_modes=downsampling_modes,
        upsampling_modes=upsampling_modes,
        depthwise_separable_conv=depthwise_separable_conv,
        residual=residual,
        actfn=actfn,
        kernel_size=kernel_size,
    )
    if conditional:
        model_parameters["out_channels"] *= 3
    print()
    pprint.pprint(model_parameters)
    print()

    unet = UNet(**model_parameters)
    unet.to(device)
    print(
        "Built UNet model with {} trainable parameters".format(
            count_parameters(unet, only_trainable=True)
        )
    )

    # define loss function (criterion) and optimizer
    criterion = EchofilterLoss(conditional=conditional, overall=overall_loss_weight)

    optimizer_name = optimizer.lower()
    if optimizer_name == "adam":
        optimizer_class = torch.optim.Adam
    elif optimizer_name == "adamw":
        optimizer_class = torch.optim.AdamW
    elif optimizer_name == "ranger":
        import ranger

        optimizer_class = ranger.Ranger
    elif optimizer_name == "rangerva":
        import ranger

        optimizer_class = ranger.RangerVA
    elif optimizer_name == "rangerqh":
        import ranger

        optimizer_class = ranger.RangerQH
    else:
        # We don't support arbitrary optimizers from torch.optim because they
        # need different configuration parameters to Adam.
        raise ValueError("Unrecognised optimizer: {}".format(optimizer))

    optimizer = optimizer_class(
        unet.parameters(),
        lr,
        betas=(momentum, 0.999),
        weight_decay=weight_decay,
    )

    schedule_data = {"name": schedule}
    lr_initial_div_factor = 1e3
    lr_final_div_factor = 1e5
    if schedule == "lrfinder":
        pass
    elif schedule == "constant":
        pass
    elif schedule == "onecycle":
        schedule_data["scheduler"] = schedulers.OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=len(loader_train),
            epochs=n_epoch,
            pct_start=warmup_pct,
            anneal_strategy=anneal_strategy,
            cycle_momentum=True,
            base_momentum=base_momentum,
            max_momentum=momentum,
            div_factor=lr_initial_div_factor,
            final_div_factor=lr_final_div_factor,
        )
    elif schedule == "mesaonecycle":
        schedule_data["scheduler"] = schedulers.MesaOneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=len(loader_train),
            epochs=n_epoch,
            pct_start=warmup_pct,
            pct_end=warmdown_pct,
            anneal_strategy=anneal_strategy,
            cycle_momentum=True,
            base_momentum=base_momentum,
            max_momentum=momentum,
            div_factor=lr_initial_div_factor,
            final_div_factor=lr_final_div_factor,
        )
    else:
        raise ValueError("Unsupported schedule: {}".format(schedule))

    if use_mixed_precision:
        print('Converting unet to mixed precision, opt="{}"'.format(amp_opt))
        unet, optimizer = apex.amp.initialize(unet, optimizer, opt_level=amp_opt)

    model_inner = unet
    if multigpu and torch.cuda.device_count() > 1:
        print("Using local parallelism on {} GPUs".format(torch.cuda.device_count()))
        model_inner = torch.nn.DataParallel(model_inner)

    # Add UI wrapper around model
    model = Echofilter(
        model_inner,
        top="boundary",
        bottom="boundary",
        conditional=conditional,
    )

    if schedule == "lrfinder":
        from torch_lr_finder import LRFinder

        print("Running learning rate finder")
        lr_finder = LRFinder(model, optimizer, criterion, device=device)
        lr_finder.range_test(loader_train, end_lr=100, num_iter=100, diverge_th=3)
        print("Plotting learning rate finder results")
        hf = plt.figure(figsize=(15, 9))
        ax = plt.axes()
        lr_finder.plot(skip_start=0, skip_end=1, log_lr=True, ax=ax)
        plt.tick_params(reset=True, color=(0.2, 0.2, 0.2))
        plt.tick_params(labelsize=14)
        ax.minorticks_on()
        ax.tick_params(direction="out")
        # Save figure
        figpth = os.path.join("models", dataset_name, log_name, "lrfinder.png")
        os.makedirs(os.path.dirname(figpth), exist_ok=True)
        plt.savefig(figpth)
        print("LR Finder results saved to {}".format(figpth))
        return

    # Initialise loop tracking
    start_epoch = 1
    best_loss_val = float("inf")

    # optionally resume from a checkpoint
    if resume:
        if not os.path.isfile(resume):
            raise EnvironmentError("No checkpoint found at '{}'".format(resume))
        print("Loading checkpoint '{}'".format(resume))
        if device is None:
            checkpoint = torch.load(resume)
        else:
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(resume, map_location=device)
        try:
            unet.load_state_dict(checkpoint["state_dict"])
        except RuntimeError:
            print(
                "Checkpoint doesn't seem to be for the UNet component. Trying"
                " to load it as the whole model instead."
            )
            model.load_state_dict(checkpoint["state_dict"])
        if restart == "cold":
            print("Loaded checkpoint '{}' for cold restart".format(resume))
        else:
            optimizer.load_state_dict(checkpoint["optimizer"])
            if use_mixed_precision and "amp" in checkpoint:
                apex.amp.load_state_dict(checkpoint["amp"])
        if restart == "warm":
            # We loaded the optimizer state from the checkpoint to get the
            # current buffer values (the weighted history of updates).
            # However, this also changed our schedule settings to the ones
            # used to train the model in the checkpoint, so we need to
            # overwrite it with the schedule initialisation step now.
            for group in optimizer.param_groups:
                if "onecycle" in schedule:
                    group["lr"] = group["initial_lr"] = lr / lr_initial_div_factor
                    group["max_lr"] = lr
                    group["min_lr"] = group["initial_lr"] / lr_final_div_factor
                    group["base_momentum"] = base_momentum
                    group["max_momentum"] = momentum
                else:
                    group["lr"] = lr
                    if "betas" in group:
                        _, beta2 = group["betas"]
                        group["betas"] = (momentum, beta2)
                    else:
                        group["momentum"] = momentum
            print("Loaded checkpoint '{}' for warm restart".format(resume))
        elif not restart:
            start_epoch = checkpoint["epoch"] + 1
            best_loss_val = checkpoint["best_loss"]
            if "scheduler" in schedule_data:
                step_num = checkpoint["epoch"] * len(loader_train)
                schedule_data["scheduler"].last_epoch = step_num
            print(
                "Loaded checkpoint '{}' (epoch {})".format(resume, checkpoint["epoch"])
            )

    # Make a tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join("runs", dataset_name, log_name))

    print("Starting training")
    t_start = time.time()
    for epoch in range(start_epoch, n_epoch + 1):

        t_epoch_start = time.time()

        # Set the seed state at the start of each epoch
        # This ensures some level of consistency between models which continue
        # running and those resumed from checkpoints (don't want to run the
        # exact initial augmented samples from the first epoch again when
        # resuming later in the run)
        seed_all(1000 + seed + epoch)
        # Can't get this to be deterministic anyway, so may as well keep the
        # non-deterministic optimization enabled
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        # Resample offsets for each window
        loader_train.dataset.initialise_datapoints()

        # train for one epoch
        (
            loss_tr,
            meters_tr,
            (ex_input_tr, ex_data_tr, ex_output_tr),
            (batch_time, data_time),
        ) = train_epoch(
            loader_train,
            model,
            criterion,
            optimizer,
            device,
            epoch,
            print_freq=print_freq,
            schedule_data=schedule_data,
            use_mixed_precision=use_mixed_precision,
        )

        t_val_start = time.time()

        # evaluate on validation set
        loss_val, meters_val, (ex_input_val, ex_data_val, ex_output_val) = validate(
            loader_val,
            model,
            criterion,
            device,
            print_freq=print_freq,
            prefix="Validation",
        )
        # evaluate on augmented validation set
        (
            loss_augval,
            meters_augval,
            (ex_input_augval, ex_data_augval, ex_output_augval),
        ) = validate(
            loader_augval,
            model,
            criterion,
            device,
            print_freq=print_freq,
            prefix="Aug-Val   ",
        )
        t_val_end = time.time()

        print(
            "{}/{}\nCompleted {} of {} epochs in {}".format(
                dataset_name,
                log_name,
                epoch,
                n_epoch,
                datetime.timedelta(seconds=time.time() - t_start),
            )
        )
        # Print metrics to terminal
        name_fmt = "{:.<28s}"
        current_lr = echofilter.optim.utils.get_current_lr(optimizer)
        current_mom = echofilter.optim.utils.get_current_momentum(optimizer)

        print((name_fmt + " {:.4e}").format("Learning rate", current_lr))
        print((name_fmt + " {:.4f}").format("Momentum", current_mom))
        print(
            (name_fmt + " Train: {:.4e}  AugVal: {:.4e}  Val: {:.4e}").format(
                "Loss", loss_tr, loss_augval, loss_val
            )
        )
        for chn in meters_tr:
            if chn.lower() != "overall":
                continue
            # For each output plane
            print(chn)
            for cr in meters_tr[chn]:
                # For each criterion
                fmt_str = name_fmt
                fmt_str += " Train: {" + meters_tr[chn][cr].fmt + "}"
                fmt_str += "    AugVal: {" + meters_augval[chn][cr].fmt + "}"
                fmt_str += "    Val: {" + meters_val[chn][cr].fmt + "}"
                print(
                    fmt_str.format(
                        meters_tr[chn][cr].name,
                        meters_tr[chn][cr].avg,
                        meters_augval[chn][cr].avg,
                        meters_val[chn][cr].avg,
                    )
                )

        # Add hyper parameters to tensorboard
        writer.add_scalar("learning_rate", current_lr, epoch)
        writer.add_scalar("momentum", current_mom, epoch)
        writer.add_scalar(
            "parameter_count", count_parameters(model, only_trainable=True), epoch
        )

        # Add metrics to tensorboard
        for loss_p, partition in (
            (loss_tr, "Train"),
            (loss_val, "Val"),
            (loss_augval, "ValAug"),
        ):
            writer.add_scalar("{}/{}".format("Loss", partition), loss_p, epoch)
        for chn in meters_tr:
            # For each output plane
            for cr in meters_tr[chn]:
                # For each criterion
                writer.add_scalar(
                    "{}/{}/{}".format(cr, chn, "Train"), meters_tr[chn][cr].avg, epoch
                )
                writer.add_scalar(
                    "{}/{}/{}".format(cr, chn, "ValAug"),
                    meters_augval[chn][cr].avg,
                    epoch,
                )
                writer.add_scalar(
                    "{}/{}/{}".format(cr, chn, "Val"), meters_val[chn][cr].avg, epoch
                )

        # Determine whether to generate sample transect plots, or skip them
        if n_epoch < 20:
            # Every epoch
            generate_sample_images = True
        else:
            # Every 20th of the way through training
            generate_sample_images = (
                int(20 * epoch / n_epoch) > 20 * (epoch - 1) / n_epoch
            )
        # But always generate samples for first two epochs and last epoch
        if epoch <= 2 or epoch == n_epoch:
            generate_sample_images = True

        def ensure_clim_met(x, x0=0.0, x1=1.0):
            x = x.clone()
            x[0, :, 0, 0] = 0
            x[0, :, 0, 1] = 1
            return x

        def add_image_border(x):
            """
            Add a green border around a a tensor of images.

            Parameters
            ----------
            x : torch.Tensor
                Tensor in NCWH or NCHW format.

            Returns
            -------
            torch.Tensor
                As ``x``, but padded with a green border.
            """
            if x.shape[1] == 1:
                x = torch.cat([x, x, x], dim=1)
            if x.shape[1] != 3:
                raise ValueError("RGB image needs three color channels")
            shp = list(x.shape)
            shp[-1] = 1
            x = torch.cat(
                [
                    torch.zeros(shp, dtype=x.dtype, device=x.device),
                    x,
                    torch.zeros(shp, dtype=x.dtype, device=x.device),
                ],
                dim=-1,
            )
            shp = list(x.shape)
            shp[-2] = 1
            x = torch.cat(
                [
                    torch.zeros(shp, dtype=x.dtype, device=x.device),
                    x,
                    torch.zeros(shp, dtype=x.dtype, device=x.device),
                ],
                dim=-2,
            )
            x[:, 1, :, 0] = 1.0
            x[:, 1, :, -1] = 1.0
            x[:, 1, 0, :] = 1.0
            x[:, 1, -1, :] = 1.0
            return x

        # Add example images to tensorboard
        for (ex_input, ex_data, ex_output), partition in (
            ((ex_input_tr, ex_data_tr, ex_output_tr), "Train"),
            ((ex_input_val, ex_data_val, ex_output_val), "Val"),
            ((ex_input_augval, ex_data_augval, ex_output_augval), "ValAug"),
        ):
            if not generate_sample_images:
                continue
            writer.add_images(
                "Input/" + partition,
                ex_input,
                epoch,
                dataformats="NCWH",
            )
            writer.add_images(
                "Overall/" + partition + "/Target",
                ensure_clim_met(add_image_border(ex_data["mask"].float().unsqueeze(1))),
                epoch,
                dataformats="NCWH",
            )
            writer.add_images(
                "Overall/" + partition + "/Output/p",
                ensure_clim_met(
                    add_image_border(ex_output["p_keep_pixel"].unsqueeze(1))
                ),
                epoch,
                dataformats="NCWH",
            )
            writer.add_images(
                "Overall/" + partition + "/Overlap",
                ensure_clim_met(
                    add_image_border(
                        torch.stack(
                            [
                                ex_output["mask_keep_pixel"].float(),
                                torch.zeros_like(
                                    ex_output["mask_keep_pixel"], dtype=torch.float
                                ),
                                ex_data["mask"].float(),
                            ],
                            dim=1,
                        )
                    )
                ),
                epoch,
                dataformats="NCWH",
            )

        for k, plot_transects_k in PLOT_TRANSECTS.items():
            if not generate_sample_images:
                continue
            if "stationary" in dataset_name and k != "mobile":
                pass
            elif k not in dataset_name:
                continue
            plot_crop_depth = crop_depth
            if plot_crop_depth is None:
                plot_crop_depth = DEFAULT_CROP_DEPTH_PLOTS
            for transect_name in plot_transects_k:
                transect, prediction = generate_from_shards(
                    os.path.join(data_dir + "_sharded", transect_name),
                    model,
                    sample_shape=sample_shape,
                    crop_depth=plot_crop_depth,
                    device=device,
                    dtype=torch.float,
                )
                hf = plt.figure(figsize=(15, 9))
                plot_transect_predictions(
                    transect, prediction, cmap="viridis", linewidth=1
                )
                transect_name = transect_name.replace("/evExports", "")
                if epoch == n_epoch:
                    # Only save png if this is the final epoch
                    figpth = os.path.join(
                        "models",
                        dataset_name,
                        log_name,
                        "samples",
                        transect_name + "_output.png",
                    )
                    os.makedirs(os.path.dirname(figpth), exist_ok=True)
                    plt.savefig(figpth)
                writer.add_figure(transect_name, hf, epoch, close=True)

        # remember best loss and save checkpoint
        is_best = loss_val < best_loss_val
        best_loss_val = min(loss_val, best_loss_val)

        checkpoint = {
            "model_parameters": model_parameters,
            "sample_shape": sample_shape,
            "epoch": epoch,
            "state_dict": unet.state_dict(),
            "best_loss": best_loss_val,
            "optimizer": optimizer.state_dict(),
            "meters": meters_val,
            "data_center": DATA_CENTER,
            "data_deviation": DATA_DEVIATION,
            "center_method": CENTER_METHOD,
            "deviation_method": DEVIATION_METHOD,
            "nan_value": NAN_VALUE,
            "wrapper_mapping": model.mapping,
            "wrapper_params": model.params,
            "training_routine": "echofilter-train {}".format(echofilter.__version__),
        }
        if use_mixed_precision:
            checkpoint["amp"] = apex.amp.state_dict()

        if (
            schedule == "mesaonecycle"
            and epoch / n_epoch <= warmdown_pct
            and (epoch + 1) / n_epoch > warmdown_pct
        ):
            dup = "ep{}".format(epoch)
        else:
            dup = None
        save_checkpoint(
            checkpoint,
            is_best,
            dirname=os.path.join("models", dataset_name, log_name),
            dup=dup,
        )
        meters_to_csv(
            meters_val, is_best, dirname=os.path.join("models", dataset_name, log_name)
        )

        # Note how long everything took
        writer.add_scalar("time/batch", batch_time.avg, epoch)
        writer.add_scalar("time/batch/data", data_time.avg, epoch)
        writer.add_scalar("time/train", t_val_start - t_epoch_start, epoch)
        writer.add_scalar("time/val", t_val_end - t_val_start, epoch)
        writer.add_scalar("time/log", time.time() - t_val_end, epoch)
        writer.add_scalar("time/epoch", time.time() - t_epoch_start, epoch)

        # Ensure the tensorboard outputs for this epoch are flushed
        writer.flush()

    # Close tensorboard connection
    writer.close()


def build_dataset(
    dataset_name,
    data_dir,
    sample_shape,
    train_partition=None,
    val_partition=None,
    crop_depth=None,
    random_crop_args=None,
):
    """
    Construct a pytorch Dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset. This can optionally be a list of
        multiple datasets joined with ``"+"``.
    data_dir : str
        Path to root data directory, containing the dataset.
    sample_shape : iterable of length 2
        The shape which will be used for training.
    train_partition : str, optional
        Name of the partition to use for training. Can optionally be a list of
        multiple partitions joined with ``"+"``. Default is ``"train"``
        (except for ``stationary2`` where it is mixed).
    val_partition : str, optional
        Name of the partition to use for validation. Can optionally be a list
        of multiple partitions joined with ``"+"``. Default is ``"validate"``
        (except for ``stationary2`` where it is mixed).
    crop_depth : float or None, optional
        Depth at which to crop samples. Default is ``None``.
    random_crop_args : dict, optional
        Arguments to control the random crop used during training. Default is
        an empty dict, which uses the default arguments of
        :class`echofilter.data.transforms.RandomCropDepth`.

    Returns
    -------
    dataset_train : echofilter.data.dataset.TransectDataset
        Dataset of training samples.
    dataset_val : echofilter.data.dataset.TransectDataset
        Dataset of validation samples.
    dataset_augval : echofilter.data.dataset.TransectDataset
        Dataset of validation samples, appyling the training augmentation
        stack.
    """
    if random_crop_args is None:
        random_crop_args = {}

    if "+" in dataset_name:
        # Join multiple datasets together
        datasets = [
            build_dataset(
                subdataset_name,
                data_dir=data_dir,
                sample_shape=sample_shape,
                train_partition=train_partition,
                val_partition=val_partition,
                crop_depth=crop_depth,
                random_crop_args=random_crop_args,
            )
            for subdataset_name in dataset_name.split("+")
            if len(subdataset_name) > 0
        ]
        return tuple(
            echofilter.data.dataset.ConcatDataset([d[i] for d in datasets])
            for i in range(len(datasets[0]))
        )

    if dataset_name == "stationary2":
        # The stationary2 dataset is MinasPassage and GrandPassage,
        # plus a second duplicate copy of GrandPassage which uses only zoomed
        # out depth crops.
        random_crop_args2 = copy.deepcopy(random_crop_args)
        random_crop_args2["p_crop_is_none"] = 0.2
        random_crop_args2["p_crop_is_optimal"] = 0.0
        random_crop_args2["p_crop_is_close"] = 0.0
        # By default, we only evaluate on the validation partition of
        # MinasPassage, and train on the train partition of both plus the
        # validation partition of GrandPassage.
        if train_partition is None and val_partition is None:
            train_partition_main = "train"
            val_partition_main = "validate"
            train_partition_aux = "train+validate"
            val_partition_aux = ""
        else:
            train_partition_main = train_partition_aux = train_partition
            val_partition_main = val_partition_aux = val_partition
        # Assemble the datasets
        datasets = [
            build_dataset(
                "MinasPassage",
                data_dir=data_dir,
                sample_shape=sample_shape,
                train_partition=train_partition_main,
                val_partition=val_partition_main,
                crop_depth=crop_depth,
                random_crop_args=random_crop_args,
            ),
            build_dataset(
                "GrandPassage",
                data_dir=data_dir,
                sample_shape=sample_shape,
                train_partition=train_partition_aux,
                val_partition=val_partition_aux,
                crop_depth=crop_depth,
                random_crop_args=random_crop_args,
            ),
            build_dataset(
                "GrandPassage",
                data_dir=data_dir,
                sample_shape=sample_shape,
                train_partition=train_partition_aux,
                val_partition=val_partition_aux,
                crop_depth=crop_depth,
                random_crop_args=random_crop_args2,
            ),
        ]
        return tuple(
            echofilter.data.dataset.ConcatDataset([d[i] for d in datasets])
            for i in range(len(datasets[0]))
        )

    if train_partition is None:
        train_partition = "train"
    if val_partition is None:
        val_partition = "validate"

    # Augmentations
    train_transform = torchvision.transforms.Compose(
        [
            echofilter.data.transforms.RandomCropDepth(**random_crop_args),
            echofilter.data.transforms.RandomReflection(),
            echofilter.data.transforms.Normalize(DATA_CENTER, DATA_DEVIATION),
            echofilter.data.transforms.ColorJitter(0.5, 0.3),
            echofilter.data.transforms.ReplaceNan(NAN_VALUE),
            echofilter.data.transforms.RandomElasticGrid(
                sample_shape,
                order=None,
                p=0.5,
                sigma=[8, 16],
                alpha=0.1,
            ),
            echofilter.data.transforms.Rescale(sample_shape, order=None),
        ]
    )
    val_transform = torchvision.transforms.Compose(
        [
            echofilter.data.transforms.OptimalCropDepth(),
            echofilter.data.transforms.Normalize(DATA_CENTER, DATA_DEVIATION),
            echofilter.data.transforms.ReplaceNan(NAN_VALUE),
            echofilter.data.transforms.Rescale(sample_shape, order=1),
        ]
    )

    train_paths = []
    for partition_name in train_partition.split("+"):
        if len(partition_name) == 0:
            continue
        train_paths += get_partition_list(
            partition_name,
            dataset=dataset_name,
            partitioning_version="firstpass",
            root_data_dir=data_dir,
            full_path=True,
            sharded=True,
        )
    val_paths = []
    for partition_name in val_partition.split("+"):
        if len(partition_name) == 0:
            continue
        val_paths += get_partition_list(
            partition_name,
            dataset=dataset_name,
            partitioning_version="firstpass",
            root_data_dir=data_dir,
            full_path=True,
            sharded=True,
        )
    print(
        "Found {:3d} train sample paths from partition {} for dataset {}".format(
            len(train_paths), train_partition, dataset_name
        )
    )
    print(
        "Found {:3d} val sample paths from partition {} for dataset {}".format(
            len(val_paths), val_partition, dataset_name
        )
    )

    dataset_args = {}
    if dataset_name == "mobile":
        dataset_args["remove_nearfield"] = False
        dataset_args["remove_offset_turbulence"] = 0
        dataset_args["remove_offset_bottom"] = 1.0
    elif dataset_name == "MinasPassage":
        dataset_args["remove_nearfield"] = True
        # NB: Nearfield distance is 1.7, except for the samples:
        # march2018_D20180502-T045216_D20180502-T102215
        # march2018_D20180502-T105216_D20180502-T162215
        # march2018_D20180502-T165215_D20180502-T222216
        # march2018_D20180502-T225214_D20180503-T042215
        # For which it is 1.744
        dataset_args["nearfield_distance"] = 1.745
        dataset_args["remove_offset_turbulence"] = 0
        dataset_args["remove_offset_bottom"] = 0
    elif dataset_name == "GrandPassage":
        dataset_args["remove_nearfield"] = True
        dataset_args["nearfield_distance"] = 1.7
        dataset_args["remove_offset_turbulence"] = 1.0
        dataset_args["remove_offset_bottom"] = 0

    dataset_train = echofilter.data.dataset.TransectDataset(
        train_paths,
        window_len=sample_shape[0],
        p_scale_window=0.8,
        num_windows_per_transect=None,
        use_dynamic_offsets=True,
        crop_depth=crop_depth,
        transform=train_transform,
        **dataset_args,
    )
    dataset_val = echofilter.data.dataset.TransectDataset(
        val_paths,
        window_len=sample_shape[0],
        p_scale_window=0,
        num_windows_per_transect=None,
        use_dynamic_offsets=False,
        crop_depth=crop_depth,
        transform=val_transform,
        **dataset_args,
    )
    dataset_augval = echofilter.data.dataset.TransectDataset(
        val_paths,
        window_len=sample_shape[0],
        p_scale_window=0.8,
        num_windows_per_transect=None,
        use_dynamic_offsets=False,
        crop_depth=crop_depth,
        transform=train_transform,
        **dataset_args,
    )
    return dataset_train, dataset_val, dataset_augval


def train_epoch(
    loader,
    model,
    criterion,
    optimizer,
    device,
    epoch,
    dtype=torch.float,
    print_freq=10,
    schedule_data=None,
    use_mixed_precision=False,
    continue_through_error=True,
):
    """
    Train a model through a single epoch of the dataset.

    Parameters
    ----------
    loader : iterable, torch.utils.data.DataLoader
        Dataloader.
    model : callable, echofilter.nn.wrapper.Echofilter
        Model.
    criterion : callable, torch.nn.modules.loss._Loss
        Loss function.
    device : str or torch.device
        Which device the data should be loaded onto.
    epoch : int
        Which epoch is being performed.
    dtype : str or torch.dtype
        Datatype which which the data should be loaded.
    print_freq : int, optional
        Number of batches between reporting progress. Default is ``10``.
    schedule_data : dict or None
        If a learning rate schedule is being used, this may be passed as a
        dictionary with the key ``"scheduler"`` mapping to the learning rate
        schedule as a callable.
    use_mixed_precision : bool
        Whether to use :meth:`apex.amp.scale_loss` to automatically scale the
        loss. Default is ``False``.
    continue_through_error : bool
        Whether to catch errors within an individual batch, ignore them and
        continue running training on the rest of the batches. If there are
        five or more errors while processing the batch, training will halt
        regardless of ``continue_through_error``. Default is ``True``.

    Returns
    -------
    average_loss : float
        Average loss as given by criterion (weighted equally for each sample
        in ``loader``).
    meters : dict of dict
        Each key is a strata of the model output, each mapping to a their own
        dictionary of evaluation criterions: "Accuracy", "Precision", "Recall",
        "F1 Score", "Jaccard".
    examples : tuple of torch.Tensor
        Tuple of `(example_input, example_data, example_output)`.
    timing : tuple of floats
        Tuple of `(batch_time, data_time)`.
    """
    if schedule_data is None:
        schedule_data = {"name": "constant"}

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.3f")

    meters = {}
    for chn in [
        "Overall",
        "Turbulence",
        "Bottom",
        "RemovedSeg",
        "Passive",
        "Patch",
        "Surface",
    ]:
        for condition in model.conditions:
            cs = condition
            if condition != "":
                cs = "|" + condition
            cc = chn + cs
            meters[cc] = {}
            meters[cc]["Accuracy"] = AverageMeter("Accuracy (" + cc + ")", ":6.2f")
            meters[cc]["Precision"] = AverageMeter("Precision (" + cc + ")", ":6.2f")
            meters[cc]["Recall"] = AverageMeter("Recall (" + cc + ")", ":6.2f")
            meters[cc]["F1 Score"] = AverageMeter("F1 Score (" + cc + ")", ":6.4f")
            meters[cc]["Jaccard"] = AverageMeter("Jaccard (" + cc + ")", ":6.4f")

    progress = ProgressMeter(
        len(loader),
        [
            batch_time,
            data_time,
            losses,
            meters["Overall"]["Accuracy"],
            meters["Overall"]["Jaccard"],
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    example_input = example_data = example_output = None
    n_backward_errors = 0

    end = time.time()
    for i, (input, metadata) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device, dtype, non_blocking=True)
        metadata = {
            k: v.to(device, dtype, non_blocking=True) for k, v in metadata.items()
        }

        # Compute output
        output = model(input)
        loss = criterion(output, metadata)
        # Record loss
        ns = input.size(0)
        losses.update(loss.item(), ns)

        with torch.no_grad():
            if i == max(0, len(loader) - 2):
                example_input = input.detach()
                example_data = {k: v.detach() for k, v in metadata.items()}
                example_output = output.detach()

            # Measure and record performance with various metrics
            for chn_cond, meters_k in meters.items():

                chnparts = chn_cond.split("|")
                chn = chnparts[0].lower()
                if len(chnparts) < 2:
                    cs = cond = ""
                else:
                    cond = chnparts[1]
                    cs = "|" + cond

                if chn.startswith("overall"):
                    output_k = output["mask_keep_pixel" + cs].float()
                    target_k = metadata["mask"]
                elif chn.startswith("turbulence"):
                    output_k = output["p_is_below_turbulence" + cs]
                    target_k = 1 - metadata["mask_turbulence"]
                elif chn.startswith("surf"):
                    output_k = output["p_is_below_surface" + cs]
                    target_k = 1 - metadata["mask_surface"]
                elif chn.startswith("bottom"):
                    output_k = output["p_is_above_bottom" + cs]
                    target_k = 1 - metadata["mask_bottom"]
                elif chn.startswith("removedseg"):
                    output_k = output["p_is_removed" + cs]
                    target_k = metadata["is_removed"]
                elif chn.startswith("passive"):
                    output_k = output["p_is_passive" + cs]
                    target_k = metadata["is_passive"]
                elif chn.startswith("patch"):
                    output_k = output["p_is_patch" + cs]
                    target_k = metadata["mask_patches"]
                else:
                    raise ValueError("Unrecognised output channel: {}".format(chn))

                if cond:
                    if cond.startswith("up"):
                        mask = metadata["is_upward_facing"] > 0.5
                    elif cond.startswith("down"):
                        mask = metadata["is_upward_facing"] < 0.5
                    else:
                        raise ValueError("Unsupported condition {}".format(cond))
                    if torch.sum(mask).item() == 0:
                        continue
                    output_k = output_k[mask]
                    target_k = target_k[mask]

                for c, v in meters_k.items():
                    c = c.lower()
                    if c == "accuracy":
                        v.update(
                            100.0 * criterions.mask_accuracy(output_k, target_k).item(),
                            ns,
                        )
                    elif c == "precision":
                        v.update(
                            100.0
                            * criterions.mask_precision(output_k, target_k).item(),
                            ns,
                        )
                    elif c == "recall":
                        v.update(
                            100.0 * criterions.mask_recall(output_k, target_k).item(),
                            ns,
                        )
                    elif c == "f1 score" or c == "f1":
                        v.update(
                            criterions.mask_f1_score(output_k, target_k).item(), ns
                        )
                    elif c == "jaccard":
                        v.update(
                            criterions.mask_jaccard_index(output_k, target_k).item(), ns
                        )
                    elif c == "active output":
                        v.update(
                            100.0 * criterions.mask_active_fraction(output_k).item(), ns
                        )
                    elif c == "active target":
                        v.update(
                            100.0 * criterions.mask_active_fraction(target_k).item(), ns
                        )
                    else:
                        raise ValueError("Unrecognised criterion: {}".format(c))

        # compute gradient and do optimizer update step
        optimizer.zero_grad()

        try:
            if use_mixed_precision:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
        except Exception as ex:
            n_backward_errors += 1
            if n_backward_errors > 5:
                # If there have been more than 5 faulty batches in this epoch, we halt
                # now since it seems like a systemic problem and not a one-off.
                continue_through_error = False
            if not continue_through_error:
                raise ex
            print("Error in backward step:")
            print("".join(traceback.TracebackException.from_exception(ex).format()))

        optimizer.step()
        if "scheduler" in schedule_data:
            schedule_data["scheduler"].step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 or i + 1 == len(loader):
            progress.display(i + 1)

    return (
        losses.avg,
        meters,
        (example_input, example_data, example_output),
        (batch_time, data_time),
    )


def validate(
    loader,
    model,
    criterion,
    device,
    dtype=torch.float,
    print_freq=10,
    prefix="Test",
    num_examples=32,
):
    """
    Validate the model's performance on the validation partition.

    Parameters
    ----------
    loader : iterable, torch.utils.data.DataLoader
        Dataloader.
    model : callable, echofilter.nn.wrapper.Echofilter
        Model.
    criterion : callable, torch.nn.modules.loss._Loss
        Loss function.
    device : str or torch.device
        Which device the data should be loaded onto.
    dtype : str or torch.dtype
        Datatype which which the data should be loaded.
    print_freq : int, optional
        Number of batches between reporting progress. Default is ``10``.
    prefix : str, optional
        Prefix string to prepend to progress meter names. Default is ``"Test"``.
    num_examples : int, optional
        Number of example inputs to return. Default is ``32``.

    Returns
    -------
    average_loss : float
        Average loss as given by criterion (weighted equally for each sample
        in ``loader``).
    meters : dict of dict
        Each key is a strata of the model output, each mapping to a their own
        dictionary of evaluation criterions: "Accuracy", "Precision", "Recall",
        "F1 Score", "Jaccard".
    examples : tuple of torch.Tensor
        Tuple of `(example_input, example_data, example_output)`.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.3f")

    meters = {}
    for chn in [
        "Overall",
        "Turbulence",
        "Bottom",
        "RemovedSeg",
        "Passive",
        "Patch",
        "Surface",
    ]:
        for condition in model.conditions:
            cs = condition
            if condition != "":
                cs = "|" + condition
            cc = chn + cs
            meters[cc] = {}
            meters[cc]["Accuracy"] = AverageMeter("Accuracy (" + cc + ")", ":6.2f")
            meters[cc]["Precision"] = AverageMeter("Precision (" + cc + ")", ":6.2f")
            meters[cc]["Recall"] = AverageMeter("Recall (" + cc + ")", ":6.2f")
            meters[cc]["F1 Score"] = AverageMeter("F1 Score (" + cc + ")", ":6.4f")
            meters[cc]["Jaccard"] = AverageMeter("Jaccard (" + cc + ")", ":6.4f")

    progress = ProgressMeter(
        len(loader),
        [
            batch_time,
            data_time,
            losses,
            meters["Overall"]["Accuracy"],
            meters["Overall"]["Jaccard"],
        ],
        prefix=prefix + ": ",
    )

    # switch to evaluate mode
    model.eval()

    example_input = []
    example_data = []
    example_output = []
    example_interval = max(1, len(loader) // num_examples)

    with torch.no_grad():
        end = time.time()
        for i, (input, metadata) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.to(device, dtype, non_blocking=True)
            metadata = {
                k: v.to(device, dtype, non_blocking=True) for k, v in metadata.items()
            }

            # Compute output
            output = model(input)
            loss = criterion(output, metadata)
            # Record loss
            ns = input.size(0)
            losses.update(loss.item(), ns)

            if i % example_interval == 0 and len(example_input) < num_examples:
                example_input.append(input[0].detach())
                example_data.append({k: v[0].detach() for k, v in metadata.items()})
                example_output.append({k: v[0].detach() for k, v in output.items()})

            # Measure and record performance with various metrics
            for chn_cond, meters_k in meters.items():

                chnparts = chn_cond.split("|")
                chn = chnparts[0].lower()
                if len(chnparts) < 2:
                    cs = cond = ""
                else:
                    cond = chnparts[1]
                    cs = "|" + cond

                if chn.startswith("overall"):
                    output_k = output["mask_keep_pixel" + cs].float()
                    target_k = metadata["mask"]
                elif chn.startswith("turbulence"):
                    output_k = output["p_is_below_turbulence" + cs]
                    target_k = 1 - metadata["mask_turbulence"]
                elif chn.startswith("surf"):
                    output_k = output["p_is_below_surface" + cs]
                    target_k = 1 - metadata["mask_surface"]
                elif chn.startswith("bottom"):
                    output_k = output["p_is_above_bottom" + cs]
                    target_k = 1 - metadata["mask_bottom"]
                elif chn.startswith("removedseg"):
                    output_k = output["p_is_removed" + cs]
                    target_k = metadata["is_removed"]
                elif chn.startswith("passive"):
                    output_k = output["p_is_passive" + cs]
                    target_k = metadata["is_passive"]
                elif chn.startswith("patch"):
                    output_k = output["p_is_patch" + cs]
                    target_k = metadata["mask_patches"]
                else:
                    raise ValueError("Unrecognised output channel: {}".format(chn))

                if cond:
                    if cond.startswith("up"):
                        mask = metadata["is_upward_facing"] > 0.5
                    elif cond.startswith("down"):
                        mask = metadata["is_upward_facing"] < 0.5
                    else:
                        raise ValueError("Unsupported condition {}".format(cond))
                    if torch.sum(mask).item() == 0:
                        continue
                    output_k = output_k[mask]
                    target_k = target_k[mask]

                for c, v in meters_k.items():
                    c = c.lower()
                    if c == "accuracy":
                        v.update(
                            100.0 * criterions.mask_accuracy(output_k, target_k).item(),
                            ns,
                        )
                    elif c == "precision":
                        v.update(
                            100.0
                            * criterions.mask_precision(output_k, target_k).item(),
                            ns,
                        )
                    elif c == "recall":
                        v.update(
                            100.0 * criterions.mask_recall(output_k, target_k).item(),
                            ns,
                        )
                    elif c == "f1 score" or c == "f1":
                        v.update(
                            criterions.mask_f1_score(output_k, target_k).item(), ns
                        )
                    elif c == "jaccard":
                        v.update(
                            criterions.mask_jaccard_index(output_k, target_k).item(), ns
                        )
                    elif c == "active output":
                        v.update(
                            100.0 * criterions.mask_active_fraction(output_k).item(), ns
                        )
                    elif c == "active target":
                        v.update(
                            100.0 * criterions.mask_active_fraction(target_k).item(), ns
                        )
                    else:
                        raise ValueError("Unrecognised criterion: {}".format(c))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0 or i + 1 == len(loader):
                progress.display(i + 1)

    # Restack samples, converting list into higher-dim tensor
    example_input = torch.stack(example_input, dim=0)
    example_data = {
        k: torch.stack([a[k] for a in example_data], dim=0) for k in example_data[0]
    }
    example_output = {
        k: torch.stack([a[k] for a in example_output], dim=0) for k in example_output[0]
    }

    return losses.avg, meters, (example_input, example_data, example_output)


def generate_from_transect(model, transect, sample_shape, device, dtype=torch.float):
    """
    Generate an output for a sample transect, .
    """
    # Put model in evaluation mode
    model.eval()

    # Make a copy of the transect which we will use to
    data = copy.deepcopy(transect)

    # Configure data to match what the model expects to see
    # Ensure depth is always increasing (which corresponds to descending from
    # the air down the water column)
    if data["depths"][-1] < data["depths"][0]:
        # Found some upward-facing data that still needs to be reflected
        for k in ["depths", "signals", "mask"]:
            data[k] = np.flip(data[k], -1).copy()

    # Apply transforms
    transform = torchvision.transforms.Compose(
        [
            echofilter.data.transforms.Normalize(DATA_CENTER, DATA_DEVIATION),
            echofilter.data.transforms.ReplaceNan(NAN_VALUE),
            echofilter.data.transforms.Rescale(
                (data["signals"].shape[0], sample_shape[1])
            ),
        ]
    )
    data = transform(data)
    input = torch.tensor(data["signals"]).unsqueeze(0).unsqueeze(0)
    input = input.to(device, dtype).contiguous()
    # Put data through model
    with torch.no_grad():
        output = model(input)
        output = {k: v.squeeze(0).cpu().numpy() for k, v in output.items()}

    output["depths"] = data["depths"]
    output["timestamps"] = data["timestamps"]

    return output


def _generate_from_loaded(transect, model, *args, crop_depth=None, **kwargs):
    """
    Generate an output from a loaded transect.
    """
    # Crop long input
    for key in (
        echofilter.data.transforms._fields_2d
        + echofilter.data.transforms._fields_1d_timelike
    ):
        if key in transect:
            transect[key] = transect[key][:MAX_INPUT_LEN]

    # Apply depth crop
    if crop_depth is not None:
        if transect["is_upward_facing"]:
            depth_crop_mask = (
                transect["depths"] >= np.max(transect["depths"]) - crop_depth
            )
        else:
            depth_crop_mask = transect["depths"] <= crop_depth

        for key in echofilter.data.transforms._fields_2d:
            if key in transect:
                transect[key] = transect[key][:, depth_crop_mask]

        for key in echofilter.data.transforms._fields_1d_depthlike:
            if key in transect:
                transect[key] = transect[key][depth_crop_mask]

    # Convert lines to masks
    ddepths = np.broadcast_to(transect["depths"], transect["Sv"].shape)
    transect["mask_turbulence"] = np.single(
        ddepths < np.expand_dims(transect["turbulence"], -1)
    )
    transect["mask_bottom"] = np.single(
        ddepths > np.expand_dims(transect["bottom"], -1)
    )
    # Add mask_patches to the data, for plotting
    transect["mask_patches"] = 1 - transect["mask"]
    transect["mask_patches"][transect["is_passive"] > 0.5] = 0
    transect["mask_patches"][transect["is_removed"] > 0.5] = 0
    transect["mask_patches"][transect["mask_turbulence"] > 0.5] = 0
    transect["mask_patches"][transect["mask_bottom"] > 0.5] = 0

    # Generate predictions for the transect
    transect["signals"] = transect.pop("Sv")
    prediction = generate_from_transect(model, transect, *args, **kwargs)
    transect["Sv"] = transect.pop("signals")

    return transect, prediction


def generate_from_file(fname, *args, **kwargs):
    """
    Generate an output for a sample transect, specified by its file path.
    """
    # Load the data
    transect = load_decomposed_transect_mask(fname)
    # Process the transect
    return _generate_from_loaded(transect, *args, **kwargs)


def generate_from_shards(fname, *args, **kwargs):
    """
    Generate an output for a sample transect, specified by a path to sharded data.
    """
    # Load the data
    transect = echofilter.raw.shardloader.load_transect_segments_from_shards_abs(fname)
    # Process the transect
    return _generate_from_loaded(transect, *args, **kwargs)


def save_checkpoint(state, is_best, dirname=".", fname_fmt="checkpoint{}.pt", dup=None):
    """
    Save a model checkpoint, using :meth:`torch.save`.

    Parameters
    ----------
    state : dict
        Model checkpoint state to record.
    is_best : bool
        Whether this model state is the best so far. If ``True``, the best
        checkpoint (by default named ``"checkpoint_best.pt"``) will be overwritten
        with this ``state``.
    dirname : str, optional
        Path to directory in which the checkpoint will be saved.
        Default is ``"."`` (current directory of the executed script).
    fname_fmt : str, optional
        Format for the file name(s) of the saved checkpoint(s). Must include
        one string argument output. Default is ``"checkpoint{}.pt"``.
    dup : str or None
        If this is not ``None``, a duplicate copy of the checkpoint is recorded
        in accordance with ``fname_fmt``. By default the duplicate output file
        name will be styled as ``"checkpoint_<dup>.pt"``.
    """
    os.makedirs(dirname, exist_ok=True)
    fname = os.path.join(dirname, fname_fmt.format(""))
    torch.save(state, fname)
    if is_best:
        shutil.copyfile(fname, os.path.join(dirname, fname_fmt.format("_best")))
    if dup:
        shutil.copyfile(
            fname, os.path.join(dirname, fname_fmt.format("_{}".format(dup)))
        )


def meters_to_csv(meters, is_best, dirname=".", filename="meters.csv"):
    """
    Export performance metrics to CSV format.

    Parameters
    ----------
    meters : dict of dict
        Collection of output meters, as a nested dictionary.
    is_best : bool
        Whether this model state is the best so far. If ``True``, the CSV file
        will be copied to ``"model_best.meters.csv"``.
    dirname : str, optional
        Path to directory in which the checkpoint will be saved.
        Default is ``"."`` (current directory of the executed script).
    filename : str, optional
        Format for the output file. Default is ``"meters.csv"``.
    """
    os.makedirs(dirname, exist_ok=True)
    df = pd.DataFrame()
    for chn in meters:
        if "|" in chn:
            # Skip conditional model evaluations
            continue
        # For each output plane
        for _, meter in meters[chn].items():
            # For each criterion
            df[meter.name] = meter.values
    df.to_csv(os.path.join(dirname, filename), index=False)
    if is_best:
        shutil.copyfile(
            os.path.join(dirname, filename),
            os.path.join(dirname, "model_best.meters.csv"),
        )


if __name__ == "__main__":
    main()
