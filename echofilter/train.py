#!/usr/bin/env python

from collections import OrderedDict
import copy
import datetime
import os
import pprint
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ranger
import torch
import torch.nn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms
from torchutils.random import seed_all
from torchutils.utils import count_parameters

try:
    import apex
except ImportError:
    apex = None

import echofilter.dataset
from echofilter.nn.unet import UNet
from echofilter.nn.wrapper import Echofilter, EchofilterLoss
from echofilter.optim import criterions, schedulers
from echofilter.optim.meters import AverageMeter, ProgressMeter
from echofilter.plotting import plot_transect_predictions
from echofilter.raw.loader import get_partition_list
from echofilter.raw.manipulate import load_decomposed_transect_mask
import echofilter.raw.shardloader
import echofilter.transforms
import echofilter.utils


## For mobile dataset,
# DATA_CENTER = -81.5
# DATA_DEVIATION = 21.9
# CENTER_METHOD = 'mean'
# DEVIATION_METHOD = 'stdev'

## For stationary dataset,
# DATA_CENTER = -78.7
# DATA_DEVIATION = 19.2
# CENTER_METHOD = 'mean'
# DEVIATION_METHOD = 'stdev'

## For intermediate values between both datasets
# DATA_CENTER = -80.
# DATA_DEVIATION = 20.
# CENTER_METHOD = 'mean'
# DEVIATION_METHOD = 'stdev'
# NAN_VALUE = -3

# Overall values to use
DATA_CENTER = -97.5
DATA_DEVIATION = 16.5
CENTER_METHOD = "pc10"
DEVIATION_METHOD = "idr"

NAN_VALUE = -1

# Transects to plot for debugging
PLOT_TRANSECTS = {
    "mobile": [
        "mobile/Survey07/Survey07_GR4_N5W_survey7",
        "mobile/Survey14/Survey14_GR4_N0W_E",
        "mobile/Survey16/Survey16_GR4_N5W_E",
        "mobile/Survey17/Survey17_GR4_N5W_E",
    ],
    "MinasPassage": [
        "MinasPassage/december2017/evExports/december2017_D20180213-T115216_D20180213-T172216",
        "MinasPassage/march2018/evExports/march2018_D20180513-T195216_D20180514-T012216",
        "MinasPassage/september2018/evExports/september2018_D20181027-T202217_D20181028-T015217",
        "MinasPassage/september2018/evExports/september2018_D20181107-T122220_D20181107-T175217",
    ],
    "GrandPassage": [
        "GrandPassage/phase1/GrandPassage_WBAT_2A_20191222",
        "GrandPassage/phase2/GrandPassage_WBAT_2B_20200125_UTC160020_ebblow",
    ],
}

DEFAULT_CROP_DEPTH_PLOTS = 70
MAX_INPUT_LEN = 4000


def train(
    data_dir="/data/dsforce/surveyExports",
    dataset_name="mobile",
    sample_shape=(128, 512),
    crop_depth=None,
    resume="",
    log_name=None,
    log_name_append=None,
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
    n_worker=8,
    batch_size=16,
    n_epoch=20,
    seed=None,
    print_freq=50,
    optimizer="rangerva",
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

    seed_all(seed)

    # Input handling
    schedule = schedule.lower()

    if base_momentum is None:
        base_momentum = momentum

    if log_name is None or log_name == "":
        log_name = datetime.datetime.now().strftime("%Y-%b-%d_%H:%M:%S")
    if log_name_append is None:
        log_name_append = os.uname()[1]
    if len(log_name_append) > 0:
        log_name += "_" + log_name_append

    print("Output will be written to {}/{}".format(dataset_name, log_name))

    if use_mixed_precision is None:
        use_mixed_precision = not "cpu" in device
    if use_mixed_precision and apex is None:
        print("NVIDIA apex must be installed to use mixed precision.")
        use_mixed_precision = False

    # Need to set the default device for apex.amp
    if device is not None and device != "cpu":
        torch.cuda.set_device(torch.device(device))

    # Build dataset
    dataset_train, dataset_val, dataset_augval = build_dataset(
        dataset_name, data_dir, sample_shape
    )

    print("Train dataset has {:4d} samples".format(len(dataset_train)))
    print("Val   dataset has {:4d} samples".format(len(dataset_val)))

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_worker,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=echofilter.utils.worker_seed_fn,
    )
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_worker,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=echofilter.utils.worker_staticseed_fn,
    )
    loader_augval = torch.utils.data.DataLoader(
        dataset_augval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_worker,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=echofilter.utils.worker_staticseed_fn,
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
    print()
    pprint.pprint(model_parameters)
    print()

    model = Echofilter(UNet(**model_parameters), top="boundary", bottom="boundary",)
    model.to(device)
    print(
        "Built model with {} trainable parameters".format(
            count_parameters(model, only_trainable=True)
        )
    )

    # define loss function (criterion) and optimizer
    criterion = EchofilterLoss(overall=overall_loss_weight)

    optimizer_name = optimizer.lower()
    if optimizer_name == "adam":
        optimizer_class = torch.optim.Adam
    elif optimizer_name == "adamw":
        optimizer_class = torch.optim.AdamW
    elif optimizer_name == "ranger":
        optimizer_class = ranger.Ranger
    elif optimizer_name == "rangerva":
        optimizer_class = ranger.RangerVA
    elif optimizer_name == "rangerqh":
        optimizer_class = ranger.RangerQH
    else:
        # We don't support arbitrary optimizers from torch.optim because they
        # need different configuration parameters to Adam.
        raise ValueError("Unrecognised optimizer: {}".format(optimizer))

    optimizer = optimizer_class(
        model.parameters(), lr, betas=(momentum, 0.999), weight_decay=weight_decay,
    )
    schedule_data = {"name": schedule}

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
            div_factor=1e3,
            final_div_factor=1e5,
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
            div_factor=1e3,
            final_div_factor=1e5,
        )
    else:
        raise ValueError("Unsupported schedule: {}".format(schedule))

    if use_mixed_precision:
        print('Converting model to mixed precision, opt="{}"'.format(amp_opt))
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=amp_opt)

    # Make a tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join("runs", dataset_name, log_name))

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
        start_epoch = checkpoint["epoch"] + 1
        best_loss_val = checkpoint["best_loss"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if use_mixed_precision and "amp" in checkpoint:
            apex.amp.load_state_dict(checkpoint["amp"])
        print("Loaded checkpoint '{}' (epoch {})".format(resume, checkpoint["epoch"]))

    print("Starting training")
    t_start = time.time()
    for epoch in range(start_epoch, n_epoch + 1):

        t_epoch_start = time.time()

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
            "Completed {} of {} epochs in {}".format(
                epoch, n_epoch, datetime.timedelta(seconds=time.time() - t_start)
            )
        )
        # Print metrics to terminal
        name_fmt = "{:.<28s}"
        current_lr = echofilter.utils.get_current_lr(optimizer)
        current_mom = echofilter.utils.get_current_momentum(optimizer)

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
        elif n_epoch < 100:
            # Every 10th of the way through training
            generate_sample_images = (epoch % (n_epoch // 10)) == 0
        else:
            # Every 10th epoch
            generate_sample_images = (epoch % 10) == 0
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
                As `x`, but padded with a green border.
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
                "Input/" + partition, ex_input, epoch, dataformats="NCWH",
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
            if plot_crop_depth is None and plot_transects_k == "mobile":
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
            "state_dict": model.state_dict(),
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
        }
        if use_mixed_precision:
            checkpoint["amp"] = apex.amp.state_dict()
        save_checkpoint(
            checkpoint, is_best, dirname=os.path.join("models", dataset_name, log_name),
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
    random_crop_args={},
):
    """
    Construct a pytorch Dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    data_dir : str
        Path to root data directory, containing the dataset.
    sample_shape : iterable of length 2
        The shape which will be used for training.
    train_partition : str, optional
        Name of the partition to use for training. Can optionally be a list of
        multiple partitions joined with `"+"`. Default is `"train"`
        (except for `stationary2` where it is mixed).
    val_partition : str, optional
        Name of the partition to use for validation. Can optionally be a list
        of multiple partitions joined with `"+"`. Default is `"validate"`
        (except for `stationary2` where it is mixed).
    crop_depth : float or None, optional
        Depth at which to crop samples. Default is `None`.
    random_crop_args : dict, optional
        Arguments to control the random crop used during training. Default is
        an empty dict, which uses the default arguments of
        `echofilter.transforms.RandomCropDepth`.

    Returns
    -------
    dataset_train : echofilter.dataset.TransectDataset
        Dataset of training samples.
    dataset_val : echofilter.dataset.TransectDataset
        Dataset of validation samples.
    dataset_augval : echofilter.dataset.TransectDataset
        Dataset of validation samples, appyling the training augmentation
        stack.
    """

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
            echofilter.dataset.ConcatDataset([d[i] for d in datasets])
            for i in range(len(datasets[0]))
        )

    if train_partition is None:
        train_partition = "train"
    if val_partition is None:
        val_partition = "validate"

    # Augmentations
    train_transform = torchvision.transforms.Compose(
        [
            echofilter.transforms.RandomCropDepth(**random_crop_args),
            echofilter.transforms.RandomReflection(),
            echofilter.transforms.Normalize(DATA_CENTER, DATA_DEVIATION),
            echofilter.transforms.ColorJitter(0.5, 0.3),
            echofilter.transforms.ReplaceNan(NAN_VALUE),
            echofilter.transforms.RandomGridSampling(sample_shape, order=None, p=0.5),
            echofilter.transforms.Rescale(sample_shape, order=None),
        ]
    )
    val_transform = torchvision.transforms.Compose(
        [
            echofilter.transforms.OptimalCropDepth(),
            echofilter.transforms.Normalize(DATA_CENTER, DATA_DEVIATION),
            echofilter.transforms.ReplaceNan(NAN_VALUE),
            echofilter.transforms.Rescale(sample_shape, order=1),
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
        dataset_args["remove_offset_top"] = 0
        dataset_args["remove_offset_bottom"] = 1.0
    elif dataset_name == "MinasPassage":
        dataset_args["remove_nearfield"] = True
        dataset_args["nearfield_distance"] = 1.7
        dataset_args["remove_offset_top"] = 0
        dataset_args["remove_offset_bottom"] = 0
    elif dataset_name == "GrandPassage":
        dataset_args["remove_nearfield"] = True
        dataset_args["nearfield_distance"] = 1.7
        dataset_args["remove_offset_top"] = 1.0
        dataset_args["remove_offset_bottom"] = 0

    dataset_train = echofilter.dataset.TransectDataset(
        train_paths,
        window_len=sample_shape[0],
        p_scale_window=0.8,
        num_windows_per_transect=None,
        use_dynamic_offsets=True,
        crop_depth=crop_depth,
        transform=train_transform,
        **dataset_args
    )
    dataset_val = echofilter.dataset.TransectDataset(
        val_paths,
        window_len=sample_shape[0],
        p_scale_window=0,
        num_windows_per_transect=None,
        use_dynamic_offsets=False,
        crop_depth=crop_depth,
        transform=val_transform,
        **dataset_args
    )
    dataset_augval = echofilter.dataset.TransectDataset(
        val_paths,
        window_len=sample_shape[0],
        p_scale_window=0.8,
        num_windows_per_transect=None,
        use_dynamic_offsets=False,
        crop_depth=crop_depth,
        transform=train_transform,
        **dataset_args
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
):
    if schedule_data is None:
        schedule_data = {"name": "constant"}

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")

    meters = {}
    for chn in ["Overall", "Top", "Bottom", "RemovedSeg", "Passive", "Patch"]:
        meters[chn] = {}
        meters[chn]["Accuracy"] = AverageMeter("Accuracy (" + chn + ")", ":6.2f")
        meters[chn]["Precision"] = AverageMeter("Precision (" + chn + ")", ":6.2f")
        meters[chn]["Recall"] = AverageMeter("Recall (" + chn + ")", ":6.2f")
        meters[chn]["F1 Score"] = AverageMeter("F1 Score (" + chn + ")", ":6.4f")
        meters[chn]["Jaccard"] = AverageMeter("Jaccard (" + chn + ")", ":6.4f")
        meters[chn]["Active output"] = AverageMeter(
            "Active output (" + chn + ")", ":6.2f"
        )
        meters[chn]["Active target"] = AverageMeter(
            "Active target (" + chn + ")", ":6.2f"
        )

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

    end = time.time()
    for i, (input, metadata) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device, dtype, non_blocking=True)
        metadata = {k: v.to(device, non_blocking=True) for k, v in metadata.items()}

        # Compute output
        output = model(input)
        loss = criterion(output, metadata)
        # Record loss
        ns = input.size(0)
        losses.update(loss.item(), ns)

        if i == max(0, len(loader) - 2):
            example_input = input.detach()
            example_data = {k: v.detach() for k, v in metadata.items()}
            example_output = output.detach()

        # Measure and record performance with various metrics
        for chn, meters_k in meters.items():
            chn = chn.lower()
            if chn == "overall":
                output_k = output["mask_keep_pixel"].float()
                target_k = metadata["mask"]
            elif chn == "top":
                output_k = output["p_is_above_top"]
                target_k = metadata["mask_top"]
            elif chn == "bottom":
                output_k = output["p_is_below_bottom"]
                target_k = metadata["mask_bot"]
            elif chn == "removedseg":
                output_k = output["p_is_removed"]
                target_k = metadata["is_removed"]
            elif chn == "passive":
                output_k = output["p_is_passive"]
                target_k = metadata["is_passive"]
            elif chn == "patch":
                output_k = output["p_is_patch"]
                target_k = metadata["mask_patches"]
            else:
                raise ValueError("Unrecognised output channel: {}".format(chn))

            for c, v in meters_k.items():
                c = c.lower()
                if c == "accuracy":
                    v.update(
                        100.0 * criterions.mask_accuracy(output_k, target_k).item(), ns
                    )
                elif c == "precision":
                    v.update(
                        100.0 * criterions.mask_precision(output_k, target_k).item(), ns
                    )
                elif c == "recall":
                    v.update(
                        100.0 * criterions.mask_recall(output_k, target_k).item(), ns
                    )
                elif c == "f1 score" or c == "f1":
                    v.update(criterions.mask_f1_score(output_k, target_k).item(), ns)
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
        if use_mixed_precision:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
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
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")

    meters = {}
    for chn in ["Overall", "Top", "Bottom", "RemovedSeg", "Passive", "Patch"]:
        meters[chn] = {}
        meters[chn]["Accuracy"] = AverageMeter("Accuracy (" + chn + ")", ":6.2f")
        meters[chn]["Precision"] = AverageMeter("Precision (" + chn + ")", ":6.2f")
        meters[chn]["Recall"] = AverageMeter("Recall (" + chn + ")", ":6.2f")
        meters[chn]["F1 Score"] = AverageMeter("F1 Score (" + chn + ")", ":6.4f")
        meters[chn]["Jaccard"] = AverageMeter("Jaccard (" + chn + ")", ":6.4f")
        meters[chn]["Active output"] = AverageMeter(
            "Active output (" + chn + ")", ":6.2f"
        )
        meters[chn]["Active target"] = AverageMeter(
            "Active target (" + chn + ")", ":6.2f"
        )

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
            metadata = {k: v.to(device, non_blocking=True) for k, v in metadata.items()}

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
            for chn, meters_k in meters.items():
                chn = chn.lower()
                if chn == "overall":
                    output_k = output["mask_keep_pixel"].float()
                    target_k = metadata["mask"]
                elif chn == "top":
                    output_k = output["p_is_above_top"]
                    target_k = metadata["mask_top"]
                elif chn == "bottom":
                    output_k = output["p_is_below_bottom"]
                    target_k = metadata["mask_bot"]
                elif chn == "removedseg":
                    output_k = output["p_is_removed"]
                    target_k = metadata["is_removed"]
                elif chn == "passive":
                    output_k = output["p_is_passive"]
                    target_k = metadata["is_passive"]
                elif chn == "patch":
                    output_k = output["p_is_patch"]
                    target_k = metadata["mask_patches"]
                else:
                    raise ValueError("Unrecognised output channel: {}".format(chn))

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
            echofilter.transforms.Normalize(DATA_CENTER, DATA_DEVIATION),
            echofilter.transforms.ReplaceNan(NAN_VALUE),
            echofilter.transforms.Rescale((data["signals"].shape[0], sample_shape[1])),
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

    # Crop long input
    for key in (
        echofilter.transforms._fields_2d + echofilter.transforms._fields_1d_timelike
    ):
        if key in transect:
            transect[key] = transect[key][:MAX_INPUT_LEN]

    # Apply depth crop
    if crop_depth is not None:
        depth_crop_mask = transect["depths"] <= crop_depth

        for key in echofilter.transforms._fields_2d:
            if key in transect:
                transect[key] = transect[key][:, depth_crop_mask]

        for key in echofilter.transforms._fields_1d_depthlike:
            if key in transect:
                transect[key] = transect[key][depth_crop_mask]

    # Convert lines to masks
    ddepths = np.broadcast_to(transect["depths"], transect["Sv"].shape)
    transect["mask_top"] = np.single(ddepths < np.expand_dims(transect["top"], -1))
    transect["mask_bot"] = np.single(ddepths > np.expand_dims(transect["bottom"], -1))
    # Add mask_patches to the data, for plotting
    transect["mask_patches"] = 1 - transect["mask"]
    transect["mask_patches"][transect["is_passive"] > 0.5] = 0
    transect["mask_patches"][transect["is_removed"] > 0.5] = 0
    transect["mask_patches"][transect["mask_top"] > 0.5] = 0
    transect["mask_patches"][transect["mask_bot"] > 0.5] = 0

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
    Generate an output for a sample transect, specified by the path to its
    sharded data.
    """
    # Load the data
    transect = echofilter.raw.shardloader.load_transect_segments_from_shards_abs(fname)
    # Process the transect
    return _generate_from_loaded(transect, *args, **kwargs)


def save_checkpoint(state, is_best, dirname=".", filename="checkpoint.pth.tar"):
    os.makedirs(dirname, exist_ok=True)
    torch.save(state, os.path.join(dirname, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(dirname, filename), os.path.join(dirname, "model_best.pth.tar")
        )


def meters_to_csv(meters, is_best, dirname=".", filename="meters.csv"):
    os.makedirs(dirname, exist_ok=True)
    df = pd.DataFrame()
    for chn in meters:
        # For each output plane
        for criterion_name, meter in meters[chn].items():
            # For each criterion
            df[meter.name] = meter.values
    df.to_csv(os.path.join(dirname, filename), index=False)
    if is_best:
        shutil.copyfile(
            os.path.join(dirname, filename),
            os.path.join(dirname, "model_best.meters.csv"),
        )


def main():
    import argparse

    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        prog = os.path.split(__file__)[1]
    parser = argparse.ArgumentParser(prog=prog, description="Echofilter training",)
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version="%(prog)s {version}".format(version=echofilter.__version__),
    )

    # Data parameters
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/data/dsforce/surveyExports",
        metavar="DIR",
        help="path to root data directory",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_name",
        type=str,
        default="mobile",
        help="which dataset to use",
    )
    parser.add_argument(
        "--shape",
        dest="sample_shape",
        nargs=2,
        type=int,
        default=(128, 512),
        help="input shape [W, H] (default: (128, 512))",
    )
    parser.add_argument(
        "--crop-depth",
        type=float,
        default=None,
        help="depth, in metres, at which data should be truncated (default: None)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--log",
        dest="log_name",
        default=None,
        type=str,
        help="output directory name (default: DATE_TIME)",
    )
    parser.add_argument(
        "--log-append",
        dest="log_name_append",
        default=None,
        type=str,
        help="string to append to output directory name (default: HOSTNAME)",
    )

    # Model parameters
    parser.add_argument(
        "--nblock",
        "--num-blocks",
        dest="n_block",
        type=int,
        default=6,
        help="number of blocks down and up in the UNet (default: 6)",
    )
    parser.add_argument(
        "--latent-channels",
        type=int,
        default=32,
        help="number of initial/final latent channels to use in the model (default: 32)",
    )
    parser.add_argument(
        "--expansion-factor",
        type=float,
        default=1.0,
        help="expansion for number of channels as model becomes deeper"
        " (default: 1., constant number of channels)",
    )
    parser.add_argument(
        "--expand-only-on-down",
        action="store_true",
        help="only expand channels on dowsampling blocks",
    )
    parser.add_argument(
        "--blocks-per-downsample",
        nargs="+",
        type=int,
        default=(2, 1),
        help="for each dim (time, depth), number of blocks between downsample"
        " steps (default: [2, 1])",
    )
    parser.add_argument(
        "--blocks-before-first-downsample",
        nargs="+",
        type=int,
        default=(2, 1),
        help="for each dim (time, depth), number of blocks before first"
        " downsample step (default: [2, 1])",
    )
    parser.add_argument(
        "--only-skip-connection-on-downsample",
        dest="always_include_skip_connection",
        action="store_false",
        help="only include skip connections when downsampling",
    )
    parser.add_argument(
        "--deepest-inner",
        type=str,
        default="horizontal_block",
        help="layer to include at the deepest point of the UNet"
        ' (default: "horizontal_block"). Set to "identity" to disable.',
    )
    parser.add_argument(
        "--intrablock-expansion",
        type=float,
        default=6.0,
        help="expansion within inverse residual blocks (default: 6.)",
    )
    parser.add_argument(
        "--se-reduction",
        "--se",
        dest="se_reduction",
        type=float,
        default=4.0,
        help="reduction within squeeze-and-excite blocks (default: 4.)",
    )
    parser.add_argument(
        "--downsampling-modes",
        nargs="+",
        type=str,
        default="max",
        help='for each downsampling step, the method to use (default: "max")',
    )
    parser.add_argument(
        "--upsampling-modes",
        nargs="+",
        type=str,
        default="bilinear",
        help='for each upsampling step, the method to use (default: "bilinear")',
    )
    parser.add_argument(
        "--fused-conv",
        dest="depthwise_separable_conv",
        action="store_false",
        help="use fused instead of depthwise separable convolutions",
    )
    parser.add_argument(
        "--no-residual",
        dest="residual",
        action="store_false",
        help="don't use residual blocks",
    )
    parser.add_argument(
        "--actfn", type=str, default="InplaceReLU", help="activation function to use",
    )
    parser.add_argument(
        "--kernel",
        dest="kernel_size",
        type=int,
        default=5,
        help="convolution kernel size (default: 5)",
    )

    # Training methodology parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='device to use (default: "cuda", using first gpu)',
    )
    parser.add_argument(
        "--no-amp",
        dest="use_mixed_precision",
        action="store_false",
        default=None,
        help="use fp32 instead of mixed precision (default: use mixed precision on gpu)",
    )
    parser.add_argument(
        "--amp-opt",
        type=str,
        default="O1",
        help='optimizer level for apex automatic mixed precision (default: "O1")',
    )
    parser.add_argument(
        "-j",
        "--workers",
        dest="n_worker",
        type=int,
        default=8,
        metavar="N",
        help="number of data loading workers (default: 8)",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        type=int,
        default=50,
        help="print frequency (default: 50)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=16,
        help="mini-batch size (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        dest="n_epoch",
        type=int,
        default=20,
        help="number of total epochs to run (default: 20)",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training.",
    )

    # Optimiser parameters
    parser.add_argument(
        "--optim",
        "--optimiser",
        "--optimizer",
        dest="optimizer",
        type=str,
        default="rangerva",
        help='optimizer name (default: "rangerva")',
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="constant",
        help='LR schedule (default: "constant")',
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        dest="lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="initial learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="momentum (default: 0.9)",
    )
    parser.add_argument(
        "--base-momentum",
        type=float,
        default=None,
        help="base momentum; only used for OneCycle schedule (default: same as momentum)",
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=1e-5,
        help="weight decay (default: 1e-5)",
    )
    parser.add_argument(
        "--warmup-pct",
        type=float,
        default=0.2,
        help="fraction of training to spend warming up LR; only used for"
        " OneCycle MesaOneCycle schedules (default: 0.2)",
    )
    parser.add_argument(
        "--warmdown-pct",
        type=float,
        default=0.7,
        help="fraction of training before warming down LR; only used for"
        " MesaOneCycle schedule (default: 0.7)",
    )
    parser.add_argument(
        "--anneal-strategy",
        type=str,
        default="cos",
        help='annealing strategy; only used for OneCycle schedule (default: "cos")',
    )
    parser.add_argument(
        "--overall-loss-weight",
        type=float,
        default=0.0,
        help="weighting for overall loss term (default: 0.)",
    )

    # Use seaborn to set matplotlib plotting defaults
    import seaborn as sns

    sns.set()

    kwargs = vars(parser.parse_args())

    for k in ["blocks_per_downsample", "blocks_before_first_downsample"]:
        if len(kwargs[k]) == 1:
            kwargs[k] = kwargs[k][0]

    print("CLI arguments:")
    print(kwargs)
    print()

    train(**kwargs)


if __name__ == "__main__":
    main()
