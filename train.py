#!/usr/bin/env python

from collections import OrderedDict
import copy
import os
import shutil
import datetime
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms
from torchutils.random import seed_all
from torchutils.utils import count_parameters
import ranger

import echofilter.dataset
import echofilter.transforms
import echofilter.shardloader
from echofilter import criterions
from echofilter.meters import AverageMeter, ProgressMeter
from echofilter.raw.loader import get_partition_list
from echofilter.raw.manipulate import load_decomposed_transect_mask
from echofilter.unet import UNet
from echofilter.wrapper import Echofilter, EchofilterLoss
from echofilter.plotting import plot_transect_predictions


## For mobile dataset,
# DATA_MEAN = -81.5
# DATA_STDEV = 21.9

## For stationary dataset,
# DATA_MEAN = -78.7
# DATA_STDEV = 19.2

# Overall values to use
DATA_MEAN = -80.
DATA_STDEV = 20.

# Transects to plot for debugging
PLOT_TRANSECTS = {
    'mobile': [
        'mobile/Survey07/Survey07_GR4_N5W_survey7',
        'mobile/Survey14/Survey14_GR4_N0W_E',
        'mobile/Survey16/Survey16_GR4_N5W_E',
        'mobile/Survey17/Survey17_GR4_N5W_E',
    ],
    'stationary': [
        'stationary/december2017/evExports/december2017_D20180213-T115216_D20180213-T172216',
        'stationary/march2018/evExports/march2018_D20180513-T195216_D20180514-T012216',
        'stationary/september2018/evExports/september2018_D20181027-T202217_D20181028-T015217',
        'stationary/september2018/evExports/september2018_D20181107-T122220_D20181107-T175217',
    ]
}


def main(
        data_dir='/data/dsforce/surveyExports',
        dataset_name='mobile',
        sample_shape=(128, 512),
        crop_depth=70,
        resume='',
        log_name=None,
        log_name_append=None,
        n_steps=4,
        latent_channels=64,
        expansion_factor=2,
        down_pool='max',
        device='cuda',
        n_worker=4,
        batch_size=64,
        n_epoch=10,
        seed=None,
        print_freq=10,
        optimizer='adamw',
        schedule='constant',
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-5,
    ):

    seed_all(seed)

    schedule = schedule.lower()

    if log_name is None or log_name == '':
        log_name = datetime.datetime.now().strftime('%Y-%b-%d_%H:%M:%S')
    if log_name_append is None:
        log_name_append = os.uname()[1]
    if len(log_name_append) > 0:
        log_name += '_' + log_name_append

    print('Output will be written to {}/{}'.format(dataset_name, log_name))

    # Augmentations
    train_transform_pre = torchvision.transforms.Compose([
        echofilter.transforms.RandomCropWidth(0.5),
        echofilter.transforms.RandomStretchDepth(0.5),
        echofilter.transforms.RandomReflection(),
        echofilter.transforms.RandomCropTop(0.75),
    ])
    train_transform_post = torchvision.transforms.Compose([
        echofilter.transforms.Normalize(DATA_MEAN, DATA_STDEV),
        echofilter.transforms.ColorJitter(0.5, 0.3),
        echofilter.transforms.ReplaceNan(-3),
        echofilter.transforms.Rescale(sample_shape),
    ])
    val_transform = torchvision.transforms.Compose([
        echofilter.transforms.Normalize(DATA_MEAN, DATA_STDEV),
        echofilter.transforms.ReplaceNan(-3),
        echofilter.transforms.Rescale(sample_shape),
    ])

    train_paths = get_partition_list(
        'train',
        dataset=dataset_name,
        partitioning_version='firstpass',
        root_data_dir=data_dir,
        full_path=True,
        sharded=True,
    )
    val_paths = get_partition_list(
        'validate',
        dataset=dataset_name,
        partitioning_version='firstpass',
        root_data_dir=data_dir,
        full_path=True,
        sharded=True,
    )
    print('Found {:3d} train sample paths'.format(len(train_paths)))
    print('Found {:3d} val sample paths'.format(len(val_paths)))

    dataset_train = echofilter.dataset.TransectDataset(
        train_paths,
        window_len=int(1.5 * sample_shape[0]),
        crop_depth=crop_depth,
        num_windows_per_transect=None,
        use_dynamic_offsets=True,
        transform_pre=train_transform_pre,
        transform_post=train_transform_post,
    )
    dataset_val = echofilter.dataset.TransectDataset(
        val_paths,
        window_len=sample_shape[0],
        crop_depth=crop_depth,
        num_windows_per_transect=None,
        use_dynamic_offsets=False,
        transform_post=val_transform,
    )
    dataset_augval = echofilter.dataset.TransectDataset(
        val_paths,
        window_len=sample_shape[0],
        crop_depth=crop_depth,
        num_windows_per_transect=None,
        use_dynamic_offsets=False,
        transform_pre=train_transform_pre,
        transform_post=train_transform_post,
    )
    print('Train dataset has {:4d} samples'.format(len(dataset_train)))
    print('Val   dataset has {:4d} samples'.format(len(dataset_val)))

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_worker,
        pin_memory=True,
        drop_last=True,
    )
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_worker,
        pin_memory=True,
        drop_last=False,
    )
    loader_augval = torch.utils.data.DataLoader(
        dataset_augval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_worker,
        pin_memory=True,
        drop_last=False,
    )
    print('Train loader has {:3d} batches'.format(len(loader_train)))
    print('Val   loader has {:3d} batches'.format(len(loader_val)))

    print()
    print(
        'Constructing U-Net model with '
        '{} steps, '
        'initial latent channels {}, '
        'expansion_factor {}'
        .format(n_steps, latent_channels, expansion_factor)
    )
    model = Echofilter(
        UNet(
            1,
            5,
            n_steps=n_steps,
            latent_channels=latent_channels,
            expansion_factor=expansion_factor,
            down_pool=down_pool,
        ),
        top='boundary',
        bottom='boundary',
    )
    model.to(device)
    print(
        'Built model with {} trainable parameters'
        .format(count_parameters(model, only_trainable=True))
    )

    # define loss function (criterion) and optimizer
    criterion = EchofilterLoss()

    optimizer_name = optimizer.lower()
    if optimizer_name == 'adam':
        optimizer_class = torch.optim.Adam
    elif optimizer_name == 'adamw':
        optimizer_class = torch.optim.AdamW
    elif optimizer_name == 'ranger':
        optimizer_class = ranger.Ranger
    elif optimizer_name == 'rangerva':
        optimizer_class = ranger.RangerVA
    elif optimizer_name == 'rangerqh':
        optimizer_class = ranger.RangerQH
    else:
        # We don't support arbitrary optimizers from torch.optim because they
        # need different configuration parameters to Adam.
        raise ValueError('Unrecognised optimizer: {}'.format(optimizer))

    optimizer = optimizer_class(
        model.parameters(),
        lr,
        betas=(momentum, 0.999),
        weight_decay=weight_decay,
    )
    schedule_data = {'name': schedule}

    if schedule == 'lrfinder':
        from torch_lr_finder import LRFinder

        print('Running learning rate finder')
        lr_finder = LRFinder(model, optimizer, criterion, device=device)
        lr_finder.range_test(loader_train, end_lr=100, num_iter=100, diverge_th=3)
        print('Plotting learning rate finder results')
        hf = plt.figure(figsize=(15, 9))
        ax = plt.axes()
        lr_finder.plot(skip_start=0, skip_end=1, log_lr=True, ax=ax)
        plt.tick_params(reset=True, color=(.2, .2, .2))
        plt.tick_params(labelsize=14)
        ax.minorticks_on()
        ax.tick_params(direction='out')
        # Save figure
        figpth = os.path.join('models', dataset_name, log_name, 'lrfinder.png')
        os.makedirs(os.path.dirname(figpth), exist_ok=True)
        plt.savefig(figpth)
        print('LR Finder results saved to {}'.format(figpth))
        return

    # Make a tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join('runs', dataset_name, log_name))

    # Initialise loop tracking
    start_epoch = 1
    best_loss_val = float('inf')

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
        start_epoch = checkpoint['epoch'] + 1
        best_loss_val = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(
            "Loaded checkpoint '{}' (epoch {})"
            .format(resume, checkpoint['epoch'])
        )

    print('Starting training')
    t_start = time.time()
    for epoch in range(start_epoch, n_epoch + 1):

        t_epoch_start = time.time()

        # Resample offsets for each window
        loader_train.dataset.initialise_datapoints()

        # train for one epoch
        loss_tr, meters_tr, (ex_input_tr, ex_data_tr, ex_output_tr) = train(
            loader_train, model, criterion, optimizer, device, epoch, print_freq=print_freq,
            schedule_data=schedule_data,
        )

        # evaluate on validation set
        loss_val, meters_val, (ex_input_val, ex_data_val, ex_output_val) = validate(
            loader_val, model, criterion, device, print_freq=print_freq, prefix='Validation'
        )
        # evaluate on augmented validation set
        loss_augval, meters_augval, (ex_input_augval, ex_data_augval, ex_output_augval) = validate(
            loader_augval, model, criterion, device, print_freq=print_freq, prefix='Aug-Val   '
        )
        print(
            'Completed {} epochs in {}'
            .format(epoch, datetime.timedelta(seconds=time.time() - t_start))
        )
        # Print metrics to terminal
        name_fmt = '{:.<28s}'
        current_lr = get_current_lr(optimizer)
        print((name_fmt + ' : {:.4e}').format('Learning rate', current_lr))
        print(
            (name_fmt + ' Train: {:.4e}  AugVal: {:.4e}  Val: {:.4e}')
            .format('Loss', loss_tr, loss_augval, loss_val)
        )
        for chn in meters_tr:
            # For each output plane
            print(chn)
            for cr in meters_tr[chn]:
                # For each criterion
                fmt_str = name_fmt
                fmt_str += ' Train: {' + meters_tr[chn][cr].fmt + '}'
                fmt_str += '    AugVal: {' + meters_augval[chn][cr].fmt + '}'
                fmt_str += '    Val: {' + meters_val[chn][cr].fmt + '}'
                print(
                    fmt_str.format(
                        meters_tr[chn][cr].name,
                        meters_tr[chn][cr].avg,
                        meters_augval[chn][cr].avg,
                        meters_val[chn][cr].avg,
                    )
                )

        # Add hyper parameters to tensorboard
        writer.add_scalar('LR', current_lr, epoch)
        # Add metrics to tensorboard
        for loss_p, partition in ((loss_tr, 'Train'), (loss_val, 'Val'), (loss_augval, 'ValAug')):
            writer.add_scalar('{}/{}'.format('Loss', partition), loss_p, epoch)
        for chn in meters_tr:
            # For each output plane
            for cr in meters_tr[chn]:
                # For each criterion
                writer.add_scalar('{}/{}/{}'.format(cr, chn, 'Train'), meters_tr[chn][cr].avg, epoch)
                writer.add_scalar('{}/{}/{}'.format(cr, chn, 'ValAug'), meters_augval[chn][cr].avg, epoch)
                writer.add_scalar('{}/{}/{}'.format(cr, chn, 'Val'), meters_val[chn][cr].avg, epoch)

        def ensure_clim_met(x, x0=0., x1=1.):
            x = x.clone()
            x[0, :, 0, 0] = 0
            x[0, :, 0, 1] = 1
            return x

        def add_image_border(x):
            '''
            Add a green border around a a tensor of images.

            Parameters
            ----------
            x : torch.Tensor
                Tensor in NCWH or NCHW format.

            Returns
            -------
            torch.Tensor
                As `x`, but padded with a green border.
            '''
            if x.shape[1] == 1:
                x = torch.cat([x, x, x], dim=1)
            if x.shape[1] != 3:
                raise ValueError('RGB image needs three color channels')
            shp = list(x.shape)
            shp[-1] = 1
            x = torch.cat([
                torch.zeros(shp, dtype=x.dtype, device=x.device),
                x,
                torch.zeros(shp, dtype=x.dtype, device=x.device),
            ], dim=-1)
            shp = list(x.shape)
            shp[-2] = 1
            x = torch.cat([
                torch.zeros(shp, dtype=x.dtype, device=x.device),
                x,
                torch.zeros(shp, dtype=x.dtype, device=x.device),
            ], dim=-2)
            x[:, 1, :, 0] = 1.
            x[:, 1, :, -1] = 1.
            x[:, 1, 0, :] = 1.
            x[:, 1, -1, :] = 1.
            return x

        # Add example images to tensorboard
        for (ex_input, ex_data, ex_output), partition in (
                ((ex_input_tr, ex_data_tr, ex_output_tr), 'Train'),
                ((ex_input_val, ex_data_val, ex_output_val), 'Val'),
                ((ex_input_augval, ex_data_augval, ex_output_augval), 'ValAug'),
            ):
            writer.add_images(
                'Input/' + partition,
                ex_input,
                epoch,
                dataformats='NCWH',
            )
            writer.add_images(
                'Top/' + partition + '/Target',
                ensure_clim_met(add_image_border(ex_data['mask_top'].unsqueeze(1))),
                epoch,
                dataformats='NCWH',
            )
            writer.add_images(
                'Top/' + partition + '/Output/p',
                ensure_clim_met(add_image_border(ex_output['p_is_above_top'].unsqueeze(1))),
                epoch,
                dataformats='NCWH',
            )
            writer.add_images(
                'Bottom/' + partition + '/Target',
                ensure_clim_met(add_image_border(ex_data['mask_bot'].unsqueeze(1))),
                epoch,
                dataformats='NCWH',
            )
            writer.add_images(
                'Bottom/' + partition + '/Output/p',
                ensure_clim_met(add_image_border(ex_output['p_is_below_bottom'].unsqueeze(1))),
                epoch,
                dataformats='NCWH',
            )
            writer.add_images(
                'Overall/' + partition + '/Target',
                ensure_clim_met(add_image_border(ex_data['mask'].float().unsqueeze(1))),
                epoch,
                dataformats='NCWH',
            )
            writer.add_images(
                'Overall/' + partition + '/Output/p',
                ensure_clim_met(add_image_border(ex_output['p_keep_pixel'].unsqueeze(1))),
                epoch,
                dataformats='NCWH',
            )
            writer.add_images(
                'Overall/' + partition + '/Output/mask',
                ensure_clim_met(add_image_border(ex_output['mask_keep_pixel'].float().unsqueeze(1))),
                epoch,
                dataformats='NCWH',
            )
            writer.add_images(
                'Overall/' + partition + '/Overlap',
                ensure_clim_met(add_image_border(
                    torch.stack([
                        ex_output['mask_keep_pixel'].float(),
                        torch.zeros_like(ex_output['mask_keep_pixel'], dtype=torch.float),
                        ex_data['mask'].float(),
                    ], dim=1)
                )),
                epoch,
                dataformats='NCWH',
            )

        for k, plot_transects_k in PLOT_TRANSECTS.items():
            if k not in dataset_name:
                continue
            for transect_name in plot_transects_k:
                transect, prediction = generate_from_shards(
                    model,
                    os.path.join(data_dir + '_sharded', transect_name),
                    sample_shape=sample_shape,
                    crop_depth=crop_depth,
                    device=device,
                    dtype=torch.float,
                )
                hf = plt.figure(figsize=(15, 9))
                plot_transect_predictions(transect, prediction, cmap='viridis', linewidth=1)
                transect_name = transect_name.replace('/evExports', '')
                if epoch == n_epoch:
                    # Only save png if this is the final epoch
                    figpth = os.path.join('models', dataset_name, log_name, 'samples', transect_name + '_output.png')
                    os.makedirs(os.path.dirname(figpth), exist_ok=True)
                    plt.savefig(figpth)
                writer.add_figure(transect_name, hf, epoch, close=True)

        # remember best loss and save checkpoint
        is_best = loss_val < best_loss_val
        best_loss_val = min(loss_val, best_loss_val)

        save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': best_loss_val,
                'optimizer': optimizer.state_dict(),
                'meters': meters_val,
            },
            is_best,
            dirname=os.path.join('models', dataset_name, log_name),
        )
        meters_to_csv(meters_val, is_best, dirname=os.path.join('models', dataset_name, log_name))

        # Ensure the tensorboard outputs for this epoch are flushed
        writer.flush()

    # Close tensorboard connection
    writer.close()


def get_current_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def train(loader, model, criterion, optimizer, device, epoch, dtype=torch.float, print_freq=10, schedule_data=None):
    if schedule_data is None:
        schedule_data = {'name': 'constant'}

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    meters = {}
    for chn in ['Overall', 'Top', 'Bottom', 'RemovedSeg', 'Passive', 'Patch']:
        meters[chn] = {}
        meters[chn]['Accuracy'] = AverageMeter('Accuracy (' + chn + ')', ':6.2f')
        meters[chn]['Precision'] = AverageMeter('Precision (' + chn + ')', ':6.2f')
        meters[chn]['Recall'] = AverageMeter('Recall (' + chn + ')', ':6.2f')
        meters[chn]['F1 Score'] = AverageMeter('F1 Score (' + chn + ')', ':6.4f')
        meters[chn]['Jaccard'] = AverageMeter('Jaccard (' + chn + ')', ':6.4f')
        meters[chn]['Active output'] = AverageMeter('Active output (' + chn + ')', ':6.2f')
        meters[chn]['Active target'] = AverageMeter('Active target (' + chn + ')', ':6.2f')

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses, meters['Overall']['Accuracy'], meters['Overall']['Jaccard']],
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
        metadata = {k: v.to(device, dtype, non_blocking=True) for k, v in metadata.items()}

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
            if chn == 'overall':
                output_k = output['mask_keep_pixel'].float()
                target_k = metadata['mask']
            elif chn == 'top':
                output_k = output['p_is_above_top']
                target_k = metadata['mask_top']
            elif chn == 'bottom':
                output_k = output['p_is_below_bottom']
                target_k = metadata['mask_bot']
            elif chn == 'removedseg':
                output_k = output['p_is_removed']
                target_k = metadata['is_removed']
            elif chn == 'passive':
                output_k = output['p_is_passive']
                target_k = metadata['is_passive']
            elif chn == 'patch':
                output_k = output['p_is_patch']
                target_k = metadata['mask_patches']
            else:
                raise ValueError('Unrecognised output channel: {}'.format(chn))

            for c, v in meters_k.items():
                c = c.lower()
                if c == 'accuracy':
                    v.update(100. * criterions.mask_accuracy(output_k, target_k).item(), ns)
                elif c == 'precision':
                    v.update(100. * criterions.mask_precision(output_k, target_k).item(), ns)
                elif c == 'recall':
                    v.update(100. * criterions.mask_recall(output_k, target_k).item(), ns)
                elif c == 'f1 score' or c == 'f1':
                    v.update(criterions.mask_f1_score(output_k, target_k).item(), ns)
                elif c == 'jaccard':
                    v.update(criterions.mask_jaccard_index(output_k, target_k).item(), ns)
                elif c == 'active output':
                    v.update(100. * criterions.mask_active_fraction(output_k).item(), ns)
                elif c == 'active target':
                    v.update(100. * criterions.mask_active_fraction(target_k).item(), ns)
                else:
                    raise ValueError('Unrecognised criterion: {}'.format(c))

        # compute gradient and do optimizer update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 or i + 1 == len(loader):
            progress.display(i + 1)

    return losses.avg, meters, (example_input, example_data, example_output)


def validate(loader, model, criterion, device, dtype=torch.float, print_freq=10,
             prefix='Test', num_examples=32):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    meters = {}
    for chn in ['Overall', 'Top', 'Bottom', 'RemovedSeg', 'Passive', 'Patch']:
        meters[chn] = {}
        meters[chn]['Accuracy'] = AverageMeter('Accuracy (' + chn + ')', ':6.2f')
        meters[chn]['Precision'] = AverageMeter('Precision (' + chn + ')', ':6.2f')
        meters[chn]['Recall'] = AverageMeter('Recall (' + chn + ')', ':6.2f')
        meters[chn]['F1 Score'] = AverageMeter('F1 Score (' + chn + ')', ':6.4f')
        meters[chn]['Jaccard'] = AverageMeter('Jaccard (' + chn + ')', ':6.4f')
        meters[chn]['Active output'] = AverageMeter('Active output (' + chn + ')', ':6.2f')
        meters[chn]['Active target'] = AverageMeter('Active target (' + chn + ')', ':6.2f')

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses, meters['Overall']['Accuracy'], meters['Overall']['Jaccard']],
        prefix=prefix + ': ',
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
            metadata = {k: v.to(device, dtype, non_blocking=True) for k, v in metadata.items()}

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
                if chn == 'overall':
                    output_k = output['mask_keep_pixel'].float()
                    target_k = metadata['mask']
                elif chn == 'top':
                    output_k = output['p_is_above_top']
                    target_k = metadata['mask_top']
                elif chn == 'bottom':
                    output_k = output['p_is_below_bottom']
                    target_k = metadata['mask_bot']
                elif chn == 'removedseg':
                    output_k = output['p_is_removed']
                    target_k = metadata['is_removed']
                elif chn == 'passive':
                    output_k = output['p_is_passive']
                    target_k = metadata['is_passive']
                elif chn == 'patch':
                    output_k = output['p_is_patch']
                    target_k = metadata['mask_patches']
                else:
                    raise ValueError('Unrecognised output channel: {}'.format(chn))

                for c, v in meters_k.items():
                    c = c.lower()
                    if c == 'accuracy':
                        v.update(100. * criterions.mask_accuracy(output_k, target_k).item(), ns)
                    elif c == 'precision':
                        v.update(100. * criterions.mask_precision(output_k, target_k).item(), ns)
                    elif c == 'recall':
                        v.update(100. * criterions.mask_recall(output_k, target_k).item(), ns)
                    elif c == 'f1 score' or c == 'f1':
                        v.update(criterions.mask_f1_score(output_k, target_k).item(), ns)
                    elif c == 'jaccard':
                        v.update(criterions.mask_jaccard_index(output_k, target_k).item(), ns)
                    elif c == 'active output':
                        v.update(100. * criterions.mask_active_fraction(output_k).item(), ns)
                    elif c == 'active target':
                        v.update(100. * criterions.mask_active_fraction(target_k).item(), ns)
                    else:
                        raise ValueError('Unrecognised criterion: {}'.format(c))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0 or i + 1 == len(loader):
                progress.display(i + 1)

    # Restack samples, converting list into higher-dim tensor
    example_input = torch.stack(example_input, dim=0)
    example_data = {k: torch.stack([a[k] for a in example_data], dim=0) for k in example_data[0]}
    example_output = {k: torch.stack([a[k] for a in example_output], dim=0) for k in example_output[0]}

    return losses.avg, meters, (example_input, example_data, example_output)


def generate_from_transect(model, transect, sample_shape, crop_depth, device, dtype=torch.float):
    '''
    Generate an output for a sample transect, .
    '''

    # Put model in evaluation mode
    model.eval()

    # Make a copy of the transect which we will use to
    data = copy.deepcopy(transect)

    # Apply depth crop
    depth_crop_mask = data['depths'] <= crop_depth
    data['depths'] = data['depths'][depth_crop_mask]
    data['signals'] = data['signals'][:, depth_crop_mask]

    # Configure data to match what the model expects to see
    # Ensure depth is always increasing (which corresponds to descending from
    # the air down the water column)
    if (data['depths'][-1] < data['depths'][0]):
        # Found some upward-facing data that still needs to be reflected
        for k in ['depths', 'signals', 'mask']:
            data[k] = np.flip(data[k], -1).copy()

    # Apply transforms
    transform = torchvision.transforms.Compose([
        echofilter.transforms.Normalize(DATA_MEAN, DATA_STDEV),
        echofilter.transforms.ReplaceNan(-3),
        echofilter.transforms.Rescale((data['signals'].shape[0], sample_shape[1])),
    ])
    data = transform(data)
    input = torch.tensor(data['signals']).unsqueeze(0).unsqueeze(0)
    input = input.to(device, dtype)
    # Put data through model
    with torch.no_grad():
        output = model(input)
        output = {k: v.squeeze(0).cpu().numpy() for k, v in output.items()}

    output['depths'] = data['depths']
    output['timestamps'] = data['timestamps']

    return output


def generate_from_file(model, fname, *args, **kwargs):
    '''
    Generate an output for a sample transect, specified by its file path.
    '''
    # Load the data
    transect = load_decomposed_transect_mask(fname)
    # Convert lines to masks
    ddepths = np.broadcast_to(transect['depths'], transect['Sv'].shape)
    transect['mask_top'] = np.single(
        ddepths < np.expand_dims(transect['top'], -1)
    )
    transect['mask_bot'] = np.single(
        ddepths > np.expand_dims(transect['bottom'], -1)
    )
    # Add mask_patches to the data, for plotting
    transect['mask_patches'] = 1 - transect['mask']
    transect['mask_patches'][transect['is_passive'] > 0.5] = 0
    transect['mask_patches'][transect['is_removed'] > 0.5] = 0
    transect['mask_patches'][transect['mask_top'] > 0.5] = 0
    transect['mask_patches'][transect['mask_bot'] > 0.5] = 0

    # Generate predictions for the transect
    transect['signals'] = transect.pop('Sv')
    prediction = generate_from_transect(model, transect, *args, **kwargs)
    transect['Sv'] = transect.pop('signals')

    return transect, prediction


def generate_from_shards(model, fname, *args, **kwargs):
    '''
    Generate an output for a sample transect, specified by the path to its
    sharded data.
    '''
    # Load the data
    transect = echofilter.shardloader.load_transect_segments_from_shards_abs(fname)

    # Convert lines to masks
    ddepths = np.broadcast_to(transect['depths'], transect['Sv'].shape)
    transect['mask_top'] = np.single(
        ddepths < np.expand_dims(transect['top'], -1)
    )
    transect['mask_bot'] = np.single(
        ddepths > np.expand_dims(transect['bottom'], -1)
    )
    # Add mask_patches to the data, for plotting
    transect['mask_patches'] = 1 - transect['mask']
    transect['mask_patches'][transect['is_passive'] > 0.5] = 0
    transect['mask_patches'][transect['is_removed'] > 0.5] = 0
    transect['mask_patches'][transect['mask_top'] > 0.5] = 0
    transect['mask_patches'][transect['mask_bot'] > 0.5] = 0

    # Generate predictions for the transect
    transect['signals'] = transect.pop('Sv')
    prediction = generate_from_transect(model, transect, *args, **kwargs)
    transect['Sv'] = transect.pop('signals')

    return transect, prediction


def save_checkpoint(state, is_best, dirname='.', filename='checkpoint.pth.tar'):
    os.makedirs(dirname, exist_ok=True)
    torch.save(state, os.path.join(dirname, filename))
    if is_best:
        shutil.copyfile(os.path.join(dirname, filename), os.path.join(dirname, 'model_best.pth.tar'))


def meters_to_csv(meters, is_best, dirname='.', filename='meters.csv'):
    os.makedirs(dirname, exist_ok=True)
    df = pd.DataFrame()
    for chn in meters:
        # For each output plane
        for criterion_name, meter in meters[chn].items():
            # For each criterion
            df[meter.name] = meter.values
    df.to_csv(os.path.join(dirname, filename), index=False)
    if is_best:
        shutil.copyfile(os.path.join(dirname, filename), os.path.join(dirname, 'model_best.meters.csv'))


if __name__ == '__main__':

    import argparse

    # Data parameters
    parser = argparse.ArgumentParser(description='Echofilter training')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/data/dsforce/surveyExports',
        metavar='DIR',
        help='path to root data directory',
    )
    parser.add_argument(
        '--dataset',
        dest='dataset_name',
        type=str,
        default='mobile',
        help='which dataset to use',
    )
    parser.add_argument(
        '--crop-depth',
        type=float,
        default=70,
        help='depth, in metres, at which data should be truncated (default: 70)',
    )
    parser.add_argument(
        '--resume',
        default='',
        type=str,
        metavar='PATH',
        help='path to latest checkpoint (default: none)',
    )
    parser.add_argument(
        '--log',
        dest='log_name',
        default=None,
        type=str,
        help='output directory name (default: DATE_TIME)',
    )
    parser.add_argument(
        '--log-append',
        dest='log_name_append',
        default=None,
        type=str,
        help='string to append to output directory name (default: HOSTNAME)',
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
    parser.add_argument(
        '--down-pool',
        type=str,
        default='max',
        help='pooling mode for downsampling within unet (default: "max")',
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
        '-p', '--print-freq',
        type=int,
        default=10,
        help='print frequency (default: 10)',
    )
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=64,
        help='mini-batch size (default: 64)',
    )
    parser.add_argument(
        '--epochs',
        dest='n_epoch',
        type=int,
        default=20,
        help='number of total epochs to run',
    )
    parser.add_argument(
        '--seed',
        default=None,
        type=int,
        help='seed for initializing training.',
    )

    # Optimiser parameters
    parser.add_argument(
        '--optim', '--optimiser', '--optimizer',
        dest='optimizer',
        type=str,
        default='adamw',
        help='optimizer name (default: "adamw")',
    )
    parser.add_argument(
        '--schedule',
        type=str,
        default='constant',
        help='LR schedule (default: "constant")',
    )
    parser.add_argument(
        '--lr', '--learning-rate',
        dest='lr',
        type=float,
        default=0.1,
        metavar='LR',
        help='initial learning rate',
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='momentum',
    )
    parser.add_argument(
        '--wd', '--weight-decay',
        dest='weight_decay',
        type=float,
        default=1e-4,
        help='weight decay (default: 1e-4)',
    )

    # Use seaborn to set matplotlib plotting defaults
    import seaborn as sns
    sns.set()

    main(**vars(parser.parse_args()))
