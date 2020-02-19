#!/usr/bin/env python

from collections import OrderedDict
import os
import shutil
import datetime
import time

import pandas as pd
import torch
import torch.nn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms
from torchutils.random import seed_all
from torchutils.utils import count_parameters

import echofilter.dataset
import echofilter.transforms
from echofilter import criterions
from echofilter.meters import AverageMeter, ProgressMeter
from echofilter.raw.loader import get_partition_list
from echofilter.unet import UNet


## For mobile dataset,
# DATA_MEAN = -81.5
# DATA_STDEV = 21.9

## For stationary dataset,
# DATA_MEAN = -78.7
# DATA_STDEV = 19.2

# Overall values to use
DATA_MEAN = -80.
DATA_STDEV = 20.


def main(
        data_dir='/data/dsforce/surveyExports',
        dataset_name='mobile',
        sample_shape=(128, 512),
        crop_depth=70,
        resume='',
        latent_channels=64,
        expansion_factor=2,
        device='cuda',
        n_worker=4,
        batch_size=64,
        n_epoch=10,
        seed=None,
        print_freq=10,
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-5,
    ):

    seed_all(seed)

    # Make a tensorboard writer
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    # Augmentations
    train_transform_pre = torchvision.transforms.Compose([
        echofilter.transforms.RandomCropWidth(0.5),
        echofilter.transforms.RandomStretchDepth(0.5),
        echofilter.transforms.RandomReflection(),
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
        'initial latent channels {}, '
        'expansion_factor {}'
        .format(latent_channels, expansion_factor)
    )
    model = UNet(1, 2, latent_channels=latent_channels, expansion_factor=expansion_factor)
    model.to(device)
    print(
        'Built model with {} trainable parameters'
        .format(count_parameters(model, only_trainable=True))
    )

    # define loss function (criterion) and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr,
        betas=(momentum, 0.999),
        weight_decay=weight_decay,
    )

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

    # Add graph to tensorboard
    batch = next(iter(loader_train))
    data = batch['signals'].unsqueeze(1)
    data = data.to(device, torch.float, non_blocking=True)
    writer.add_graph(model, data)

    print('Starting training')
    t_start = time.time()
    for epoch in range(start_epoch, n_epoch + 1):

        t_epoch_start = time.time()

        # Resample offsets for each window
        loader_train.dataset.initialise_datapoints()

        # train for one epoch
        loss_tr, meters_tr, (ex_data_tr, ex_target_tr, ex_output_tr) = train(
            loader_train, model, criterion, optimizer, device, epoch, print_freq=print_freq
        )

        # evaluate on validation set
        loss_val, meters_val, (ex_data_val, ex_target_val, ex_output_val) = validate(
            loader_val, model, criterion, device, print_freq=print_freq, prefix='Validation'
        )
        # evaluate on augmented validation set
        loss_augval, meters_augval, (ex_data_augval, ex_target_augval, ex_output_augval) = validate(
            loader_augval, model, criterion, device, print_freq=print_freq, prefix='Aug-Val   '
        )
        print(
            'Completed {} epochs in {}'
            .format(epoch, datetime.timedelta(seconds=time.time() - t_start))
        )
        name_fmt = '{:.<23s}'
        print(
            (name_fmt + ' Train: {:.4e}  AugVal: {:.4e}  Val: {:.4e}')
            .format('Loss', loss_tr, loss_augval, loss_val)
        )
        for meter_tr, meter_val, meter_augval in zip(meters_tr, meters_val, meters_augval):
            if meter_tr.name != meter_val.name:
                fmt_str = name_fmt + ' Train: {' + meter_tr.fmt + '}'
                print(fmt_str.format(meter_tr.name, meter_tr.avg))
                fmt_str = name_fmt + ' AugVal: {' + meter_augval.fmt + '}'
                print(fmt_str.format(meter_augval.name, meter_augval.avg))
                fmt_str = name_fmt + ' Val: {' + meter_val.fmt + '}'
                print(fmt_str.format(meter_val.name, meter_val.avg))
            else:
                fmt_str = name_fmt
                fmt_str += ' Train: {' + meter_tr.fmt + '}'
                fmt_str += '    AugVal: {' + meter_augval.fmt + '}'
                fmt_str += '    Val: {' + meter_val.fmt + '}'
                print(fmt_str.format(meter_tr.name, meter_tr.avg, meter_augval.avg, meter_val.avg))

        # Add metrics to tensorboard
        for loss_p, partition in ((loss_tr, 'Train'), (loss_val, 'Val'), (loss_augval, 'ValAug')):
            writer.add_scalar('{}/{}'.format('Loss', partition), loss_p, epoch)
        for meters, partition in ((meters_tr, 'Train'), (meters_val, 'Val'), (meters_augval, 'ValAug')):
            for meter in meters:
                name = meter.name
                if '(top)' in name:
                    name = name.replace('(top)', '').strip()
                    name += '/Top'
                elif '(bottom)' in name:
                    name = name.replace('(bottom)', '').strip()
                    name += '/Bottom'
                else:
                    name += '/Overall'
                writer.add_scalar('{}/{}'.format(name, partition), meter.avg, epoch)

        # Add example images to tensorboard
        for (ex_data, ex_target, ex_output), partition in (
                ((ex_data_tr, ex_target_tr, ex_output_tr), 'Train'),
                ((ex_data_val, ex_target_val, ex_output_val), 'Val'),
                ((ex_data_augval, ex_target_augval, ex_output_augval), 'ValAug'),
            ):
            writer.add_images(
                'Input/' + partition,
                ex_data,
                epoch,
                dataformats='NCWH',
            )
            writer.add_images(
                'Top/' + partition + '/Target',
                ex_target[:, :1],
                epoch,
                dataformats='NCWH',
            )
            writer.add_images(
                'Bottom/' + partition + '/Target',
                ex_target[:, 1:],
                epoch,
                dataformats='NCWH',
            )
            writer.add_images(
                'Top/' + partition + '/Output/Logits',
                ex_output[:, :1],
                epoch,
                dataformats='NCWH',
            )
            writer.add_images(
                'Bottom/' + partition + '/Output/Logits',
                ex_output[:, 1:],
                epoch,
                dataformats='NCWH',
            )
            msk = torch.sigmoid(ex_output)
            msk[0, :, 0, 0] = 0
            msk[0, :, 0, 1] = 1
            writer.add_images(
                'Top/' + partition + '/Output/Probabilities',
                msk[:, :1],
                epoch,
                dataformats='NCWH',
            )
            writer.add_images(
                'Bottom/' + partition + '/Output/Probabilities',
                msk[:, 1:],
                epoch,
                dataformats='NCWH',
            )

        # remember best loss and save checkpoint
        is_best = loss_val < best_loss_val
        best_loss_val = min(loss_val, best_loss_val)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss_val,
            'optimizer': optimizer.state_dict(),
            'meters': meters_val,
        }, is_best)
        meters_to_csv(meters_val, is_best)

        # Ensure the tensorboard outputs for this epoch are flushed
        writer.flush()

    # Close tensorboard connection
    writer.close()


def train(loader, model, criterion, optimizer, device, epoch, dtype=torch.float, print_freq=10):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    accuracies = AverageMeter('Accuracy', ':6.2f')
    precisions = AverageMeter('Precision', ':6.2f')
    recalls = AverageMeter('Recall', ':6.2f')
    f1s = AverageMeter('F1', ':6.4f')
    jaccards = AverageMeter('Jaccard', ':6.4f')

    top_accuracies = AverageMeter('Accuracy (top)', ':6.2f')
    top_precisions = AverageMeter('Precision (top)', ':6.2f')
    top_recalls = AverageMeter('Recall (top)', ':6.2f')
    top_f1s = AverageMeter('F1 (top)', ':6.4f')
    top_jaccards = AverageMeter('Jaccard (top)', ':6.4f')
    top_active_output = AverageMeter('Active output (top)', ':6.2f')
    top_active_target = AverageMeter('Active target (top)', ':6.2f')

    bot_accuracies = AverageMeter('Accuracy (bottom)', ':6.2f')
    bot_precisions = AverageMeter('Precision (bottom)', ':6.2f')
    bot_recalls = AverageMeter('Recall (bottom)', ':6.2f')
    bot_f1s = AverageMeter('F1 (bottom)', ':6.4f')
    bot_jaccards = AverageMeter('Jaccard (bottom)', ':6.4f')
    bot_active_output = AverageMeter('Active output (bottom)', ':6.2f')
    bot_active_target = AverageMeter('Active target (bottom)', ':6.2f')

    meters = [
        accuracies, precisions, recalls, f1s, jaccards,
        top_accuracies, top_precisions, top_recalls, top_f1s, top_jaccards, top_active_output, top_active_target,
        bot_accuracies, bot_precisions, bot_recalls, bot_f1s, bot_jaccards, bot_active_output, bot_active_target,
    ]

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses, accuracies, f1s, jaccards],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    example_data = example_output = example_target = None

    end = time.time()
    for i, batch in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        data = batch['signals'].unsqueeze(1)
        target = torch.stack((batch['mask_top'], batch['mask_bot']), dim=1)

        data = data.to(device, dtype, non_blocking=True)
        target = target.to(device, dtype, non_blocking=True)

        # Compute output
        output = model(data)
        loss = criterion(output, target)
        # Record loss
        ns = data.size(0)
        losses.update(loss.item(), ns)

        if i == max(0, len(loader) - 2):
            example_data = data.detach()
            example_target = target.detach()
            example_output = output.detach()

        # Measure and record performance with various metrics
        accuracies.update(100.0 * criterions.mask_accuracy_with_logits(output, target).item(), ns)
        precisions.update(100.0 * criterions.mask_precision_with_logits(output, target).item(), ns)
        recalls.update(100.0 * criterions.mask_recall_with_logits(output, target).item(), ns)
        f1s.update(criterions.mask_f1_score_with_logits(output, target).item(), ns)
        jaccards.update(criterions.mask_jaccard_index_with_logits(output, target).item(), ns)

        top_output, bot_output = output.unbind(1)
        top_target, bot_target = target.unbind(1)

        top_accuracies.update(100.0 * criterions.mask_accuracy_with_logits(top_output, top_target).item(), ns)
        top_precisions.update(100.0 * criterions.mask_precision_with_logits(top_output, top_target).item(), ns)
        top_recalls.update(100.0 * criterions.mask_recall_with_logits(top_output, top_target).item(), ns)
        top_f1s.update(criterions.mask_f1_score_with_logits(top_output, top_target).item(), ns)
        top_jaccards.update(criterions.mask_jaccard_index_with_logits(top_output, top_target).item(), ns)
        top_active_output.update(100.0 * criterions.mask_active_fraction(top_output).item(), ns)
        top_active_target.update(100.0 * criterions.mask_active_fraction(top_target).item(), ns)

        bot_accuracies.update(100.0 * criterions.mask_accuracy_with_logits(bot_output, bot_target).item(), ns)
        bot_precisions.update(100.0 * criterions.mask_precision_with_logits(bot_output, bot_target).item(), ns)
        bot_recalls.update(100.0 * criterions.mask_recall_with_logits(bot_output, bot_target).item(), ns)
        bot_f1s.update(criterions.mask_f1_score_with_logits(bot_output, bot_target).item(), ns)
        bot_jaccards.update(criterions.mask_jaccard_index_with_logits(bot_output, bot_target).item(), ns)
        bot_active_output.update(100.0 * criterions.mask_active_fraction(bot_output).item(), ns)
        bot_active_target.update(100.0 * criterions.mask_active_fraction(bot_target).item(), ns)

        # compute gradient and do optimizer update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 or i + 1 == len(loader):
            progress.display(i + 1)

    return losses.avg, meters, (example_data, example_target, example_output)


def validate(loader, model, criterion, device, dtype=torch.float, print_freq=10,
             prefix='Test', num_examples=32):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    accuracies = AverageMeter('Accuracy', ':6.2f')
    precisions = AverageMeter('Precision', ':6.2f')
    recalls = AverageMeter('Recall', ':6.2f')
    f1s = AverageMeter('F1', ':6.4f')
    jaccards = AverageMeter('Jaccard', ':6.4f')

    top_accuracies = AverageMeter('Accuracy (top)', ':6.2f')
    top_precisions = AverageMeter('Precision (top)', ':6.2f')
    top_recalls = AverageMeter('Recall (top)', ':6.2f')
    top_f1s = AverageMeter('F1 (top)', ':6.4f')
    top_jaccards = AverageMeter('Jaccard (top)', ':6.4f')
    top_active_output = AverageMeter('Active output (top)', ':6.2f')
    top_active_target = AverageMeter('Active target (top)', ':6.2f')

    bot_accuracies = AverageMeter('Accuracy (bottom)', ':6.2f')
    bot_precisions = AverageMeter('Precision (bottom)', ':6.2f')
    bot_recalls = AverageMeter('Recall (bottom)', ':6.2f')
    bot_f1s = AverageMeter('F1 (bottom)', ':6.4f')
    bot_jaccards = AverageMeter('Jaccard (bottom)', ':6.4f')
    bot_active_output = AverageMeter('Active output (bottom)', ':6.2f')
    bot_active_target = AverageMeter('Active target (bottom)', ':6.2f')

    meters = [
        accuracies, precisions, recalls, f1s, jaccards,
        top_accuracies, top_precisions, top_recalls, top_f1s, top_jaccards, top_active_output, top_active_target,
        bot_accuracies, bot_precisions, bot_recalls, bot_f1s, bot_jaccards, bot_active_output, bot_active_target,
    ]

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses, accuracies, f1s, jaccards],
        prefix=prefix + ': ',
    )

    # switch to evaluate mode
    model.eval()

    example_data = []
    example_output = []
    example_target = []
    example_interval = max(1, len(loader) // num_examples)

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            data = batch['signals'].unsqueeze(1)
            target = torch.stack((batch['mask_top'], batch['mask_bot']), dim=1)

            data = data.to(device, dtype, non_blocking=True)
            target = target.to(device, dtype, non_blocking=True)

            # Compute output
            output = model(data)
            loss = criterion(output, target)
            # Record loss
            ns = data.size(0)
            losses.update(loss.item(), ns)

            if i % example_interval == 0 and len(example_data) < num_examples:
                example_data.append(data[0].detach())
                example_target.append(target[0].detach())
                example_output.append(output[0].detach())

            # Measure and record performance with various metrics
            accuracies.update(100.0 * criterions.mask_accuracy_with_logits(output, target, reduction='none'))
            precisions.update(100.0 * criterions.mask_precision_with_logits(output, target, reduction='none'))
            recalls.update(100.0 * criterions.mask_recall_with_logits(output, target, reduction='none'))
            f1s.update(criterions.mask_f1_score_with_logits(output, target, reduction='none'))
            jaccards.update(criterions.mask_jaccard_index_with_logits(output, target, reduction='none'))

            top_output, bot_output = output.unbind(1)
            top_target, bot_target = target.unbind(1)

            top_accuracies.update(100.0 * criterions.mask_accuracy_with_logits(top_output, top_target, reduction='none'))
            top_precisions.update(100.0 * criterions.mask_precision_with_logits(top_output, top_target, reduction='none'))
            top_recalls.update(100.0 * criterions.mask_recall_with_logits(top_output, top_target, reduction='none'))
            top_f1s.update(criterions.mask_f1_score_with_logits(top_output, top_target, reduction='none'))
            top_jaccards.update(criterions.mask_jaccard_index_with_logits(top_output, top_target, reduction='none'))
            top_active_output.update(100.0 * criterions.mask_active_fraction(top_output, reduction='none'))
            top_active_target.update(100.0 * criterions.mask_active_fraction(top_target, reduction='none'))

            bot_accuracies.update(100.0 * criterions.mask_accuracy_with_logits(bot_output, bot_target, reduction='none'))
            bot_precisions.update(100.0 * criterions.mask_precision_with_logits(bot_output, bot_target, reduction='none'))
            bot_recalls.update(100.0 * criterions.mask_recall_with_logits(bot_output, bot_target, reduction='none'))
            bot_f1s.update(criterions.mask_f1_score_with_logits(bot_output, bot_target, reduction='none'))
            bot_jaccards.update(criterions.mask_jaccard_index_with_logits(bot_output, bot_target, reduction='none'))
            bot_active_output.update(100.0 * criterions.mask_active_fraction(bot_output, reduction='none'))
            bot_active_target.update(100.0 * criterions.mask_active_fraction(bot_target, reduction='none'))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0 or i + 1 == len(loader):
                progress.display(i + 1)

    # Restack samples, converting list into higher-dim tensor
    example_data = torch.stack(example_data, dim=0)
    example_target = torch.stack(example_target, dim=0)
    example_output = torch.stack(example_output, dim=0)

    return losses.avg, meters, (example_data, example_target, example_output)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def meters_to_csv(meters, is_best, filename='meters.csv'):
    df = pd.DataFrame()
    for meter in meters:
        df[meter.name] = meter.values
    df.to_csv(filename, index=False)
    if is_best:
        shutil.copyfile(filename, 'model_best.meters.csv')


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

    # Model parameters
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

    main(**vars(parser.parse_args()))
