#!/usr/bin/env python

import shutil
import time

import torch
import torch.nn
import torch.optim
import torch.utils.data
import torchvision.transforms
from torchutils.random import seed_all

import echofilter.dataset
import echofilter.transforms
from echofilter.meters import AverageMeter, ProgressMeter
from echofilter.rawloader import get_partition_list
from echofilter.unet import UNet


DATA_MEAN = -81.5
DATA_STDEV = 21.9


def main(
        data_dir='/data/dsforce/surveyExports',
        dataset_name='mobile',
        sample_shape=(128, 512),
        crop_depth=70,
        latent_channels=64,
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

    # Augmentations
    train_transform_pre = torchvision.transforms.Compose([
        echofilter.transforms.RandomCropWidth(0.5),
        echofilter.transforms.RandomStretchDepth(0.5),
        echofilter.transforms.RandomReflection(),
    ])
    train_transform_post = torchvision.transforms.Compose([
        echofilter.transforms.Rescale(sample_shape),
        echofilter.transforms.Normalize(DATA_MEAN, DATA_STDEV),
        echofilter.transforms.ColorJitter(0.5, 0.3),
    ])
    val_transform = torchvision.transforms.Compose([
        echofilter.transforms.Rescale(sample_shape),
        echofilter.transforms.Normalize(DATA_MEAN, DATA_STDEV),
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

    dataset_train = echofilter.dataset.TransectDataset(
        train_paths,
        window_len=int(1.5 * sample_shape[0]),
        crop_depth=crop_depth,
        num_windows_per_transect=10,
        use_dynamic_offsets=True,
        transform_pre=train_transform_pre,
        transform_post=train_transform_post,
    )
    dataset_val = echofilter.dataset.TransectDataset(
        val_paths,
        window_len=sample_shape[0],
        crop_depth=crop_depth,
        num_windows_per_transect=20,
        use_dynamic_offsets=False,
        transform_post=val_transform,
    )

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_worker,
        pin_memory=True,
        drop_last=False,
    )
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_worker,
        pin_memory=True,
        drop_last=False,
    )

    model = UNet(1, 2, latent_channels=latent_channels)
    model.to(device)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr,
        betas=(momentum, 0.999),
        weight_decay=weight_decay,
    )

    best_loss = float('inf')
    for epoch in range(n_epoch):

        # train for one epoch
        train(loader_train, model, criterion, optimizer, device, epoch, print_freq=print_freq)

        # evaluate on validation set
        loss = validate(loader_val, model, criterion, device, print_freq=print_freq)

        # remember best loss and save checkpoint
        is_best = loss < best_loss
        best_loss = max(loss, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(loader, model, criterion, optimizer, device, epoch, print_freq=10):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        data = batch['signals'].unsqueeze(1)
        target = torch.stack((batch['mask_top'], batch['mask_bot']), dim=1)

        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), data.size(0))

        # compute gradient and do optimizer update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)


def validate(loader, model, criterion, device, print_freq=10):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(loader),
        [batch_time, losses],
        prefix='Test: ',
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(loader):

            data = batch['signals'].unsqueeze(1)
            target = torch.stack((batch['mask_top'], batch['mask_bot']), dim=1)

            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

    return losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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

    # Model parameters
    parser.add_argument(
        '--latent-channels',
        type=int,
        default=64,
        help='number of latent channels to use in the model',
    )

    # Training methodology parameters
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
