import os
import random
import warnings

import numpy as np
import torch.utils.data

import echofilter.shardloader


class TransectDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            transect_paths,
            window_len=128,
            crop_depth=70,
            num_windows_per_transect=0,
            use_dynamic_offsets=True,
            transform_pre=None,
            transform_post=None,
            ):
        '''
        TransectDataset

        Parameters
        ----------
        transect_paths : list
            Absolute paths to transects.
        window_len : int
            Width (number of timestamps) to load. Default is `128`.
        crop_depth : float
            Maximum depth to include, in metres. Deeper data will be cropped
            away. Default is `70`.
        num_windows_per_transect : int
            Number of windows to extract for each transect. Start indices for
            the windows will be equally spaced across the total width of the
            transect. If this is `0`, the number of windows will be inferred
            automatically based on `window_len` and the total width of the
            transect, resulting in a different number of windows for each
            transect. Default is `0`.
        use_dynamic_offsets : bool
            Whether starting indices for each window should be randomly offset.
            Set to `True` for training and `False` for testing. Default is
            `True`.
        transform_pre : callable
            Operations to perform to the dictionary containing a single sample.
            These are performed before generating the masks. Default is `None`.
        transform_post : callable
            Operations to perform to the dictionary containing a single sample.
            These are performed after generating the masks. Default is `None`.
        '''
        super(TransectDataset, self).__init__()
        self.transect_paths = transect_paths
        self.window_len = window_len
        self.crop_depth = crop_depth
        self.num_windows = num_windows_per_transect
        self.use_dynamic_offsets = use_dynamic_offsets
        self.transform_pre = transform_pre
        self.transform_post = transform_post
        self.initialise_datapoints()

    def initialise_datapoints(self):

        self.datapoints = []

        for transect_path in self.transect_paths:
            # Check how many segments the transect was divided into
            segments_meta_fname = os.path.join(transect_path, 'n_segment.txt')
            if not os.path.isfile(segments_meta_fname):
                # Silently skip missing transects
                continue
            with open(segments_meta_fname, 'r') as f:
                n_segment = int(f.readline().strip())

            # For each segment, specify some samples over its duration
            for i_segment in range(n_segment):
                seg_path = os.path.join(transect_path, str(i_segment))
                # Lookup the number of rows in the transect
                # Load the sharding metadata
                with open(os.path.join(seg_path, 'shard_size.txt'), 'r') as f:
                    n_timestamps, shard_len = f.readline().strip().split(',')
                    n_timestamps = int(n_timestamps)
                # Generate an array for window centers within the transect
                # - if this is for training, we want to randomise the offsets
                # - if this is for validation, we want stable windows
                num_windows = self.num_windows
                if self.num_windows is None or self.num_windows == 0:
                    # Load enough windows to include all datapoints
                    num_windows = int(np.ceil(n_timestamps / self.window_len))
                centers = np.linspace(0, n_timestamps, num_windows + 1)[:num_windows]
                if len(centers) > 1:
                    max_dy_offset = centers[1] - centers[0]
                else:
                    max_dy_offset = n_timestamps
                if self.use_dynamic_offsets:
                    centers += random.random() * max_dy_offset
                else:
                    centers += max_dy_offset / 2
                centers = np.round(centers)
                # Add each (transect, center) to the list for this epoch
                for center_idx in centers:
                    self.datapoints.append((seg_path, int(center_idx)))

    def __getitem__(self, index):
        transect_pth, center_idx = self.datapoints[index]
        # Load data from shards
        sample = echofilter.shardloader.load_transect_from_shards_abs(
            transect_pth,
            center_idx - int(self.window_len / 2),
            center_idx - int(self.window_len / 2) + self.window_len,
            pad_mode='reflect',
        )
        sample['d_top'] = sample.pop('top')
        sample['d_bot'] = sample.pop('bottom')
        sample['d_surf'] = sample.pop('surface')
        sample['d_top-original'] = sample.pop('top-original')
        sample['d_bot-original'] = sample.pop('bottom-original')
        sample['signals'] = sample.pop('Sv')
        sample['mask'] = sample['mask'].astype(np.float)
        # Handle missing top and bottom lines during passive segments
        if sample['is_upward_facing']:
            passive_top_val = np.min(sample['depths'])
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'All-NaN (slice|axis) encountered')
                passive_bot_val = np.nanmax(sample['d_bot'])
            if np.isnan(passive_bot_val):
                passive_bot_val = np.max(sample['depths']) - 1.69
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'All-NaN (slice|axis) encountered')
                passive_top_val = np.nanmin(sample['d_top'])
            if np.isnan(passive_top_val):
                passive_top_val = 0.966
            passive_bot_val = np.max(sample['depths'])
        sample['d_top'][np.isnan(sample['d_top'])] = passive_top_val
        sample['d_bot'][np.isnan(sample['d_bot'])] = passive_bot_val
        sample['d_surf'][np.isnan(sample['d_surf'])] = np.min(sample['depths'])
        sample['d_top-original'][np.isnan(sample['d_top-original'])] = passive_top_val
        sample['d_bot-original'][np.isnan(sample['d_bot-original'])] = passive_bot_val

        if self.transform_pre is not None:
            sample = self.transform_pre(sample)

        # Apply depth crop
        depth_crop_mask = sample['depths'] <= self.crop_depth
        sample['depths'] = sample['depths'][depth_crop_mask]
        sample['signals'] = sample['signals'][:, depth_crop_mask]
        sample['mask'] = sample['mask'][:, depth_crop_mask]

        # Convert lines to masks and relative lines
        ddepths = np.broadcast_to(sample['depths'], sample['signals'].shape)
        for suffix in ['', '-original']:
            sample['mask_top' + suffix] = np.single(
                ddepths < np.expand_dims(sample['d_top' + suffix], -1)
            )
            sample['mask_bot' + suffix] = np.single(
                ddepths > np.expand_dims(sample['d_bot' + suffix], -1)
            )
        sample['mask_surf'] = np.single(ddepths < np.expand_dims(sample['d_surf'], -1))

        depth_range = abs(sample['depths'][-1] - sample['depths'][0])
        for key in ['d_top', 'd_bot', 'd_surf', 'd_top-original', 'd_bot-original']:
            sample['r' + key[1:]] = sample[key] / depth_range

        # Make a mask indicating left-over patches. This is 0 everywhere,
        # except 1s whereever pixels in the overall mask are removed for
        # reasons not explained by the top and bottom lines, and is_passive and
        # is_removed indicators.
        sample['mask_patches'] = 1 - sample['mask']
        sample['mask_patches'][sample['is_passive'] > 0.5] = 0
        sample['mask_patches'][sample['is_removed'] > 0.5] = 0

        sample['mask_patches-original'] = sample['mask_patches'].copy()
        for suffix in ('', '-original'):
            sample['mask_patches' + suffix][sample['mask_top' + suffix] > 0.5] = 0
            sample['mask_patches' + suffix][sample['mask_bot' + suffix] > 0.5] = 0
            sample['mask_patches' + suffix] = np.single(sample['mask_patches' + suffix])

        if self.transform_post is not None:
            sample = self.transform_post(sample)

        return sample

    def __len__(self):
        return len(self.datapoints)
