"""
Tools for converting a dataset of echograms (transects) into a Pytorch dataset
and sampling from it.
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

import os
import random
import warnings

import numpy as np
import torch.utils.data

from ..raw import shardloader
from . import utils


class TransectDataset(torch.utils.data.Dataset):
    """
    Load a collection of transects as a PyTorch dataset.

    Parameters
    ----------
    transect_paths : list
        Absolute paths to transects.
    window_len : int
        Width (number of timestamps) to load. Default is `128`.
    p_scale_window : float, optional
        Probability of rescaling window. Default is `0`, which results in no
        randomization of the window widths.
    window_sf : float, optional
        Maximum window scale factor. Scale factors will be log-uniformly
        sampled in the range `1/window_sf` to `window_sf`. Default is `2`.
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
    crop_depth : float
        Maximum depth to include, in metres. Deeper data will be cropped
        away. Default is `None`.
    transform : callable
        Operations to perform to the dictionary containing a single sample.
        These are performed before generating the turbulence/bottom/overall
        mask. Default is `None`.
    remove_nearfield : bool, optional
        Whether to remove turbulence and bottom lines affected by nearfield
        removal. If `True` (default), targets for the line near to the
        sounder (bottom if upward facing, turbulence otherwise) which are
        closer than or equal to a distance of `nearfield_distance` become
        reduced to `nearfield_visible_dist`.
    nearfield_distance : float, optional
        Nearfield distance in metres. Regions closer than the nearfield
        may have been masked out from the dataset, but their effect will
        be removed from the targets if `remove_nearfield=True`.
        Default is `1.7`.
    nearfield_visible_dist : float, optional
        The distance at which the effect of being to close to the sounder
        is obvious to the naked eye, and hence the distance which nearfield
        will be mapped to if `remove_nearfield=True`. Default is `0.0`.
    remove_offset_turbulence : float, optional
        Line offset built in to the turbulence line. If given, this will be
        removed from the samples within the dataset. Default is `0`.
    remove_offset_bottom : float, optional
        Line offset built in to the bottom line. If given, this will be
        removed from the samples within the dataset. Default is `0`.
    """

    def __init__(
        self,
        transect_paths,
        window_len=128,
        p_scale_window=0,
        window_sf=2,
        num_windows_per_transect=0,
        use_dynamic_offsets=True,
        crop_depth=None,
        transform=None,
        remove_nearfield=True,
        nearfield_distance=1.7,
        nearfield_visible_dist=0.0,
        remove_offset_turbulence=0,
        remove_offset_bottom=0,
    ):
        super(TransectDataset, self).__init__()
        self.transect_paths = transect_paths
        self.window_len = window_len
        self.p_scale_window = p_scale_window
        self.max_window_sf = window_sf if p_scale_window else 1
        self.num_windows = num_windows_per_transect
        self.use_dynamic_offsets = use_dynamic_offsets
        self.crop_depth = crop_depth
        self.transform = transform
        self.remove_nearfield = remove_nearfield
        self.nearfield_distance = nearfield_distance
        self.nearfield_visible_dist = nearfield_visible_dist
        self.remove_offset_turbulence = remove_offset_turbulence
        self.remove_offset_bottom = remove_offset_bottom
        self.initialise_datapoints()

    def initialise_datapoints(self):
        """
        Parse `transect_paths` to generate sampling windows for each transect.
        Manually calling this method will resample the transect offsets and
        widths if they were randomly generated.
        """

        self.datapoints = []

        for transect_path in self.transect_paths:
            # Check how many segments the transect was divided into
            segments_meta_fname = os.path.join(transect_path, "n_segment.txt")
            if not os.path.isfile(segments_meta_fname):
                # Silently skip missing transects
                continue
            with open(segments_meta_fname, "r") as f:
                n_segment = int(f.readline().strip())

            # For each segment, specify some samples over its duration
            for i_segment in range(n_segment):
                seg_path = os.path.join(transect_path, str(i_segment))
                # Lookup the number of rows in the transect
                # Load the sharding metadata
                with open(os.path.join(seg_path, "shard_size.txt"), "r") as f:
                    n_timestamps, shard_len = f.readline().strip().split(",")
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
                # Generate a width for each window, and add each
                # (transect, center, width) to the list for this epoch
                for center_idx in centers:
                    cur_win_len = self.window_len
                    if self.p_scale_window and random.random() < self.p_scale_window:
                        sf = np.exp(np.log(self.max_window_sf) * random.uniform(-1, 1))
                        cur_win_len *= sf
                    cur_win_len = int(np.round(cur_win_len))
                    self.datapoints.append((seg_path, int(center_idx), cur_win_len))

    def __getitem__(self, index):
        transect_pth, center_idx, win_len = self.datapoints[index]
        # Load data from shards
        sample = shardloader.load_transect_from_shards_abs(
            transect_pth,
            center_idx - int(win_len / 2),
            center_idx - int(win_len / 2) + win_len,
            pad_mode="reflect",
        )
        sample["d_turbulence"] = sample.pop("turbulence")
        sample["d_bottom"] = sample.pop("bottom")
        sample["d_surface"] = sample.pop("surface")
        sample["d_turbulence-original"] = sample.pop("turbulence-original")
        sample["d_bottom-original"] = sample.pop("bottom-original")
        sample["signals"] = sample.pop("Sv")
        if sample["depths"][-1] < sample["depths"][0]:
            # Found some upward-facing data that needs to be reflected
            for k in ["depths", "signals", "mask"]:
                sample[k] = np.flip(sample[k], -1).copy()

        # Convert mask patches into floating point arrays
        for suffix in ("", "-original", "-ntob"):
            sample["mask_patches" + suffix] = sample["mask_patches" + suffix].astype(
                np.float32
            )

        sample["depths"] = sample["depths"].astype(np.float32)

        # Fix any broken surface lines (these were generated automatically
        # by Echoview and not adjusted by human annotator, so are not
        # guaranteed to be sane).
        # Note any locations where the labels are not sane. These can be
        # removed from the training loss.
        sample["is_bad_labels"] = sample["d_surface"] >= sample["d_bottom"]
        # Surface line can not be below turbulence line
        sample["d_surface"] = np.minimum(sample["d_surface"], sample["d_turbulence"])
        # Ensure the bottom line is always below the surface line as well
        sample["d_bottom"] = np.maximum(sample["d_bottom"], sample["d_surface"] + 0.02)

        if sample["is_upward_facing"]:
            min_top_depth = np.min(sample["depths"])
            max_bot_depth = np.max(sample["depths"]) - self.nearfield_visible_dist
        else:
            min_top_depth = 0.966
            max_bot_depth = np.max(sample["depths"])

        if self.remove_nearfield:
            depth_intv = np.abs(sample["depths"][-1] - sample["depths"][-2])
            if sample["is_upward_facing"]:
                nearfield_threshold = (
                    np.max(sample["depths"])
                    - depth_intv * 2.5
                    - self.nearfield_distance
                )
                was_in_nearfield = sample["d_bottom"] >= nearfield_threshold
                sample["d_bottom"][was_in_nearfield] = max_bot_depth
                was_in_nearfield_og = sample["d_bottom-original"] >= nearfield_threshold
                sample["d_bottom-original"][was_in_nearfield_og] = max_bot_depth
                # Extend/contract mask_patches where necessary
                idx_search = utils.last_nonzero(sample["depths"] < nearfield_threshold)
                idx_fillto = utils.first_nonzero(sample["depths"] > max_bot_depth)
                is_close_patch = np.any(
                    sample["mask_patches"][:, idx_search:idx_fillto], -1
                )
                sample["mask_patches"][is_close_patch, idx_search:idx_fillto] = 1
                is_close_patch_og = sample["mask_patches-original"][:, idx_search] > 0
                is_close_patch_og = np.expand_dims(is_close_patch_og, -1)
                sample["mask_patches-original"][:, idx_search:] = is_close_patch_og
                sample["mask_patches-ntob"][:, idx_search:] = is_close_patch_og
            else:
                was_in_nearfield = (
                    sample["d_turbulence"] < self.nearfield_distance + depth_intv
                )
                sample["d_turbulence"][was_in_nearfield] = min_top_depth
                was_in_nearfield_og = np.zeros_like(sample["is_removed"], dtype="bool")
                # Extend/contract mask_patches where necessary
                idx_search = utils.first_nonzero(
                    sample["depths"] > self.nearfield_distance, invalid_val=0
                )
                idx_fillfr = utils.last_nonzero(
                    sample["depths"] < min_top_depth, invalid_val=-1
                )
                is_close_patch = np.any(
                    sample["mask_patches"][:, idx_fillfr + 1 : idx_search + 1], -1
                )
                sample["mask_patches"][is_close_patch, idx_fillfr + 1 : idx_search] = 1
                is_close_patch = np.any(
                    sample["mask_patches-ntob"][:, idx_fillfr + 1 : idx_search + 1], -1
                )
                sample["mask_patches-ntob"][
                    is_close_patch, idx_fillfr + 1 : idx_search
                ] = 1
                is_close_patch_og = sample["mask_patches-original"][:, idx_search] > 0
                is_close_patch_og = np.expand_dims(is_close_patch_og, -1)
                sample["mask_patches-original"][:, :idx_search] = is_close_patch_og
        else:
            was_in_nearfield = np.zeros_like(sample["is_removed"], dtype="bool")
            was_in_nearfield_og = np.zeros_like(sample["is_removed"], dtype="bool")

        if self.remove_offset_turbulence:
            # Check the mask beforehand
            _ddepths = np.broadcast_to(sample["depths"], sample["signals"].shape)
            _in_mask = _ddepths < np.expand_dims(sample["d_turbulence"], -1)
            _in_mask_og = _ddepths < np.expand_dims(sample["d_turbulence-original"], -1)
            # Shift lines up higher (less deep)
            sample["d_turbulence"][
                (~was_in_nearfield) | sample["is_upward_facing"]
            ] -= self.remove_offset_turbulence
            sample["d_turbulence-original"][
                (~was_in_nearfield_og) | sample["is_upward_facing"]
            ] -= self.remove_offset_turbulence
            # Extend mask_patches where necessary
            _fx_mask = _ddepths < np.expand_dims(sample["d_turbulence"], -1)
            _df_mask = _in_mask * (_in_mask ^ _fx_mask)
            is_close_patch = np.any(
                _df_mask[:, :-1] * sample["mask_patches"][:, 1:], -1
            )
            sample["mask_patches"][is_close_patch, :] += _df_mask[is_close_patch, :]
            # ... and extend ntob mask patches too
            sample["mask_patches-ntob"][is_close_patch, :] += _df_mask[
                is_close_patch, :
            ]
            # ... and extend og mask patches too
            _fx_mask_og = _ddepths < np.expand_dims(sample["d_turbulence-original"], -1)
            _df_mask = _in_mask_og * (_in_mask_og ^ _fx_mask_og)
            is_close_patch = np.any(
                _df_mask[:, :-1] * sample["mask_patches-original"][:, 1:], -1
            )
            sample["mask_patches-original"][is_close_patch, :] += _df_mask[
                is_close_patch, :
            ]

        if self.remove_offset_bottom:
            # Check the mask beforehand
            _ddepths = np.broadcast_to(sample["depths"], sample["signals"].shape)
            _in_mask = _ddepths > np.expand_dims(sample["d_bottom"], -1)
            _in_mask_og = _ddepths > np.expand_dims(sample["d_bottom-original"], -1)
            # Shift lines down lower (more deep)
            sample["d_bottom"][
                (~was_in_nearfield) | ~sample["is_upward_facing"]
            ] += self.remove_offset_bottom
            sample["d_bottom-original"][
                (~was_in_nearfield_og) | sample["is_upward_facing"]
            ] += self.remove_offset_bottom
            # Extend mask_patches where necessary
            _fx_mask = _ddepths > np.expand_dims(sample["d_bottom"], -1)
            _df_mask = _in_mask * (_in_mask ^ _fx_mask)
            is_close_patch = np.any(
                _df_mask[:, 1:] * sample["mask_patches"][:, :-1], -1
            )
            sample["mask_patches"][is_close_patch, :] += _df_mask[is_close_patch, :]
            # ... and extend og mask patches too
            _fx_mask_og = _ddepths > np.expand_dims(sample["d_bottom-original"], -1)
            _df_mask = _in_mask_og * (_in_mask_og ^ _fx_mask_og)
            is_close_patch = np.any(
                _df_mask[:, 1:] * sample["mask_patches-original"][:, :-1], -1
            )
            sample["mask_patches-original"][is_close_patch, :] += _df_mask[
                is_close_patch, :
            ]
            # ... and extend ntob mask patches too
            sample["mask_patches-ntob"][is_close_patch, :] += _df_mask[
                is_close_patch, :
            ]

        if self.remove_offset_turbulence or self.remove_offset_bottom:
            # Change any 2s in the mask to be 1s
            for suffix in ("", "-original", "-ntob"):
                sample["mask_patches" + suffix] = (
                    sample["mask_patches" + suffix] > 0.5
                ).astype(np.float32)

        # Ensure all line values are finite
        for key in [
            "d_surface",
            "d_turbulence",
            "d_turbulence-original",
            "d_bottom",
            "d_bottom-original",
        ]:
            where_invalid = ~np.isfinite(sample[key])
            if np.sum(where_invalid) > 0 and np.sum(where_invalid) < len(sample[key]):
                sample[key][where_invalid] = np.interp(
                    sample["timestamps"][where_invalid],
                    sample["timestamps"][~where_invalid],
                    sample[key][~where_invalid],
                )

        # Apply depth crop
        if self.crop_depth is not None:
            depth_crop_mask = sample["depths"] <= self.crop_depth
            sample["depths"] = sample["depths"][depth_crop_mask]
            sample["signals"] = sample["signals"][:, depth_crop_mask]

        if self.transform is not None:
            sample = self.transform(sample)

        # Convert lines to masks and relative lines
        ddepths = np.broadcast_to(sample["depths"], sample["signals"].shape)
        for suffix in ["", "-original"]:
            sample["mask_turbulence" + suffix] = ddepths < np.expand_dims(
                sample["d_turbulence" + suffix], -1
            )
            sample["mask_bottom" + suffix] = ddepths > np.expand_dims(
                sample["d_bottom" + suffix], -1
            )
        sample["mask_surface"] = ddepths < np.expand_dims(sample["d_surface"], -1)

        depth_range = abs(sample["depths"][-1] - sample["depths"][0])
        for key in [
            "d_turbulence",
            "d_bottom",
            "d_surface",
            "d_turbulence-original",
            "d_bottom-original",
        ]:
            sample["r" + key[1:]] = sample[key] / depth_range

        # Change is_surrogate_surface to exclude passive data
        sample["is_surrogate_surface"][sample["is_passive"] > 0.5] = 0

        # Create mask corresponding to the aggregate of all elements we need
        # masked in/out
        sample["mask"] = np.ones_like(sample["signals"])
        sample["mask"][sample["is_passive"] > 0.5] = 0
        sample["mask"][sample["is_removed"] > 0.5] = 0
        sample["mask"][sample["mask_turbulence"] > 0.5] = 0
        sample["mask"][sample["mask_bottom"] > 0.5] = 0
        sample["mask"][sample["mask_patches"] > 0.5] = 0

        # Determine the boundary index for depths
        for sfx in {"turbulence", "turbulence-original", "surface"}:
            # Ties are broken to the smaller index
            sample["index_" + sfx] = np.searchsorted(
                sample["depths"], sample["d_" + sfx], side="left"
            )
            # It is possible for the turbulence line to be above the field of
            # view, and impossible for the turbulence line to below
            sample["index_" + sfx] = np.maximum(
                0, np.minimum(len(sample["depths"]) - 1, sample["index_" + sfx])
            )
        for sfx in {"bottom", "bottom-original"}:
            # Ties are broken to the larger index
            sample["index_" + sfx] = np.searchsorted(
                sample["depths"], sample["d_" + sfx], side="right"
            )
            # It is possible for the bottom line to be below the field of view,
            # and impossible for the bottom line to above
            sample["index_" + sfx] -= 1
            sample["index_" + sfx] = np.maximum(
                0, np.minimum(len(sample["depths"]) - 1, sample["index_" + sfx])
            )

        # Ensure everything is float32 datatype
        for key in sample:
            sample[key] = sample[key].astype(np.float32)
            # Ensure no entries are invalid numbers
            nan_count = np.sum(np.isnan(sample[key]))
            if nan_count > 0:
                print("WARNING: NaN present in {}".format(key))
                sample[key][np.isnan(sample[key])] = 0
            inf_count = np.sum(np.isinf(sample[key]))
            if inf_count > 0:
                print("WARNING: inf present in {}".format(key))
                sample[key][np.isinf(sample[key])] = 0
            inf_count = np.sum(~np.isfinite(sample[key]))
            if inf_count > 0:
                print("WARNING: non-finite numbers present in {}".format(key))
                sample[key][~np.isfinite(sample[key])] = 0

        input = np.expand_dims(sample["signals"], 0)
        return input, sample

    def __len__(self):
        return len(self.datapoints)


class ConcatDataset(torch.utils.data.ConcatDataset):
    """
    Dataset as a concatenation of multiple TransectDatasets.

    This class is useful to assemble different existing datasets.

    Parameters
    ----------
    datasets : sequence
        List of datasets to be concatenated.

    Notes
    -----
    A subclass of `torch.utils.data.ConcatDataset` which supports the
    `initialise_datapoints` method.
    """

    def initialise_datapoints(self):
        for dataset in self.datasets:
            dataset.initialise_datapoints()


class StratifiedRandomSampler(torch.utils.data.Sampler):
    """
    Samples elements randomly without repetition, stratified across datasets in
    the data_source.

    Parameters
    ----------
    data_source : torch.utils.data.ConcatDataset
        Dataset to sample from. Must possess a `cumulative_sizes` attribute.
    """

    def __init__(self, data_source):
        self.data_source = data_source

    @property
    def num_samples(self):
        # dataset size might change at runtime
        return len(self.data_source)

    def __iter__(self):

        n_sample = len(self.data_source)
        n_dataset = len(self.data_source.cumulative_sizes)

        perms = []
        lower = 0
        for upper in self.data_source.cumulative_sizes:
            p = list(range(lower, upper))
            random.shuffle(p)
            perms.append(p)
            lower = upper

        dataset_sizes = np.array([len(p) for p in perms])
        target_fraction = dataset_sizes / n_sample
        sub_tallies = np.zeros(n_dataset, dtype=int)
        cur_fraction = np.zeros(n_dataset)

        indices = []
        while len(indices) < n_sample:
            i = np.argmax(target_fraction - cur_fraction)
            indices.append(perms[i].pop())
            sub_tallies[i] += 1
            cur_fraction = sub_tallies / len(indices)

        return iter(indices)

    def __len__(self):
        return self.num_samples
