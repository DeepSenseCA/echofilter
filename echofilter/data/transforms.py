"""
Transformations and augmentations to be applied to echogram transects.
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

import collections
import os
import random
import warnings

import numpy as np
import scipy.interpolate
import scipy.ndimage.filters
import skimage.transform

from echofilter.nn.modules.utils import _pair


_fields_2d = (
    "Sv",
    "signals",
    "mask",
    "mask_turbulence",
    "mask_top",
    "mask_bottom",
    "mask_bot",
    "mask_turbulence-original",
    "mask_top-original",
    "mask_bottom-original",
    "mask_bot-original",
    "mask_surface",
    "mask_surf",
    "mask_patches",
    "mask_patches-original",
    "mask_patches-ntob",
)
_fields_1d_timelike = (
    "timestamps",
    "turbulence",
    "top",
    "bottom",
    "turbulence-original",
    "top-original",
    "bottom-original",
    "surface",
    "surf",
    "d_turbulence",
    "d_top",
    "d_bottom",
    "d_bot",
    "r_turbulence",
    "r_top",
    "r_bottom",
    "r_bot",
    "d_turbulence-original",
    "d_top-original",
    "d_bottom-original",
    "d_bot-original",
    "r_turbulence-original",
    "r_top-original",
    "r_bottom-original",
    "r_bot-original",
    "d_surface",
    "d_surf",
    "r_surface",
    "r_surf",
    "is_surrogate_surface",
    "is_bad_labels",
    "is_passive",
    "is_removed",
)
_fields_1d_depthlike = ("depths",)
_fields_0d = ("is_upward_facing",)
_fields_needing_linear = (
    "timestamps",
    "depths",
    "is_surrogate_surface",
    "is_bad_labels",
    "is_passive",
    "is_removed",
)


class Rescale(object):
    """
    Rescale the image(s) in a sample to a given size.

    Parameters
    ----------
    output_size : tuple or int
        Desired output size. If tuple, output is matched to output_size. If
        int, output is square.
    order : int or None, optional
        Order of the interpolation, for both image and vector elements.
        For images-like components, the interpolation is 2d.
        The following values are supported:

        - 0: Nearest-neighbor
        - 1: Linear (default)
        - 2: Quadratic
        - 3: Cubic

        If `None`, the order is randomly selected as either `0` or `1`.
    """

    order2kind = {
        0: "nearest",
        1: "linear",
        2: "quadratic",
        3: "cubic",
    }

    def __init__(self, output_size, order=1):
        self.output_size = _pair(output_size)
        self.order = order

    def __call__(self, sample):

        order = self.order
        if order is None:
            order = np.random.randint(1)

        kind = self.order2kind[order]

        # 2D arrays (image-like)
        for key in _fields_2d:
            if key not in sample:
                continue
            if sample[key].shape == self.output_size:
                continue
            _dtype = sample[key].dtype
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", "Bi-quadratic interpolation behavior has changed"
                )
                sample[key] = skimage.transform.resize(
                    np.asarray(sample[key]).astype(np.float),
                    self.output_size,
                    order=order,
                    clip=False,
                    preserve_range=False,
                )
            sample[key] = sample[key].astype(_dtype)

        # 1D arrays (column-like)
        for key in _fields_1d_timelike:
            if key not in sample:
                continue
            if sample[key].shape == self.output_size[:1]:
                continue
            _kind = "linear" if key == "timestamps" else kind
            if order > 1 and (
                key in _fields_needing_linear
                or ("bot" in key and sample["is_upward_facing"])
            ):
                _kind = "linear"
            _dtype = sample[key].dtype
            sample[key] = scipy.interpolate.interp1d(
                np.arange(len(sample[key])),
                sample[key],
                kind=_kind,
            )(np.linspace(0, len(sample[key]) - 1, self.output_size[0]))
            sample[key] = sample[key].astype(_dtype)

        # 1D arrays (row-like)
        for key in _fields_1d_depthlike:
            if key not in sample:
                continue
            if sample[key].shape == self.output_size[1:]:
                continue
            _kind = "linear" if key == "depths" else kind
            _dtype = sample[key].dtype
            sample[key] = scipy.interpolate.interp1d(
                np.arange(len(sample[key])),
                sample[key],
                kind=_kind,
            )(np.linspace(0, len(sample[key]) - 1, self.output_size[1]))
            sample[key] = sample[key].astype(_dtype)

        return sample


class RandomGridSampling(Rescale):
    """
    Resample data onto a new grid, which is randomly resampled.

    Parameters
    ----------
    output_size : tuple or int
        Desired output size. If tuple, output is matched to output_size. If
        int, output is square.
    p : float, optional
        Probability of performing the RandomGrid operation. Default is `0.5`.
    order : int or None, optional
        Order of the interpolation, for both image and vector elements.
        For images-like components, the interpolation is 2d.
        The following values are supported:

        - 0: Nearest-neighbor
        - 1: Linear (default)
        - 2: Quadratic
        - 3: Cubic

        If `None`, the order is randomly selected from the set
        {`0`, `1`, `3`}.
    """

    def __init__(self, *args, p=0.5, **kwargs):
        super(RandomGridSampling, self).__init__(*args, **kwargs)
        self.p = p

    def __call__(self, sample):

        if random.random() > self.p:
            # Nothing to do
            return sample

        order = self.order
        if order is None:
            # Randomly sample 0, 1, or 3
            order = np.random.randint(3)
            if order == 2:
                order += 1

        kind = self.order2kind[order]

        # Randomly re-sample x and y
        nx = len(sample["timestamps"])
        ny = len(sample["depths"])
        x_out = np.sort(np.random.uniform(0, nx - 1, size=self.output_size[0]))
        y_out = np.sort(np.random.uniform(0, ny - 1, size=self.output_size[1]))

        # 2D arrays (image-like)
        for key in _fields_2d:
            if key not in sample:
                continue
            _dtype = sample[key].dtype
            sample[key] = np.asarray(sample[key])
            if order == 0:
                if sample[key].shape != (nx, ny):
                    raise ValueError(
                        'Expected sample["{}"] to be shaped {}'.format(key, (nx, ny))
                    )
                sample[key] = sample[key][np.round(x_out).astype(int)][
                    :, np.round(y_out).astype(int)
                ]
            else:
                sample[key] = scipy.interpolate.RectBivariateSpline(
                    np.linspace(0, nx - 1, sample[key].shape[0]),
                    np.linspace(0, ny - 1, sample[key].shape[1]),
                    sample[key].astype(np.float),
                    kx=order,
                    ky=order,
                )(x_out, y_out)
            sample[key] = sample[key].astype(_dtype)

        # 1D arrays (column-like)
        for key in _fields_1d_timelike:
            if key not in sample:
                continue
            _kind = "linear" if key == "timestamps" else kind
            if order > 1 and (
                key in _fields_needing_linear
                or ("bot" in key and sample["is_upward_facing"])
            ):
                _kind = "linear"
            _dtype = sample[key].dtype
            sample[key] = scipy.interpolate.interp1d(
                np.linspace(0, nx - 1, len(sample[key])),
                sample[key],
                kind=_kind,
            )(x_out)
            sample[key] = sample[key].astype(_dtype)

        # 1D arrays (row-like)
        for key in _fields_1d_depthlike:
            if key not in sample:
                continue
            _kind = "linear" if key == "depths" else kind
            _dtype = sample[key].dtype
            sample[key] = scipy.interpolate.interp1d(
                np.linspace(0, ny - 1, len(sample[key])),
                sample[key],
                kind=_kind,
            )(y_out)
            sample[key] = sample[key].astype(_dtype)

        return sample


class RandomElasticGrid(Rescale):
    """
    Resample data onto a new grid, which is elastically deformed from the
    original sampling grid.

    Parameters
    ----------
    output_size : tuple or int or None
        Desired output size. If tuple, output is matched to output_size. If
        int, output is square. If `None`, the size remains unchanged from the
        input.
    p : float, optional
        Probability of performing the RandomGrid operation. Default is `0.5`.
    sigma : float, optional
        Gaussian filter kernel size. Default is `8.0`.
    alpha : float, optional
        Maximum size of image distortions, relative to the length of the side
        of the image. Default is `0.05`.
    order : int or None, optional
        Order of the interpolation, for both image and vector elements.
        For images-like components, the interpolation is 2d.
        The following values are supported:

        - 0: Nearest-neighbor
        - 1: Linear (default)
        - 2: Quadratic
        - 3: Cubic

        If `None`, the order is randomly selected from the set
        {`1`, `2`, `3`}.
    """

    def __init__(self, output_size, p=0.5, sigma=8.0, alpha=0.05, order=1):
        self.output_size = None if output_size is None else _pair(output_size)
        self.order = order
        self.p = p
        self.sigma = _pair(sigma)
        self.alpha = _pair(alpha)

    def __call__(self, sample):

        if random.random() > self.p:
            # Nothing to do
            return sample

        order = self.order
        if order is None:
            # Randomly sample 1, 2 or 3
            order = random.randint(1, 3)

        kind = self.order2kind[order]

        # Randomly re-sample x and y
        nx = len(sample["timestamps"])
        ny = len(sample["depths"])
        output_size = self.output_size
        if output_size is None:
            output_size = (nx, ny)
        if nx < 2 or ny < 2:
            raise ValueError("Input image shape ({}, {}) is too small".format(nx, ny))
        if output_size[0] < 2 or output_size[1] < 2:
            raise ValueError("Output image shape {} is too small".format(output_size))

        x_out = np.linspace(0, nx - 1, output_size[0])
        x_intv = x_out[1] - x_out[0]
        dx = x_intv * 2 * (np.random.rand(*x_out.shape) - 0.5)
        dx = scipy.ndimage.filters.gaussian_filter1d(dx, self.sigma[0], cval=0)
        dx *= self.alpha[0] * nx
        x_out += dx

        y_out = np.linspace(0, ny - 1, output_size[1])
        y_intv = y_out[1] - y_out[0]
        dy = y_intv * 2 * (np.random.rand(*y_out.shape) - 0.5)
        dy = scipy.ndimage.filters.gaussian_filter1d(dy, self.sigma[1], cval=0)
        dy *= self.alpha[1] * ny
        y_out += dy

        x_out = x_out.clip(0, nx - 1)
        y_out = y_out.clip(0, ny - 1)
        x_out = np.sort(x_out)
        y_out = np.sort(y_out)

        # 2D arrays (image-like)
        for key in _fields_2d:
            if key not in sample:
                continue
            _dtype = sample[key].dtype
            sample[key] = np.asarray(sample[key])
            if order == 0:
                if sample[key].shape != (nx, ny):
                    raise ValueError(
                        'Expected sample["{}"] to be shaped {}'.format(key, (nx, ny))
                    )
                sample[key] = sample[key][np.round(x_out).astype(int)][
                    :, np.round(y_out).astype(int)
                ]
            else:
                sample[key] = scipy.interpolate.RectBivariateSpline(
                    np.linspace(0, nx - 1, sample[key].shape[0]),
                    np.linspace(0, ny - 1, sample[key].shape[1]),
                    sample[key].astype(np.float),
                    kx=order,
                    ky=order,
                )(x_out, y_out)
            sample[key] = sample[key].astype(_dtype)

        # 1D arrays (column-like)
        for key in _fields_1d_timelike:
            if key not in sample:
                continue
            _kind = "linear" if key == "timestamps" else kind
            if key in {"is_passive", "is_removed"} and order > 1:
                _kind = "linear"
            _dtype = sample[key].dtype
            sample[key] = scipy.interpolate.interp1d(
                np.linspace(0, nx - 1, len(sample[key])),
                sample[key],
                kind=_kind,
            )(x_out)
            sample[key] = sample[key].astype(_dtype)

        # 1D arrays (row-like)
        for key in _fields_1d_depthlike:
            if key not in sample:
                continue
            _kind = "linear" if key == "depths" else kind
            _dtype = sample[key].dtype
            sample[key] = scipy.interpolate.interp1d(
                np.linspace(0, ny - 1, len(sample[key])),
                sample[key],
                kind=_kind,
            )(y_out)
            sample[key] = sample[key].astype(_dtype)

        return sample


class Normalize(object):
    """
    Normalize offset and scaling of image (mean and standard deviation).

    Note that changes are made inplace.

    Parameters
    ----------
    center : {"mean", "median", "pc10"} or float
        If a float, a pre-computed centroid measure of the distribution of
        samples, such as the pixel mean. If a string, a method to use to
        determine the center value.
    deviation : {"stdev", "mad", "iqr", "idr", "i7r"} or float
        If a float, a pre-computed deviation measure of the distribution of
        samples. If a string, a method to use to determine the deviation.
    robust2stdev : bool, optional
        Whether to convert robust measures to estimates of the standard
        deviation. Default is `True`.
    """

    def __init__(self, center, deviation, robust2stdev=True):
        self.center = center
        self.deviation = deviation
        self.robust2stdev = robust2stdev

    def __call__(self, sample):

        if not isinstance(self.center, str):
            center = self.center
        elif self.center.lower() == "mean":
            center = np.nanmean(sample["signals"])
        elif self.center.lower() == "median":
            center = np.nanmedian(sample["signals"])
        elif self.center.lower() == "pc10":
            center = np.nanpercentile(sample["signals"], 10)
        else:
            raise ValueError("Unrecognised center method: {}".format(self.center))

        if not isinstance(self.deviation, str):
            deviation = self.deviation
        elif self.deviation.lower() in {"std", "stdev"}:
            deviation = np.nanstd(sample["signals"])
        elif self.deviation.lower() == "mad":
            deviation = np.nanmedian(
                np.abs(sample["signals"] - np.nanmedian(sample["signals"]))
            )
            if self.robust2stdev:
                deviation *= 1.4826
        elif self.deviation.lower() == "iqr":
            deviation = np.diff(np.nanpercentile(sample["signals"], [25, 75])).item()
            if self.robust2stdev:
                deviation /= 1.35
        elif self.deviation.lower() == "idr":
            deviation = np.diff(np.nanpercentile(sample["signals"], [10, 90])).item()
            if self.robust2stdev:
                deviation /= 2.56
        elif self.deviation.lower() == "i7r":
            deviation = np.diff(np.nanpercentile(sample["signals"], [7, 93])).item()
            if self.robust2stdev:
                deviation /= 3.0
        else:
            raise ValueError("Unrecognised deviation method: {}".format(self.deviation))

        sample["signals"] -= center
        sample["signals"] /= deviation

        return sample


class ReplaceNan(object):
    """
    Replace NaNs with a finite float value.

    Parameters
    ----------
    nan_val : float, optional
        Value to replace NaNs with. Default is `0.0`.
    """

    def __init__(self, nan_val=0.0):
        self.nan_val = nan_val

    def __call__(self, sample):

        # Can't use np.nan_to_num to assign nan to a specific value if
        # numpy version <= 1.17.
        sample["signals"][np.isnan(sample["signals"])] = self.nan_val

        return sample


class RandomReflection(object):
    """
    Randomly reflect a sample.

    Parameters
    ----------
    axis : int, optional
        Axis to reflect. Default is 0.
    p : float, optional
        Probability of reflection. Default is 0.5.
    """

    def __init__(self, axis=0, p=0.5):
        self.axis = axis
        self.p = p

    def __call__(self, sample):

        if random.random() > self.p:
            # Nothing to do
            return sample

        # Reflect data
        for key in _fields_2d + _fields_1d_timelike:
            if key in sample:
                sample[key] = np.flip(sample[key], self.axis).copy()

        return sample


class RandomCropWidth(object):
    """
    Randomly crop a sample in the width dimension.

    Parameters
    ----------
    max_crop_fraction : float
        Maximum amount of material to crop away, as a fraction of the total
        width. The `crop_fraction` will be sampled uniformly from the range
        `[0, max_crop_fraction]`. The crop is always centred.
    """

    def __init__(self, max_crop_fraction):
        self.max_crop_fraction = max_crop_fraction

    def __call__(self, sample):

        width = sample["signals"].shape[0]

        crop_fraction = random.uniform(0.0, self.max_crop_fraction)
        crop_amount = crop_fraction * width

        lft = int(crop_amount / 2)
        rgt = lft + width - int(crop_amount)

        # Crop data
        for key in _fields_2d + _fields_1d_timelike:
            if key in sample:
                sample[key] = sample[key][lft:rgt]

        return sample


def optimal_crop_depth(transect):
    """
    Crop a sample depthwise to contain only the space between highest surface
    and deepest seafloor.

    Parameters
    ----------
    transect : dict
        Transect dictionary.
    """

    d0 = np.min(transect["depths"])

    depth_intv = abs(transect["depths"][1] - transect["depths"][0])
    shallowest_depth = None
    if transect["is_upward_facing"]:
        for key in ("d_surface", "surface", "d_surf", "surf"):
            if key not in transect:
                continue
            surf_options = transect[key][transect[key] > d0]
            if len(surf_options) == 0:
                continue
            d = np.min(surf_options)
            if shallowest_depth is None:
                shallowest_depth = d
            else:
                shallowest_depth = min(d, shallowest_depth)
    if shallowest_depth is None:
        shallowest_depth = d0
    shallowest_depth -= 5 * depth_intv

    deepest_depth = None
    if not transect["is_upward_facing"]:
        for key in (
            "d_bottom-original",
            "d_bot-original",
            "d_bottom",
            "d_bot",
            "bottom-original",
            "bottom",
        ):
            if key not in transect:
                continue
            d = np.max(transect[key])
            if deepest_depth is None:
                deepest_depth = d
            else:
                deepest_depth = max(d, deepest_depth)
    if deepest_depth is None:
        deepest_depth = np.max(transect["depths"])
    deepest_depth += 5 * depth_intv

    if shallowest_depth >= deepest_depth:
        return transect

    depth_mask = (shallowest_depth <= transect["depths"]) & (
        transect["depths"] <= deepest_depth
    )

    for key in _fields_1d_depthlike:
        if key in transect:
            transect[key] = transect[key][depth_mask]

    for key in _fields_2d:
        if key in transect:
            transect[key] = transect[key][:, depth_mask]

    return transect


class OptimalCropDepth(object):
    """
    A transform which crops a sample depthwise to contain only the space
    between highest surface and deepest seafloor.
    """

    def __call__(self, sample):
        return optimal_crop_depth(sample)


class RandomCropDepth(object):
    """
    Randomly crop a sample depthwise.

    Parameters
    ----------
    p_crop_is_none : float, optional
        Probability of not doing any crop.
        Default is `0.1`.
    p_crop_is_optimal : float, optional
        Probability of doing an "optimal" crop, running `optimal_crop_depth`.
        Default is `0.1`.
    p_crop_is_close : float, optional
        Probability of doing crop which is zoomed in and close to the "optimal"
        crop, running `optimal_crop_depth`. Default is `0.4`.
        If neither no crop, optimal, nor close-to-optimal crop is selected,
        the crop is randomly sized over the full extent of the range of depths.
    p_nearfield_side_crop : float, optional
        Probability that the nearfield side is cropped.
        Default is `0.5`.
    fraction_close : float, optional
        Fraction by which crop is increased/decreased in either direction when
        doing a close to optimal crop.
        Default is `0.25`.
    """

    def __init__(
        self,
        p_crop_is_none=0.1,
        p_crop_is_optimal=0.1,
        p_crop_is_close=0.4,
        p_nearfield_side_crop=0.5,
        fraction_close=0.25,
    ):
        self.p_crop_is_none = p_crop_is_none
        self.p_crop_is_optimal = p_crop_is_optimal
        self.p_crop_is_close = p_crop_is_close
        self.p_nearfield_side_crop = p_nearfield_side_crop
        self.fraction_close = fraction_close

    def __call__(self, sample):

        p = random.random()

        # Possibility of doing no crop at all
        p -= self.p_crop_is_none
        if p < 0:
            return sample

        # Possibility of doing an optimal crop
        p -= self.p_crop_is_optimal
        if p < 0:
            return optimal_crop_depth(sample)

        depth_intv = abs(sample["depths"][1] - sample["depths"][0])

        lim_top_shallowest = np.min(sample["depths"])
        lim_top_deepest = max(
            lim_top_shallowest + depth_intv, np.min(sample["d_bottom"]) - depth_intv
        )
        lim_bot_deepest = np.max(sample["depths"])
        lim_bot_shallowest = min(
            lim_bot_deepest - depth_intv,
            max(np.max(sample["d_turbulence"]), np.min(sample["d_bottom"])),
        )

        if sample["is_upward_facing"]:
            surf_options = sample["d_surface"][sample["d_surface"] > lim_top_shallowest]
            if len(surf_options) == 0:
                opt_top_depth = lim_top_shallowest
            else:
                opt_top_depth = max(lim_top_shallowest, np.min(surf_options))
            opt_bot_depth = lim_bot_deepest
        else:
            opt_top_depth = lim_top_shallowest
            opt_bot_depth = np.max(sample["d_bottom-original"])

        depth_range = abs(opt_bot_depth - opt_top_depth)
        close_dist_grow = self.fraction_close * depth_range
        close_dist_shrink = (
            depth_range * self.fraction_close / (1 + self.fraction_close)
        )

        close_top_shallowest = min(
            lim_top_deepest, max(lim_top_shallowest, opt_top_depth - close_dist_grow)
        )
        close_top_deepest = min(
            lim_top_deepest,
            opt_top_depth + close_dist_shrink,
            np.percentile(sample["d_turbulence"], 25),
        )
        if sample["is_upward_facing"]:
            close_bot_shallowest = max(
                lim_bot_shallowest,
                opt_bot_depth - close_dist_shrink,
            )
        else:
            close_bot_shallowest = max(
                lim_bot_shallowest,
                opt_bot_depth - close_dist_shrink,
                np.percentile(sample["d_bottom-original"], 50),
            )
        close_bot_shallowest = min(lim_bot_deepest, close_bot_shallowest)
        close_bot_deepest = min(
            lim_bot_deepest,
            max(lim_bot_shallowest, opt_bot_depth) + close_dist_grow,
        )

        if (
            close_top_shallowest > close_top_deepest
            or close_bot_shallowest > close_bot_deepest
            or close_top_deepest >= close_bot_shallowest
        ):
            raise ValueError(
                "Nonsensical depth limits:\n"
                "  opt_top_depth        = {:7.3f}\n"
                "  opt_bot_depth        = {:7.3f}\n"
                "  close_top_shallowest = {:7.3f}\n"
                "  close_top_deepest    = {:7.3f}\n"
                "  close_bot_shallowest = {:7.3f}\n"
                "  close_bot_deepest    = {:7.3f}\n".format(
                    opt_top_depth,
                    opt_bot_depth,
                    close_top_shallowest,
                    close_top_deepest,
                    close_bot_shallowest,
                    close_bot_deepest,
                )
            )

        rand_top_shallowest = lim_top_shallowest
        rand_top_deepest = min(
            lim_top_deepest,
            max(
                np.percentile(sample["d_turbulence"], 50),
                opt_top_depth + close_dist_shrink,
            ),
        )
        rand_top_deepest = max(lim_top_shallowest, rand_top_deepest)
        if sample["is_upward_facing"]:
            rand_bot_shallowest = close_bot_shallowest
        else:
            rand_bot_shallowest = max(
                lim_bot_shallowest,
                np.percentile(sample["d_bottom-original"], 50),
            )
        rand_bot_shallowest = min(lim_bot_deepest, rand_bot_shallowest)
        rand_bot_deepest = lim_bot_deepest

        if (
            rand_top_shallowest > rand_top_deepest
            or rand_bot_shallowest > rand_bot_deepest
            or rand_top_deepest >= rand_bot_shallowest
        ):
            raise ValueError(
                "Nonsensical depth limits:\n"
                "  opt_top_depth       = {:7.3f}\n"
                "  opt_bot_depth       = {:7.3f}\n"
                "  rand_top_shallowest = {:7.3f}\n"
                "  rand_top_deepest    = {:7.3f}\n"
                "  rand_bot_shallowest = {:7.3f}\n"
                "  rand_bot_deepest    = {:7.3f}\n".format(
                    opt_top_depth,
                    opt_bot_depth,
                    rand_top_shallowest,
                    rand_top_deepest,
                    rand_bot_shallowest,
                    rand_bot_deepest,
                )
            )

        # Select whether to do a close or fully random crop
        if p < self.p_crop_is_close:
            # Close crop
            close_crop = True
            top_shallowest = close_top_shallowest
            top_deepest = close_top_deepest
            bot_shallowest = close_bot_shallowest
            bot_deepest = close_bot_deepest
            # Select limits of depth range. Equally likely to expand or contract
            # in each direction.
            if random.random() < 0.5:
                shallowest_depth = random.uniform(top_shallowest, opt_top_depth)
            else:
                shallowest_depth = random.uniform(opt_top_depth, top_deepest)

            if random.random() < 0.5:
                deepest_depth = random.uniform(bot_shallowest, opt_bot_depth)
            else:
                deepest_depth = random.uniform(opt_bot_depth, bot_deepest)
        else:
            # Random crop
            close_crop = False
            top_shallowest = rand_top_shallowest
            top_deepest = rand_top_deepest
            bot_shallowest = rand_bot_shallowest
            bot_deepest = rand_bot_deepest
            # Select limits of depth range. Either randomly select uniformly
            # across the whole range, or not.
            if (
                sample["is_upward_facing"]
                or random.random() < self.p_nearfield_side_crop
            ):
                shallowest_depth = random.uniform(top_shallowest, top_deepest)
            else:
                shallowest_depth = opt_top_depth
            if (
                not sample["is_upward_facing"]
                or random.random() < self.p_nearfield_side_crop
            ):
                deepest_depth = random.uniform(bot_shallowest, bot_deepest)
            else:
                deepest_depth = opt_bot_depth

        # Crop data
        depth_crop_mask = (shallowest_depth <= sample["depths"]) & (
            sample["depths"] <= deepest_depth
        )

        for key in _fields_1d_depthlike:
            if key in sample:
                sample[key] = sample[key][depth_crop_mask]

        for key in _fields_2d:
            if key in sample:
                sample[key] = sample[key][:, depth_crop_mask]

        return sample


class ColorJitter(object):
    """
    Randomly change the brightness and contrast of a normalized image.

    Note that changes are made inplace.

    Parameters
    ----------
    brightness : float or tuple of float (min, max)
        How much to jitter brightness. `brightness_factor` is chosen uniformly
        from `[-brightness, brightness]` or the given `[min, max]`.
        `brightness_factor` is then added to the image.
    contrast : float or tuple of float (min, max)
        How much to jitter contrast. `contrast_factor` is chosen uniformly from
        `[max(0, 1 - contrast), 1 + contrast]` or the given `[min, max]`.
        Should be non negative numbers.
    """

    def __init__(self, brightness=0, contrast=0):
        self.brightness = self._check_input(
            brightness,
            "brightness",
            center=0,
            bound=(float("-inf"), float("inf")),
            clip_first_on_zero=False,
        )
        self.contrast = self._check_input(contrast, "contrast")

    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, (float, int)):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with length 2.".format(
                    name
                )
            )

        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, sample):
        init_op = random.randint(0, 1)
        for i_op in range(2):
            op_num = (init_op + i_op) % 2
            if op_num == 0 and self.brightness is not None:
                brightness_factor = random.uniform(
                    self.brightness[0], self.brightness[1]
                )
                sample["signals"] += brightness_factor
            elif op_num == 1 and self.contrast is not None:
                contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
                sample["signals"] *= contrast_factor
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "brightness={0}".format(self.brightness)
        format_string += ", contrast={0})".format(self.contrast)
        format_string += ")"
        return format_string
