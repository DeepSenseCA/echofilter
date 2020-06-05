import collections
import os
import random

import numpy as np
import skimage.transform


_fields_2d = (
    'signals', 'mask',
    'mask_top', 'mask_bot',
    'mask_top-original', 'mask_bot-original',
    'mask_surf',
    'mask_patches', 'mask_patches-original', 'mask_patches-ntob',
)
_fields_1d_timelike = (
    'timestamps',
    'd_top', 'd_bot', 'r_top', 'r_bot',
    'd_top-original', 'd_bot-original', 'r_top-original', 'r_bot-original',
    'd_surf', 'r_surf',
    'is_passive', 'is_removed',
)
_fields_1d_depthlike = ('depths', )
_fields_0d = ('is_upward_facing', )


class Rescale(object):
    '''
    Rescale the image(s) in a sample to a given size.

    Parameters
    ----------
    output_size : tuple or int
        Desired output size. If tuple, output is matched to output_size. If
        int, output is square.
    '''

    def __init__(self, output_size):
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        elif isinstance(output_size, collections.Sequence):
            output_size = tuple(output_size)
        else:
            raise ValueError('Output size must be an int or a tuple.')
        self.output_size = output_size

    def __call__(self, sample):

        # 2D arrays (image-like)
        for key in _fields_2d:
            if key in sample:
                sample[key] = skimage.transform.resize(
                    np.asarray(sample[key]).astype(np.float),
                    self.output_size,
                    clip=False,
                    preserve_range=False,
                )

        # 1D arrays (column-like)
        for key in _fields_1d_timelike:
            if key in sample:
                sample[key] = np.interp(
                    np.linspace(0, len(sample[key]) - 1, self.output_size[0]),
                    np.linspace(0, len(sample[key]) - 1, len(sample[key])),
                    sample[key],
                )

        # 1D arrays (row-like)
        for key in _fields_1d_depthlike:
            if key in sample:
                sample[key] = np.interp(
                    np.linspace(0, len(sample[key]) - 1, self.output_size[1]),
                    np.linspace(0, len(sample[key]) - 1, len(sample[key])),
                    sample[key],
                )

        return sample


class Normalize(object):
    '''
    Normalize mean and standard deviation of image.

    Note that changes are made inplace.

    Parameters
    ----------
    mean : float
        Expected sample pixel mean.
    stdev : float
        Expected sample standard deviation of pixel intensities.
    '''

    def __init__(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev

    def __call__(self, sample):

        sample['signals'] -= self.mean
        sample['signals'] /= self.stdev

        return sample


class ReplaceNan(object):
    '''
    Replace NaNs with a finite float value.

    Parameters
    ----------
    nan_val : float, optional
        Value to replace NaNs with. Default is `0.0`.
    '''

    def __init__(self, nan_val=0.0):
        self.nan_val = nan_val

    def __call__(self, sample):

        # Can't use np.nan_to_num to assign nan to a specific value if
        # numpy version <= 1.17.
        sample['signals'][np.isnan(sample['signals'])] = self.nan_val

        return sample


class RandomReflection(object):
    '''
    Randomly reflect a sample.

    Parameters
    ----------
    axis : int, optional
        Axis to reflect. Default is 0.
    p : float, optional
        Probability of reflection. Default is 0.5.
    '''

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
    '''
    Randomly crop a sample in the width dimension.

    Parameters
    ----------
    max_crop_fraction : float
        Maximum amount of material to crop away, as a fraction of the total
        width. The `crop_fraction` will be sampled uniformly from the range
        `[0, max_crop_fraction]`. The crop is always centred.
    '''

    def __init__(self, max_crop_fraction):
        self.max_crop_fraction = max_crop_fraction

    def __call__(self, sample):

        width = sample['signals'].shape[0]

        crop_fraction = random.uniform(0., self.max_crop_fraction)
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
    shallowest_depth = np.min(transect["depths"])
    if transect["is_upward_facing"]:
        for key in ("d_surf", "surface"):
            if key not in transect:
                continue
            surf_options = transect[key][transect[key] > shallowest_depth]
            if len(surf_options) > 0:
                shallowest_depth = np.min(surf_options)
                break

    deepest_depth = np.max(transect["depths"])
    if not transect["is_upward_facing"]:
        for key in ("d_bot-original", "d_bot", "bottom-original", "bottom"):
            if key not in transect:
                continue
            deepest_depth = np.max(transect[key])
            break

    if shallowest_depth >= deepest_depth:
        return transect

    depth_mask = (shallowest_depth <= transect["depths"]) & (transect["depths"] <= deepest_depth)

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
        p_crop_is_none=.1,
        p_crop_is_optimal=.1,
        p_crop_is_close=.4,
        p_nearfield_side_crop=.5,
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
        lim_top_deepest = max(lim_top_shallowest, np.min(sample["d_bot"]) - depth_intv)
        lim_bot_shallowest = max(np.max(sample["d_top"]), np.min(sample["d_bot"]))
        lim_bot_deepest = np.max(sample["depths"])

        if sample["is_upward_facing"]:
            surf_options = sample["d_surf"][sample["d_surf"] > lim_top_shallowest]
            if len(surf_options) == 0:
                opt_top_depth = lim_top_shallowest
            else:
                opt_top_depth = max(lim_top_shallowest, np.min(surf_options))
            opt_bot_depth = lim_bot_deepest
        else:
            opt_top_depth = lim_top_shallowest
            opt_bot_depth = np.max(sample["d_bot-original"])

        depth_range = abs(opt_bot_depth - opt_top_depth)
        close_dist_grow = self.fraction_close * depth_range
        close_dist_shrink = depth_range * self.fraction_close / (1 + self.fraction_close)

        close_top_shallowest = max(lim_top_shallowest, opt_top_depth - close_dist_grow)
        close_top_deepest = min(
            lim_top_deepest,
            opt_top_depth + close_dist_shrink,
            np.percentile(sample["d_top"], 25),
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
                np.percentile(sample["d_bot"], 50),
            )
        close_bot_deepest = min(lim_bot_deepest, opt_bot_depth + close_dist_grow)

        if (
            close_top_shallowest > close_top_deepest or
            close_bot_shallowest > close_bot_deepest or
            close_top_deepest >= close_bot_shallowest
        ):
            raise ValueError(
                "Nonsensical depth limits:\n"
                "  opt_top_depth        = {:7.3f}\n"
                "  opt_bot_depth        = {:7.3f}\n"
                "  close_top_shallowest = {:7.3f}\n"
                "  close_top_deepest    = {:7.3f}\n"
                "  close_bot_shallowest = {:7.3f}\n"
                "  close_bot_deepest    = {:7.3f}\n"
                .format(
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
            max(np.percentile(sample["d_top"], 50), opt_top_depth + close_dist_shrink),
        )
        if sample["is_upward_facing"]:
            rand_bot_shallowest = close_bot_shallowest
        else:
            rand_bot_shallowest = max(
                lim_bot_shallowest,
                np.percentile(sample["d_bot-original"], 50),
            )
        rand_bot_deepest = lim_bot_deepest

        if (
            rand_top_shallowest > rand_top_deepest or
            rand_bot_shallowest > rand_bot_deepest or
            rand_top_deepest >= rand_bot_shallowest
        ):
            raise ValueError(
                "Nonsensical depth limits:\n"
                "  opt_top_depth       = {:7.3f}\n"
                "  opt_bot_depth       = {:7.3f}\n"
                "  rand_top_shallowest = {:7.3f}\n"
                "  rand_top_deepest    = {:7.3f}\n"
                "  rand_bot_shallowest = {:7.3f}\n"
                "  rand_bot_deepest    = {:7.3f}\n"
                .format(
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
            if sample["is_upward_facing"] or random.random() < self.p_nearfield_side_crop:
                shallowest_depth = random.uniform(top_shallowest, top_deepest)
            else:
                shallowest_depth = opt_top_depth
            if not sample["is_upward_facing"] or random.random() < self.p_nearfield_side_crop:
                deepest_depth = random.uniform(bot_shallowest, bot_deepest)
            else:
                deepest_depth = opt_bot_depth

        # Crop data
        depth_crop_mask = (shallowest_depth <= sample["depths"]) & (sample["depths"] <= deepest_depth)

        for key in _fields_1d_depthlike:
            if key in sample:
                sample[key] = sample[key][depth_crop_mask]

        for key in _fields_2d:
            if key in sample:
                sample[key] = sample[key][:, depth_crop_mask]

        return sample


class ColorJitter(object):
    '''
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
    '''
    def __init__(self, brightness=0, contrast=0):
        self.brightness = self._check_input(
            brightness,
            'brightness',
            center=0,
            bound=(float('-inf'), float('inf')),
            clip_first_on_zero=False,
        )
        self.contrast = self._check_input(contrast, 'contrast')

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, (float, int)):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, sample):
        init_op = random.randint(0, 1)
        for i_op in range(2):
            op_num = (init_op + i_op) % 2
            if op_num == 0 and self.brightness is not None:
                brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
                sample['signals'] += brightness_factor
            elif op_num == 1 and self.contrast is not None:
                contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
                sample['signals'] *= contrast_factor
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0})'.format(self.contrast)
        format_string += ')'
        return format_string
