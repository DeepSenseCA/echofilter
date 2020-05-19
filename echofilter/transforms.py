import collections
import os
import random

import numpy as np
import scipy.interpolate
import skimage.transform


_fields_2d = (
    'signals', 'mask',
    'mask_top', 'mask_bot',
    'mask_top-original', 'mask_bot-original',
    'mask_surf',
    'mask_patches', 'mask_patches-original',
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
    order : int or None, optional
        Order of the interpolation, for both image and vector elements.
        For images-like components, the interpolation is 2d.
        The following values are supported:

        - 0: Nearest-neighbor
        - 1: Linear (default)
        - 2: Quadratic
        - 3: Cubic

        If `None`, the order is randomly selected from these values.
    '''

    order2kind = {
        0: 'nearest',
        1: 'linear',
        2: 'quadratic',
        3: 'cubic',
    }

    def __init__(self, output_size, order=1):
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        elif isinstance(output_size, collections.Sequence):
            output_size = tuple(output_size)
        else:
            raise ValueError('Output size must be an int or a tuple.')
        self.output_size = output_size
        self.order = order

    def __call__(self, sample):

        order = self.order
        if order is None:
            order = np.random.randint(4)

        kind = self.order2kind[order]

        # 2D arrays (image-like)
        for key in _fields_2d:
            if key not in sample:
                continue
            sample[key] = skimage.transform.resize(
                np.asarray(sample[key]).astype(np.float),
                self.output_size,
                order=order,
                clip=False,
                preserve_range=False,
            )

        # 1D arrays (column-like)
        for key in _fields_1d_timelike:
            if key not in sample:
                continue
            _kind = 'linear' if key == 'timestamps' else kind
            sample[key] = scipy.interpolate.interp1d(
                np.arange(len(sample[key])),
                sample[key],
                kind=_kind,
            )(np.linspace(0, len(sample[key]) - 1, self.output_size[0]))

        # 1D arrays (row-like)
        for key in _fields_1d_depthlike:
            if key not in sample:
                continue
            _kind = 'linear' if key == 'depths' else kind
            sample[key] = scipy.interpolate.interp1d(
                np.arange(len(sample[key])),
                sample[key],
                kind=_kind,
            )(np.linspace(0, len(sample[key]) - 1, self.output_size[1]))

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


class RandomStretchDepth(object):
    '''
    Rescale a set of images in a sample to a given size.

    Note that this transform doesn't change images, just the `depth`, `d_top`,
    and `d_bot`.
    Note that changes are made inplace.

    Parameters
    ----------
    max_factor : float
        Maximum stretch factor. A number between `[1, 1 + max_factor]` will be
        generated, and the depth will either be divided or multiplied by the
        generated stretch factor.
    expected_bottom_gap : float
        Expected gap between actual ocean floor and target bottom line.
    '''

    def __init__(self, max_factor, expected_bottom_gap=1):
        self.max_factor = max_factor
        self.expected_bottom_gap = expected_bottom_gap

    def __call__(self, sample):

        factor = random.uniform(1.0, 1.0 + self.max_factor)

        if random.random() > 0.5:
            factor = 1. / factor

        if not sample['is_upward_facing']:
            sample['d_bot'] += self.expected_bottom_gap
        for key in ('depths', 'd_top', 'd_bot', 'd_top-original', 'd_bot-original', 'd_surf'):
            sample[key] *= factor
        if not sample['is_upward_facing']:
            sample['d_bot'] -= self.expected_bottom_gap

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


class RandomCropTop(object):
    '''
    Randomly crop the top off a sample.

    Parameters
    ----------
    max_crop_fraction : float, optional
        Maximum amount of material to crop away, as a fraction of the total
        height. The crop depth will be sampled uniformly from the range
        of shallowest measure to shallowest point on the top line
        (assuming this is not deepr than the `max_crop_fraction`).
        If `None` (default), the crop is unlimited.
    '''

    def __init__(self, max_crop_fraction=None):
        self.max_crop_fraction = max_crop_fraction

    def __call__(self, sample):

        shallowest_measure = sample['depths'][0]
        shallowest_line = np.nanmin(sample['d_top'])
        if self.max_crop_fraction is None:
            deepest_crop = shallowest_line
        else:
            max_crop_depth = (
                sample['depths'][0] +
                (sample['depths'][-1] - sample['depths'][0]) * self.max_crop_fraction
            )
            deepest_crop = np.minimum(shallowest_line, max_crop_depth)

        crop_depth = random.uniform(shallowest_measure, deepest_crop)

        # Crop data
        depth_crop_mask = sample['depths'] >= crop_depth

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
