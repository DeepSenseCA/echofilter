"""
echofilter.nn utility functions.
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

import math
import numbers
import random

import numpy as np
import torch


def logavgexp(
    input, dim, keepdim=False, temperature=None, internal_dtype=torch.float32
):
    """
    Returns the log of meaned exponentials of each row of the `input` tensor in
    the given dimension `dim`. The computation is numerically stabilized.

    If `keepdim` is `True`, the output tensor is of the same size as `input`
    except in the dimension `dim` where it is of size `1`. Otherwise, `dim` is
    squeezed (see :meth:`torch.squeeze()`), resulting in the output tensor
    having 1 fewer dimension.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor.
    dim : int
        The dimension to reduce.
    keepdim : bool, optional
        Whether the output tensor has `dim` retained or not.
        Default is `False`.
    temperature : float or None, optional
        A temperature which is applied to the logits. Temperatures must be
        positive. Temperatures greater than `1` make the result closer to the
        average of `input`, whilst temperatures `0<t<1` make the result closer
        to the maximum of `input`. If `None` (default) or `1`, no temperature
        is applied.
    internal_dtype : torch.dtype, optional
        A data type which the `input` will be cast as before computing the
        log-sum-exp step. Default is :attr:`torch.float32`.

    Returns
    -------
    torch.Tensor
        The log-average-exp of `input`.
    """

    if isinstance(temperature, numbers.Number) and temperature == 1:
        temperature = None

    input_dtype = input.dtype

    if internal_dtype is not None:
        input = input.to(internal_dtype)

    if isinstance(temperature, torch.Tensor):
        temperature = temperature.to(input.dtype)

    if temperature is not None:
        input = input.div(temperature)

    log_n = math.log(input.shape[dim])  # TODO: can be cached
    lae = torch.logsumexp(input, dim=dim, keepdim=True).sub(log_n)

    if temperature is not None:
        lae = lae.mul(temperature)

    if not keepdim:
        lae = lae.squeeze(dim)

    return lae.to(input_dtype)


class TensorDict(torch.nn.ParameterDict):
    r"""Holds tensors in a dictionary.

    TensorDict can be indexed like a regular Python dictionary, but implements
    methods such as `to` which operate on all elements within it.

    :class:`TensorDict` is an **ordered** dictionary that respects

    - the order of insertion, and

    - in :meth:`~TensorDict.update`, the order of the merged ``OrderedDict``
      or another :class:`TensorDict` (the argument to :meth:`~TensorDict.update`).

    Note that :meth:`~TensorDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping.

    Arguments:
        parameters (iterable, optional): a mapping (dictionary) of
            (string : :class:`torch.Tensor`) or an iterable of key-value pairs
            of type (string, :class:`torch.Tensor`)
    """

    def __init__(self, tensors=None):
        super(TensorDict, self).__init__(tensors)

    def __setitem__(self, key, parameter):
        r"""Adds a tensor to the module.

        The parameter can be accessed as an attribute using given key.

        Args:
            key (string): key of the parameter. The parameter can be accessed
                from this module using the given key
            parameter (Parameter): parameter to be added to the module.
        """
        if "_parameters" not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call"
            )

        elif not isinstance(key, torch._six.string_classes):
            raise TypeError(
                "parameter key should be a string. " "Got {}".format(torch.typekey(key))
            )
        elif "." in key:
            raise KeyError('parameter key can\'t contain "."')
        elif key == "":
            raise KeyError('parameter key can\'t be empty string ""')
        elif hasattr(self, key) and key not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(key))

        if parameter is None:
            self._parameters[key] = None
        else:
            self._parameters[key] = parameter

    def detach(self):
        out = TensorDict()
        for k, p in self._parameters.items():
            out[k] = p.detach()
        return out

    def detach_(self):
        for k, p in self._parameters.items():
            self._parameters[k] = p.detach()
        return self._parameters

    def extra_repr(self):
        child_lines = []
        for k, p in self._parameters.items():
            size_str = "x".join(str(size) for size in p.size())
            device_str = "" if not p.is_cuda else " (GPU {})".format(p.get_device())
            parastr = "{} containing: [{} of size {}{}]".format(
                p.__class__, torch.typename(p.data), size_str, device_str
            )
            child_lines.append("  (" + k + "): " + parastr)
        tmpstr = "\n".join(child_lines)
        return tmpstr


def count_parameters(model, only_trainable=True):
    r"""
    Count the number of (trainable) parameters within a model and its children.

    Arguments:
        model (torch.nn.Model): the model.
        only_trainable (bool, optional): indicates whether the count should be restricted
            to only trainable parameters (ones which require grad), otherwise all
            parameters are included. Default is ``True``.

    Returns:
        int: total number of (trainable) parameters possessed by the model.
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def seed_all(seed=None, only_current_gpu=False, mirror_gpus=False):
    r"""
    Initialises the random number generators for random, numpy, and both CPU and GPU(s)
    for torch.

    Arguments:
        seed (int, optional): seed value to use for the random number generators.
            If :attr:`seed` is ``None`` (default), seeds are picked at random using
            the methods built in to each RNG.
        only_current_gpu (bool, optional): indicates whether to only re-seed the current
            cuda device, or to seed all of them. Default is ``False``.
        mirror_gpus (bool, optional): indicates whether all cuda devices should receive
            the same seed, or different seeds. If :attr:`mirror_gpus` is ``False`` and
            :attr:`seed` is not ``None``, each device receives a different but
            deterministically determined seed. Default is ``False``.

    Note that we override the settings for the cudnn backend whenever this function is
    called. If :attr:`seed` is not ``None``, we set::

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    in order to ensure experimental results behave deterministically and are repeatible.
    However, enabling deterministic mode may result in an impact on performance. See
    `link`_ for more details. If :attr:`seed` is ``None``, we return the cudnn backend
    to its performance-optimised default settings of::

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    .. _link:
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    # Note that random, np.random and torch's RNG all have different
    # implementations so they will produce different numbers even with
    # when they are seeded the same.

    # Seed Python's built-in random number generator
    random.seed(seed)
    # Seed numpy's random number generator
    np.random.seed(seed)

    def get_seed():
        """
        On Python 3.2 and above, and when system sources of randomness are
        available, use `os.urandom` to make a new seed. Otherwise, use the
        current time.
        """
        try:
            import os

            # Use system's source of entropy (on Linux, syscall `getrandom()`)
            s = int.from_bytes(os.urandom(4), byteorder="little")
        except AttributeError:
            from datetime import datetime

            # Get the current time in mircoseconds, and map to an integer
            # in the range [0, 2**32)
            s = (
                int(
                    (datetime.utcnow() - datetime(1970, 1, 1)).total_seconds() * 1000000
                )
                % 4294967296
            )
        return s

    # Seed pytorch's random number generator on the CPU
    # torch doesn't support a None argument, so we have to source our own seed
    # with high entropy if none is given.
    s = seed if seed is not None else get_seed()
    torch.manual_seed(s)

    if seed is None:
        # Since seeds are random, we don't care about determinism and
        # will set the backend up for optimal performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        # Ensure cudNN is deterministic, so the results are consistent
        # for this seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Seed pytorch's random number generator on the GPU(s)
    if only_current_gpu:
        # Only re-seed the current GPU
        if mirror_gpus:
            # ... re-seed with the same as the CPU seed
            torch.cuda.manual_seed(s)
        elif seed is None:
            # ... re-seed at random, however pytorch deems fit
            torch.cuda.seed()
        else:
            # ... re-seed with a deterministic seed based on, but
            # not equal to, the CPU seed
            torch.cuda.manual_seed((seed + 1) % 4294967296)
    elif mirror_gpus:
        # Seed multiple GPUs, each with the same seed
        torch.cuda.manual_seed_all(s)
    elif seed is None:
        # Seed multiple GPUs, all with unique seeds
        # ... a random seed for each GPU, however pytorch deems fit
        torch.cuda.seed_all()
    else:
        # Seed multiple GPUs, all with unique seeds
        # ... different deterministic seeds for each GPU
        # We assign the seeds in ascending order, and can't exceed the
        # random state's maximum value of 2**32 == 4294967296
        for device in range(torch.cuda.device_count()):
            with torch.cuda.device(device):
                torch.cuda.manual_seed((seed + 1 + device) % 4294967296)
