"""
echofilter.nn utility functions.
"""

import math
import numbers

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
    squeezed (see `torch.squeeze()`), resulting in the output tensor having 1
    fewer dimension.

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
        log-sum-exp step. Default is `torch.float32`.

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

    :class:`~TensorDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~TensorDict.update`, the order of the merged ``OrderedDict``
      or another :class:`~TensorDict` (the argument to :meth:`~TensorDict.update`).

    Note that :meth:`~TensorDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping.

    Arguments:
        parameters (iterable, optional): a mapping (dictionary) of
            (string : :class:`~torch.Tensor`) or an iterable of key-value pairs
            of type (string, :class:`~torch.Tensor`)
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