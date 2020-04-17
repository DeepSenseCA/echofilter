'''
General utility functions.
'''

import numpy as np
import torch
import torch.nn as nn


def first_nonzero(arr, axis, invalid_val=-1):
    '''
    Find the index of the first non-zero element in an array.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to search.
    axis : int
        Axis along which to search for a non-zero element.
    invalid_val : any, optional
        Value to return if all elements are zero. Default is `-1`.
    '''
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def last_nonzero(arr, axis, invalid_val=-1):
    '''
    Find the index of the last non-zero element in an array.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to search.
    axis : int
        Axis along which to search for a non-zero element.
    invalid_val : any, optional
        Value to return if all elements are zero. Default is `-1`.
    '''
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def get_current_lr(optimizer):
    return optimizer.param_groups[0]['lr']


class TensorDict(nn.ParameterDict):
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
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")

        elif not isinstance(key, torch._six.string_classes):
            raise TypeError("parameter key should be a string. "
                            "Got {}".format(torch.typekey(key)))
        elif '.' in key:
            raise KeyError("parameter key can't contain \".\"")
        elif key == '':
            raise KeyError("parameter key can't be empty string \"\"")
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
            size_str = 'x'.join(str(size) for size in p.size())
            device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
            parastr = '{} containing: [{} of size {}{}]'.format(
                p.__class__, torch.typename(p.data), size_str, device_str)
            child_lines.append('  (' + k + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr
