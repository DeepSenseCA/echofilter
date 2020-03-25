'''
General utility functions.
'''

import numpy as np


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
