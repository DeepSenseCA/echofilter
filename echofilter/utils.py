"""
General utility functions.
"""

import argparse
import colorama
import contextlib

import numpy as np


def first_nonzero(arr, axis=-1, invalid_val=-1):
    """
    Find the index of the first non-zero element in an array.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to search.
    axis : int, optional
        Axis along which to search for a non-zero element. Default is `-1`.
    invalid_val : any, optional
        Value to return if all elements are zero. Default is `-1`.
    """
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def last_nonzero(arr, axis=-1, invalid_val=-1):
    """
    Find the index of the last non-zero element in an array.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to search.
    axis : int, optional
        Axis along which to search for a non-zero element. Default is `-1`.
    invalid_val : any, optional
        Value to return if all elements are zero. Default is `-1`.
    """
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def get_indicator_onoffsets(indicator):
    """
    Find the onsets and offsets of nonzero entries in an indicator.

    Parameters
    ----------
    indicator : 1d numpy.ndarray
        Input vector, which is sometimes zero and sometimes nonzero.

    Returns
    -------
    onsets : list
        Onset indices, where each entry is the start of a sequence of nonzero
        values in the input `indicator`.
    offsets : list
        Offset indices, where each entry is the last in a sequence of nonzero
        values in the input `indicator`, such that
        `indicator[onsets[i] : offsets[i] + 1] != 0`.
    """
    indices = np.nonzero(indicator)[0]

    if len(indices) == 0:
        return [], []

    onsets = [indices[0]]
    offsets = []
    breaks = np.nonzero(indices[1:] - indices[:-1] > 1)[0]
    for break_idx in breaks:
        offsets.append(indices[break_idx])
        onsets.append(indices[break_idx + 1])
    offsets.append(indices[-1])

    return onsets, offsets


class DedentTextHelpFormatter(argparse.HelpFormatter):
    """
    Help message formatter which retains formatting of all help text, except
    from indentation. Leading new lines are also stripped.
    """

    def _split_lines(self, text, width):
        import textwrap

        return textwrap.dedent(text.lstrip("\n")).splitlines()

    def _fill_text(self, text, width, indent):
        import textwrap

        return "".join(
            indent + line
            for line in textwrap.dedent(text.lstrip("\n")).splitlines(keepends=True)
        )


class FlexibleHelpFormatter(argparse.HelpFormatter):
    """
    Help message formatter which can handle different formatting
    specifications.

    The following formatters are supported:

        - "R|" : raw; will be left as is, processed using
                 `argparse.RawTextHelpFormatter`.
        - "d|" : raw except for indentation; will be dedented and leading
                 newlines stripped only, processed using
                 `argparse.RawTextHelpFormatter`.

    The format specifier will be stripped from the text.

    Notes
    -----
    Based on:
        - https://stackoverflow.com/a/22157266/1960959
        - https://sourceforge.net/projects/ruamel-std-argparse/
    """

    def _split_lines(self, text, *args, **kwargs):
        if len(text) < 2 or text[1] != "|":
            return super()._split_lines(text, *args, **kwargs)
        if text[0] == "R":
            return argparse.RawTextHelpFormatter._split_lines(
                self, text[2:], *args, **kwargs
            )
        if text[0] == "d":
            return DedentTextHelpFormatter._split_lines(self, text[2:], *args, **kwargs)
        raise ValueError("Invalid format code: {}".format(text[0]))

    def _fill_text(self, text, *args, **kwargs):
        if len(text) < 2 or text[1] != "|":
            return super()._fill_text(text, *args, **kwargs)
        if text[0] == "R":
            return argparse.RawTextHelpFormatter._fill_text(
                self, text[2:], *args, **kwargs
            )
        if text[0] == "d":
            return DedentTextHelpFormatter._fill_text(self, text[2:], *args, **kwargs)
        raise ValueError("Invalid format code: {}".format(text[0]))


class error_styling(contextlib.AbstractContextManager):
    """
    Wrap an error message in ANSI codes to stylise its appearance in the
    terminal as red and bold (bright). If the context is exited with an error,
    that error message will be red.

    Parameters
    ----------
    message : str
        Text of the error message to stylise.

    Returns
    -------
    str
        Stylised message.
    """

    def __init__(self, message=""):
        # Make the error message be bold and red
        if message:
            # Bold for the message, then return to normal font weight
            message = colorama.Style.BRIGHT + message + colorama.Style.NORMAL
            # Make the error message, and everything which comes after it, be
            # red. We don't reset the colour in case we are inside a larger
            # error message, which should also be red.
            message = colorama.Fore.RED + message
        self.message = message

    def __enter__(self):
        # Change all text sent to the terminal to be red, until we leave this
        # context
        print(colorama.Fore.RED, end="")
        return self.message

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            # If we leave the context normally, reset the text color and
            # introduce a new line.
            # Now all changes we have made when we entered the context have
            # been reset.
            print(colorama.Fore.RESET)
        else:
            # If we leave the context with an error, ensure the error message
            # is definitely red.
            print(colorama.Fore.RED, end="")
