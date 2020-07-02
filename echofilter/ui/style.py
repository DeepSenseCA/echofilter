"""
User interface styling, using ANSI codes and colorama.
"""

import contextlib

import colorama


colorama.init()


class _AbstractStyle(object):
    start = ""
    reset = ""

    @classmethod
    def apply(cls, string):
        return cls.start + string + cls.reset


class ErrorStyle(_AbstractStyle):
    start = colorama.Fore.RED
    reset = colorama.Fore.RESET


class WarningStyle(_AbstractStyle):
    start = colorama.Fore.CYAN
    reset = colorama.Fore.RESET


class ProgressStyle(_AbstractStyle):
    start = colorama.Fore.GREEN
    reset = colorama.Fore.RESET


class DryrunStyle(_AbstractStyle):
    start = colorama.Fore.MAGENTA
    reset = colorama.Fore.RESET


class SkipStyle(_AbstractStyle):
    start = colorama.Fore.YELLOW
    reset = colorama.Fore.RESET


class OverwriteStyle(_AbstractStyle):
    start = colorama.Fore.BLUE + colorama.Style.BRIGHT
    reset = colorama.Fore.RESET + colorama.Style.NORMAL


class HighlightStyle(_AbstractStyle):
    start = colorama.Style.BRIGHT
    reset = colorama.Style.NORMAL


class AsideStyle(_AbstractStyle):
    start = colorama.Style.DIM
    reset = colorama.Style.NORMAL


def error_fmt(string):
    return ErrorStyle.apply(string)


def warning_fmt(string):
    return WarningStyle.apply(string)


def progress_fmt(string):
    return ProgressStyle.apply(string)


def dryrun_fmt(string):
    return DryrunStyle.apply(string)


def skip_fmt(string):
    return SkipStyle.apply(string)


def overwrite_fmt(string):
    return OverwriteStyle.apply(string)


def highlight_fmt(string):
    return HighlightStyle.apply(string)


def aside_fmt(string):
    return AsideStyle.apply(string)


class error_message(contextlib.AbstractContextManager):
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
            message = HighlightStyle.start + message + HighlightStyle.reset
            # Make the error message, and everything which comes after it, be
            # red. We don't reset the colour in case we are inside a larger
            # error message, which should also be red.
            message = ErrorStyle.start + message
        self.message = message

    def __enter__(self):
        # Change all text sent to the terminal to be red, until we leave this
        # context
        print(ErrorStyle.start, end="")
        return self.message

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            # If we leave the context normally, reset the text color and
            # introduce a new line.
            # Now all changes we have made when we entered the context have
            # been reset.
            print(ErrorStyle.reset)
        else:
            # If we leave the context with an error, ensure the error message
            # is definitely red.
            print(ErrorStyle.start, end="")
