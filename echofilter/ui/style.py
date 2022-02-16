"""
User interface styling, using ANSI codes and colorama.
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

import contextlib

import colorama


colorama.init()


class _AbstractStyle(object):
    """
    Abstract class for formatting styles.
    """

    start = ""
    reset = ""

    @classmethod
    def apply(cls, string):
        """
        Apply the ANSI formatting.

        Parameters
        ----------
        string : str
            Input string to format.

        Returns
        -------
        formatted_string : str
            String prepended with a start ANSI code and appended with a
            reset ANSI code which undoes the start code.
        """
        return cls.start + string + cls.reset


class ErrorStyle(_AbstractStyle):
    """
    Defines the style for an error string; red foreground.
    """

    start = colorama.Fore.RED
    reset = colorama.Fore.RESET


class WarningStyle(_AbstractStyle):
    """
    Defines the style for a warning string; cyan foreground.
    """

    start = colorama.Fore.CYAN
    reset = colorama.Fore.RESET


class ProgressStyle(_AbstractStyle):
    """
    Defines the style for a progress string; green foreground.
    """

    start = colorama.Fore.GREEN
    reset = colorama.Fore.RESET


class DryrunStyle(_AbstractStyle):
    """
    Defines the style for dry-run text; magenta foreground.
    """

    start = colorama.Fore.MAGENTA
    reset = colorama.Fore.RESET


class SkipStyle(_AbstractStyle):
    """
    Defines the style for skip text; yellow foreground.
    """

    start = colorama.Fore.YELLOW
    reset = colorama.Fore.RESET


class OverwriteStyle(_AbstractStyle):
    """
    Defines the style for overwrite text; bright blue.
    """

    start = colorama.Fore.BLUE + colorama.Style.BRIGHT
    reset = colorama.Fore.RESET + colorama.Style.NORMAL


class HighlightStyle(_AbstractStyle):
    """
    Defines the style for highlighted text; bright style.
    """

    start = colorama.Style.BRIGHT
    reset = colorama.Style.NORMAL


class AsideStyle(_AbstractStyle):
    """
    Defines the style for aside text; dim style.
    """

    start = colorama.Style.DIM
    reset = colorama.Style.NORMAL


def error_fmt(string):
    """
    Wrap a string in ANSI codes to render it in the style of an error
    when printed at the terminal.

    Parameters
    ----------
    string : str
        Input string to format.

    Returns
    -------
    formatted_string : str
        String prepended with a start ANSI code and appended with a
        reset ANSI code which undoes the start code.
    """
    return ErrorStyle.apply(string)


def warning_fmt(string):
    """
    Wrap a string in ANSI codes to render it in the style of a warning
    when printed at the terminal.

    Parameters
    ----------
    string : str
        Input string to format.

    Returns
    -------
    formatted_string : str
        String prepended with a start ANSI code and appended with a
        reset ANSI code which undoes the start code.
    """
    return WarningStyle.apply(string)


def progress_fmt(string):
    """
    Wrap a string in ANSI codes to render it in the style of progress text
    when printed at the terminal.

    Parameters
    ----------
    string : str
        Input string to format.

    Returns
    -------
    formatted_string : str
        String prepended with a start ANSI code and appended with a
        reset ANSI code which undoes the start code.
    """
    return ProgressStyle.apply(string)


def dryrun_fmt(string):
    """
    Wrap a string in ANSI codes to render it in the style of dry-run text
    when printed at the terminal.

    Parameters
    ----------
    string : str
        Input string to format.

    Returns
    -------
    formatted_string : str
        String prepended with a start ANSI code and appended with a
        reset ANSI code which undoes the start code.
    """
    return DryrunStyle.apply(string)


def skip_fmt(string):
    """
    Wrap a string in ANSI codes to render it in the style of a skip message
    when printed at the terminal.

    Parameters
    ----------
    string : str
        Input string to format.

    Returns
    -------
    formatted_string : str
        String prepended with a start ANSI code and appended with a
        reset ANSI code which undoes the start code.
    """
    return SkipStyle.apply(string)


def overwrite_fmt(string):
    """
    Wrap a string in ANSI codes to render it in the style of an overwrite
    message when printed at the terminal.

    Parameters
    ----------
    string : str
        Input string to format.

    Returns
    -------
    formatted_string : str
        String prepended with a start ANSI code and appended with a
        reset ANSI code which undoes the start code.
    """
    return OverwriteStyle.apply(string)


def highlight_fmt(string):
    """
    Wrap a string in ANSI codes to render it in a highlighted style
    when printed at the terminal.

    Parameters
    ----------
    string : str
        Input string to format.

    Returns
    -------
    formatted_string : str
        String prepended with a start ANSI code and appended with a
        reset ANSI code which undoes the start code.
    """
    return HighlightStyle.apply(string)


def aside_fmt(string):
    """
    Wrap a string in ANSI codes to render it in an aside (de-emphasised) style
    when printed at the terminal.

    Parameters
    ----------
    string : str
        Input string to format.

    Returns
    -------
    formatted_string : str
        String prepended with a start ANSI code and appended with a
        reset ANSI code which undoes the start code.
    """
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


class warning_message(contextlib.AbstractContextManager):
    """
    Wrap a warning message in ANSI codes to stylise its appearance in the
    terminal as cyan and bold (bright). All statements printed during the
    context will be in cyan.

    Parameters
    ----------
    message : str
        Text of the warning message to stylise.

    Returns
    -------
    str
        Stylised message.
    """

    def __init__(self, message=""):
        # Make the error message be bold and cyan
        if message:
            # Bold for the message, then return to normal font weight
            message = HighlightStyle.start + message + HighlightStyle.reset
            # Make the warning message, and everything which comes after it, be
            # cyan. We don't reset the colour in case we are inside a larger
            # message, which should also be cyan.
            message = WarningStyle.start + message
        self.message = message

    def __enter__(self):
        # Change all text sent to the terminal to be cyan, until we leave this
        # context
        print(WarningStyle.start, end="")
        return self.message

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            # If we leave the context normally, reset the text color and
            # introduce a new line.
            # Now all changes we have made when we entered the context have
            # been reset.
            print(WarningStyle.reset)
        else:
            # If we leave the context with an error, use error message
            # styling instead.
            print(ErrorStyle.start, end="")
