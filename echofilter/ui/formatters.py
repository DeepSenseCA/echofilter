"""
Provides extensions to argparse.
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

import argparse


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

    `"R|"`
        Raw. will be left as is, processed using
        `argparse.RawTextHelpFormatter`.
    `"d|"`
        Raw except for indentation. Will be dedented and leading
        newlines stripped only, processed using
        `argparse.RawTextHelpFormatter`.

    The format specifier will be stripped from the text.

    Notes
    -----
    Based on https://stackoverflow.com/a/22157266/1960959
    and https://sourceforge.net/projects/ruamel-std-argparse/.
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


def format_parser_for_sphinx(parser):
    """
    Pre-format parser help for sphinx-argparse processing.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Initial argument parser.

    Returns
    -------
    parser : argparse.ArgumentParser
        The same argument parser, but with raw help text touched up so it
        renders correctly when passed through sphinx-argparse.
    """
    for action_group in parser._action_groups:
        for action in action_group._group_actions:
            # Get the help text for this action
            help = action.help
            # Remove quotes around default strings, to prevent double-marking.
            help = help.replace('"%(default)s"', "%(default)s")
            help = help.replace("'%(default)s'", "%(default)s")
            # Wrap any default values in `` so they are rendered as code.
            help = help.replace("%(default)s", "``%(default)s``")
            # But also ensure we don't add backticks around backticks
            help = help.replace("````", "``")
            # Remove flexible formatter indictor, if present
            if (
                parser.formatter_class == FlexibleHelpFormatter
                and len(help) > 1
                and help[1] == "|"
            ):
                help = help[2:]
            # Overwrite the help text
            action.help = help
    return parser
