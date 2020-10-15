"""
Provides extensions to argparse.
"""

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
