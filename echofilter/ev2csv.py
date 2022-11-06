#!/usr/bin/env python
"""
Export raw EV files in CSV format.
"""

# ev2csv is part of Echofilter.
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

import os
import sys
import warnings

from tqdm.auto import tqdm

import echofilter.path
import echofilter.ui
import echofilter.utils
import echofilter.win


# Provide a warning for non-Windows users
if not echofilter.path.check_if_windows():
    msg = (
        "\nev2csv requires the Echoview application, which is only"
        " available on Windows operating systems."
    )
    with echofilter.ui.style.warning_message(msg) as msg:
        print("")
        warnings.warn(msg, category=RuntimeWarning)


DEFAULT_VARNAME = "Fileset1: Sv pings T1"


def run_ev2csv(
    paths,
    variable_name=DEFAULT_VARNAME,
    source_dir=".",
    recursive_dir_search=True,
    output_dir="",
    suffix=None,
    keep_ext=False,
    skip_existing=False,
    overwrite_existing=False,
    minimize_echoview=False,
    hide_echoview="new",
    verbose=1,
    dry_run=False,
):
    """
    Export EV files to raw CSV files.

    Parameters
    ----------
    paths : iterable
        Paths to input EV files to process, or directories containing EV files.
        These may be full paths or paths relative to `source_dir`. For each
        folder specified, any files with extension `"csv"` within the folder
        and all its tree of subdirectories will be processed.
    variable_name : str, optional
        Name of the Echoview acoustic variable to export. Default is
        `"Fileset1: Sv pings T1"`.
    source_dir : str, optional
        Path to directory where files are found. Default is `"."`.
    recursive_dir_search : bool, optional
        How to handle directory inputs in `paths`. If `False`, only files
        (with the correct extension) in the directory will be included.
        If `True`, subdirectories will also be walked through to find input
        files. Default is `True`.
    output_dir : str, optional
        Directory where output files will be written. If this is an empty
        string (`""`, default), outputs are written to the same directory as
        each input file. Otherwise, they are written to `output_dir`,
        preserving their path relative to `source_dir` if relative paths were
        used.
    suffix : str, optional
        Output filename suffix. Default is `"_Sv_raw.csv"` if `keep_ext=False`,
        or `".Sv_raw.csv"` if `keep_ext=True`.
    keep_ext : bool, optional
        Whether to preserve the file extension in the input file name when
        generating output file name. Default is `False`, removing the
        extension.
    skip_existing : bool, optional
        Whether to skip processing files whose destination paths already
        exist. If `False` (default), an error is raised if the destination file
        already exists.
    overwrite_existing : bool, optional
        Whether to overwrite existing output files. If `False` (default), an
        error is raised if the destination file already exists.
    minimize_echoview : bool, optional
        If `True`, the Echoview window being used will be minimized while this
        function is running. Default is `False`.
    hide_echoview : {"never", "new", "always"}, optional
        Whether to hide the Echoview window entirely while the code runs.
        If `hide_echoview="new"`, the application is only hidden if it
        was created by this function, and not if it was already running.
        If `hide_echoview="always"`, the application is hidden even if it was
        already running. In the latter case, the window will be revealed again
        when this function is completed. Default is `"new"`.
    verbose : int, optional
        Level of verbosity. Default is `1`.
    dry_run : bool, optional
        If `True`, perform a trial run with no changes made. Default is
        `False`.

    Returns
    -------
    list of str
        Paths to generated CSV files.
    """

    if suffix is not None:
        pass
    elif keep_ext:
        suffix = ".Sv_raw.csv"
    else:
        suffix = "_Sv_raw.csv"

    files = list(
        echofilter.path.parse_files_in_folders(
            paths, source_dir, "ev", recursive=recursive_dir_search
        )
    )
    if verbose >= 1:
        print("Processing {} file{}".format(len(files), "" if len(files) == 1 else "s"))

    if len(files) == 1 or verbose <= 0:
        maybe_tqdm = lambda x: x
    else:
        maybe_tqdm = lambda x: tqdm(x, desc="ev2csv")

    skip_count = 0
    output_files = []

    # Open Echoview connection
    with echofilter.win.maybe_open_echoview(
        do_open=not dry_run,
        minimize=minimize_echoview,
        hide=hide_echoview,
    ) as ev_app:
        for fname in maybe_tqdm(files):
            if verbose >= 2:
                print("Exporting {} to raw CSV".format(fname))

            # Check what the full path should be
            fname_full = echofilter.path.determine_file_path(fname, source_dir)

            # Determine where destination should be placed
            destination = echofilter.path.determine_destination(
                fname, fname_full, source_dir, output_dir
            )
            if not keep_ext:
                destination = os.path.splitext(destination)[0]
            destination += suffix

            # Check whether to skip processing this file
            if not os.path.exists(destination):
                pass
            elif skip_existing:
                if verbose >= 2:
                    print("Skipping {}".format(fname))
                skip_count += 1
                continue
            elif not overwrite_existing:
                raise EnvironmentError(
                    "Output {} already exists.\n"
                    " Run with overwrite_existing=True (with the command line"
                    " interface, use the --force flag) to overwrite existing"
                    " outputs, or skip_existing=True (with the command line"
                    " interface, use the --skip-existing flag) to skip existing"
                    " outputs.".format(destination)
                )

            if dry_run:
                if verbose >= 1:
                    print("Would write to CSV file to {}".format(destination))
                continue

            # Export a single EV file to raw CSV
            ev2csv(
                fname_full,
                destination,
                variable_name=variable_name,
                ev_app=ev_app,
                verbose=verbose - 1,
            )
            output_files.append(destination)

    if verbose >= 1:
        s = "Finished {}processing {} file{}.".format(
            "simulating " if dry_run else "",
            len(files),
            "" if len(files) == 1 else "s",
        )
        if skip_count > 0:
            s += " Of these, {} file{} skipped.".format(
                skip_count,
                " was" if skip_count == 1 else "s were",
            )
        print(s)

    return output_files


def ev2csv(
    input,
    destination,
    variable_name=DEFAULT_VARNAME,
    ev_app=None,
    verbose=0,
):
    """
    Export a single EV file to CSV.

    Parameters
    ----------
    input : str
        Path to input file.
    destination : str
        Filename of output destination.
    variable_name : str, optional
        Name of the Echoview acoustic variable to export. Default is
        `"Fileset1: Sv pings T1"`.
    ev_app : win32com.client.Dispatch object or None, optional
        An object which can be used to interface with the Echoview application,
        as returned by `win32com.client.Dispatch`. If `None` (default), a
        new instance of the application is opened (and closed on completion).
    verbose : int, optional
        Level of verbosity. Default is `0`.

    Returns
    -------
    destination : str
        Absolute path to `destination`.
    """

    if verbose >= 1:
        print("  Opening {} in Echoview".format(input))

    # Ensure input and destination are absolute paths
    input = os.path.abspath(input)
    destination = os.path.abspath(destination)

    # Open the EV file
    with echofilter.win.open_ev_file(input, ev_app) as ev_file:

        # Find the right variable
        av = ev_file.Variables.FindByName(variable_name).AsVariableAcoustic()

        # Make sure we don't exclude anything, i.e. export "raw" data
        av.Properties.Analysis.ExcludeAbove = "None"
        av.Properties.Analysis.ExcludeBelow = "None"
        av.Properties.Analysis.ExcludeBadDataRegions = False
        av.Properties.Analysis.ExcludeBadLineStatusPings = False
        av.Properties.Data.ApplyMinimumThreshold = False
        av.Properties.Data.ApplyMaximumThreshold = False
        av.Properties.Data.ApplyMinimumTsThreshold = False
        av.Properties.Data.ApplyTimeVariedThreshold = False

        # Export the raw file
        if verbose >= 1:
            print("  Writing output {}".format(destination))
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        av.ExportData(destination, -1, -1)

    # The file is automatically closed when we leave the context
    return destination


def get_parser():
    """
    Build parser for ev2csv command line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        CLI argument parser for ev2csv.
    """

    import argparse

    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        prog = os.path.split(__file__)[1]
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Echoview to raw CSV exporter",
        formatter_class=echofilter.ui.formatters.FlexibleHelpFormatter,
        add_help=False,
    )

    # Actions
    group_action = parser.add_argument_group(
        "Actions",
        "These arguments specify special actions to perform. The main action"
        " of this program is supressed if any of these are given.",
    )
    group_action.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit.",
    )
    group_action.add_argument(
        "--version",
        "-V",
        action="version",
        version="%(prog)s {version}".format(version=echofilter.__version__),
        help="Show program's version number and exit.",
    )

    # Input files
    group_positional = parser.add_argument_group("Positional arguments")
    group_positional.add_argument(
        "paths",
        type=str,
        nargs="+",
        default=[],
        metavar="FILE_OR_DIRECTORY",
        help="""d|
            File(s)/directory(ies) to process.
            Inputs can be absolute paths or relative paths to
            either files or directories. Paths can be given
            relative to the current directory, or optionally be
            relative to the SOURCE_DIR argument specified with
            ``--source-dir``. For each directory given, the directory
            will be searched recursively for files bearing an
            extension specified by SEARCH_EXTENSION (see the
            ``--extension`` argument for details).
            Multiple files and directories can be specified,
            separated by spaces.
            This is a required argument. At least one input file
            or directory must be given.
            In order to process the directory given by SOURCE_DIR,
            specify "." for this argument, such as::

                ev2csv . --source-dir SOURCE_DIR
        """,
    )
    group_infile = parser.add_argument_group(
        "Input file arguments",
        "Optional parameters specifying which files will processed.",
    )
    group_infile.add_argument(
        "--source-dir",
        "-d",
        dest="source_dir",
        type=str,
        default=".",
        metavar="SOURCE_DIR",
        help="""
            Path to source directory which contains the files and folders
            specified by the paths argument. Default: "%(default)s" (the
            current directory).
        """,
    )
    group_infile.add_argument(
        "--recursive-dir-search",
        dest="recursive_dir_search",
        action="store_true",
        default=True,
        help="""d|
            For any directories provided in the FILE_OR_DIRECTORY
            input, all subdirectories will also be recursively
            walked through to find files to process.
            This is the default behaviour.
        """,
    )
    group_infile.add_argument(
        "--no-recursive-dir-search",
        dest="recursive_dir_search",
        action="store_false",
        help="""
            For any directories provided in the FILE_OR_DIRECTORY
            input, only files within the specified directory will
            be included in the files to process. Subfolders within
            the directory will not be included.
        """,
    )
    group_infile.add_argument(
        "--skip-existing",
        "--skip",
        dest="skip_existing",
        action="store_true",
        help="""
            Skip processing files for which all outputs already exist
        """,
    )

    # Output files
    group_outfile = parser.add_argument_group(
        "Destination file arguments",
        "Optional parameters specifying where output files will be located.",
    )
    group_outfile.add_argument(
        "--output-dir",
        "-o",
        metavar="OUTPUT_DIR",
        type=str,
        default="",
        help="""
            Path to output directory. If empty (default), each output is placed
            in the same directory as its input file. If OUTPUT_DIR is
            specified, the full output path for each file all contains the
            subtree of the input file relative to the base directory given by
            SOURCE_DIR.
        """,
    )
    group_outfile.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="""
            Perform a trial run, with no changes made. Text printed to the
            command prompt indicates which files would be processed, but work
            is only simulated and not performed.
        """,
    )
    group_outfile.add_argument(
        "--force",
        "-f",
        dest="overwrite_existing",
        action="store_true",
        help="""
            Overwrite existing files without warning. Default behaviour is to
            stop processing if an output file already exists.
        """,
    )
    group_outfile.add_argument(
        "--output-suffix",
        "--suffix",
        dest="suffix",
        type=str,
        default=None,
        help="""
            Output filename suffix. Default is ``"_Sv_raw.csv"``, or
            ``".Sv_raw.csv"`` if the ``--keep_ext`` argument is supplied.
        """,
    )

    # Input data transforms
    group_inproc = parser.add_argument_group(
        "Input processing arguments",
        "Optional parameters specifying how data will be loaded from the input"
        " files and transformed before it given to the model.",
    )
    group_inproc.add_argument(
        "--variable-name",
        "--vn",
        dest="variable_name",
        type=str,
        default=DEFAULT_VARNAME,
        help="""
            Name of the Echoview acoustic variable to load from EV files.
            Default: "%(default)s".
        """,
    )

    # Echoview interaction arguments
    group_evwin = parser.add_argument_group(
        "Echoview window management",
        "Optional parameters specifying how to interact with any Echoview"
        " windows which are used during this process.",
    )
    group_evwin_hiding = group_evwin.add_mutually_exclusive_group()
    group_evwin_hiding.add_argument(
        "--hide-echoview",
        dest="hide_echoview",
        action="store_const",
        const="new",
        help="""
            Hide any Echoview window spawned by this program. If it must use
            an Echoview instance which was already running, that window is not
            hidden. This is the default behaviour.
        """,
    )
    group_evwin_hiding.add_argument(
        "--show-echoview",
        dest="hide_echoview",
        action="store_const",
        const="never",
        default=None,
        help="""
            Don't hide an Echoview window created to run this code. (Disables
            the default behaviour which is equivalent to ``--hide-echoview``.)
        """,
    )
    group_evwin_hiding.add_argument(
        "--always-hide-echoview",
        "--always-hide",
        dest="hide_echoview",
        action="store_const",
        const="always",
        help="""
            Hide the Echoview window while this code runs, even if this
            process is utilising an Echoview window which was already open.
        """,
    )
    group_evwin.add_argument(
        "--minimize-echoview",
        dest="minimize_echoview",
        action="store_true",
        help="""
            Minimize any Echoview window used to runs this code while it runs.
            The window will be restored once the program is finished.
            If this argument is supplied, ``--show-echoview`` is implied unless
            ``--hide-echoview`` is also given.
        """,
    )

    # Verbosity controls
    group_verb = parser.add_argument_group(
        "Verbosity arguments",
        "Optional parameters controlling how verbose the program should be"
        " while it is running.",
    )
    group_verb.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help="""
            Increase the level of verbosity of the program. This can be
            specified multiple times, each will increase the amount of detail
            printed to the terminal. The default verbosity level is %(default)s.
        """,
    )
    group_verb.add_argument(
        "--quiet",
        "-q",
        action="count",
        default=0,
        help="""
            Decrease the level of verbosity of the program. This can be
            specified multiple times, each will reduce the amount of detail
            printed to the terminal.
        """,
    )

    return parser


def _get_parser_sphinx():
    """
    Pre-format parser help for sphinx-argparse processing.
    """
    return echofilter.ui.formatters.format_parser_for_sphinx(get_parser())


def main():
    """
    Run ev2csv command line interface.
    """
    parser = get_parser()
    kwargs = vars(parser.parse_args())
    kwargs["verbose"] -= kwargs.pop("quiet", 0)

    if kwargs["hide_echoview"] is None:
        kwargs["hide_echoview"] = "never" if kwargs["minimize_echoview"] else "new"

    run_ev2csv(**kwargs)


if __name__ == "__main__":
    main()
