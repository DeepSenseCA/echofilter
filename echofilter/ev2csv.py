#!/usr/bin/env python
"""
Export raw EV files in CSV format.
"""

import os
import sys
import warnings

from tqdm.auto import tqdm

import echofilter.ev
import echofilter.path


# Provide a warning for non-Windows users
if not echofilter.path.check_if_windows():
    print()
    warnings.warn(
        "ev2csv requires the EchoView application, which is only available on"
        " Windows operating systems.",
        category=RuntimeWarning,
    )
    print()


DEFAULT_VARNAME = "Fileset1: Sv pings T1"


def run_ev2csv(
    files,
    variable_name=DEFAULT_VARNAME,
    data_dir=".",
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
    files : iterable
        Paths to input EV files to process, or directories containing EV files.
        These may be full paths or paths relative to `data_dir`. For each
        folder specified, any files with extension `'csv'` within the folder
        and all its tree of subdirectories will be processed.
    variable_name : str, optional
        Name of the EchoView acoustic variable to export. Default is
        `'Fileset1: Sv pings T1'`.
    data_dir : str, optional
        Path to directory where files are found. Default is `'.'`.
    output_dir : str, optional
        Directory where output files will be written. If this is `''`, outputs
        are written to the same directory as each input file. Otherwise, they
        are written to `output_dir`, preserving their path relative to
        `data_dir` if relative paths were used. Default is `''`.
    suffix : str, optional
        Output filename suffix. Default is `'_Sv_raw.csv'` if `keep_ext=False`,
        or `'.Sv_raw.csv'` if `keep_ext=True`.
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

    files = list(echofilter.path.parse_files_in_folders(files, data_dir, "ev"))
    if verbose >= 1:
        print("Processing {} file{}".format(len(files), "" if len(files) == 1 else "s"))

    if len(files) == 1 or verbose <= 0:
        maybe_tqdm = lambda x: x
    else:
        maybe_tqdm = lambda x: tqdm(x, desc="ev2csv")

    skip_count = 0
    output_files = []

    # Open EchoView connection
    with echofilter.ev.maybe_open_echoview(
        do_open=not dry_run, minimize=minimize_echoview, hide=hide_echoview,
    ) as ev_app:
        for fname in maybe_tqdm(files):
            if verbose >= 2:
                print("Exporting {} to raw CSV".format(fname))

            # Check what the full path should be
            fname_full = echofilter.path.determine_file_path(fname, data_dir)

            # Determine where destination should be placed
            destination = echofilter.path.determine_destination(
                fname, fname_full, data_dir, output_dir
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
                skip_count, " was" if skip_count == 1 else "s were",
            )
        print(s)

    return output_files


def ev2csv(
    input, destination, variable_name=DEFAULT_VARNAME, ev_app=None, verbose=0,
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
        Name of the EchoView acoustic variable to export. Default is
        `'Fileset1: Sv pings T1'`.
    ev_app : win32com.client.Dispatch object or None, optional
        An object which can be used to interface with the EchoView application,
        as returned by `win32com.client.Dispatch`. If `None` (default), a
        new instance of the application is opened (and closed on completion).
    verbose : int, optional
        Level of verbosity. Default is `0`.
    """

    if verbose >= 1:
        print("  Opening {} in EchoView".format(input))

    # Ensure input and destination are absolute paths
    input = os.path.abspath(input)
    destination = os.path.abspath(destination)

    # Open the EV file
    with echofilter.ev.open_ev_file(input, ev_app) as ev_file:

        # Find the right variable
        av = ev_file.Variables.FindByName(variable_name).AsVariableAcoustic()

        # Make sure we don't exclude anything, i.e. export "raw" data
        av.Properties.Analysis.ExcludeAbove = "None"
        av.Properties.Analysis.ExcludeBelow = "None"
        av.Properties.Analysis.ExcludeBadDataRegions = False

        # Export the raw file
        if verbose >= 1:
            print("  Writing output {}".format(destination))
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        av.ExportData(destination, -1, -1)

    # The file is automatically closed when we leave the context
    return destination


def main():
    import argparse

    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        prog = os.path.split(__file__)[1]
    parser = argparse.ArgumentParser(
        prog=prog, description="EchoView to raw CSV exporter",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version="%(prog)s {version}".format(version=echofilter.__version__),
    )
    parser.add_argument(
        "files",
        type=str,
        nargs="+",
        default=[],
        metavar="PATH",
        help="file(s) to process. For each directory given, all csv files"
        " within that directory and its subdirectories will be processed.",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        dest="data_dir",
        type=str,
        default=".",
        metavar="DIR",
        help='path to directory containing FILE (default: ".")',
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="",
        metavar="DIR",
        help="path to output directory. If empty, output is placed in the same"
        ' directory as the input file. (default: "")',
    )
    parser.add_argument(
        "--output-suffix",
        "--suffix",
        dest="suffix",
        type=str,
        default=None,
        help='Output filename suffix. Default is "_Sv_raw.csv", or'
        ' ".Sv_raw.csv" --keep_ext is used',
    )
    parser.add_argument(
        "--variable-name",
        "--vn",
        dest="variable_name",
        type=str,
        default=DEFAULT_VARNAME,
        help="Name of the EchoView acoustic variable to export. Default is {}.".format(
            DEFAULT_VARNAME
        ),
    )
    parser.add_argument(
        "--keep-ext",
        action="store_true",
        help="keep the input file extension in output file names",
    )
    parser.add_argument(
        "--force",
        "-f",
        dest="overwrite_existing",
        action="store_true",
        help="overwrite existing files without warning",
    )
    parser.add_argument(
        "--skip-existing",
        "--skip",
        dest="skip_existing",
        action="store_true",
        help="skip processing files for which all outputs already exist",
    )
    parser.add_argument(
        "--minimize-echoview",
        dest="minimize_echoview",
        action="store_true",
        help="minimize the Echoview window while this code runs",
    )
    parser.add_argument(
        "--show-echoview",
        dest="hide_echoview",
        action="store_const",
        const="never",
        default=None,
        help="don't hide an Echoview window created to run this code",
    )
    parser.add_argument(
        "--hide-echoview",
        dest="hide_echoview",
        action="store_const",
        const="new",
        help="hide Echoview window, but only if it was not already open (default behaviour)",
    )
    parser.add_argument(
        "--always-hide-echoview",
        "--always-hide",
        dest="hide_echoview",
        action="store_const",
        const="always",
        help="hide the Echoview window while this code runs, even if it was already open",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="perform a trial run with no changes made",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help="increase verbosity, print more progress details",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="count",
        default=0,
        help="decrease verbosity, print fewer progress details",
    )
    kwargs = vars(parser.parse_args())
    kwargs["verbose"] -= kwargs.pop("quiet", 0)

    if kwargs["hide_echoview"] is None:
        kwargs["hide_echoview"] = "never" if kwargs["minimize_echoview"] else "new"

    run_ev2csv(**kwargs)


if __name__ == "__main__":
    main()