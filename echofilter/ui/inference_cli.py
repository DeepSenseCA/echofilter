#!/usr/bin/env python

"""
Provides a command line interface for the inference routine.

This is separated out from inference.py so the responsiveness for simple
commands like --help and --version is faster, not needing to import the full
dependency stack.
"""

import argparse
from collections import OrderedDict
import os
import sys

import appdirs

from .. import __meta__
from .. import path
from . import formatters, style


CHECKPOINT_RESOURCES = OrderedDict(
    [
        (
            # 2020-06-15_05.39.07_conditional_effunet6x.2-1_lc32-se2_bs12-lr0.016_mesa_mom.92-.98_w.1-.5_ep100_elastic_resume5
            "conditional_mobile-stationary2_effunet6x2-1_lc32_v1.0.ckpt.tar",
            {"gdrive": "1vEshqJbj906tCEA4KybUmlBXN9YEG_Fu"},
        ),
        (
            # 2020-06-14_14.24.21_effunet6x.2-1_lc32-se2_bs20-lr0.02_mesa_mom.92-.98_w.1-.5_ep150_elastic_epseed_resume2
            "stationary2_effunet6x2-1_lc32_v1.0.ckpt.tar",
            {"gdrive": "1qPg9lUbtLk_DOR86sPS-DAvOaAlnvAF_"},
        ),
        (
            # 2020-06-14_13.53.10_effunet6x.2-1_lc32-se2_bs30-lr0.02_mesa_mom.92-.98_w.1-.5_ep300_elastic_epseed_resume2
            "mobile_effunet6x2-1_lc32_v1.0.ckpt.tar",
            {"gdrive": "1yODjTcRsGc3v8cNZqFZjqbtGHHwoAohU"},
        ),
    ]
)

DEFAULT_CHECKPOINT = next(iter(CHECKPOINT_RESOURCES))

DEFAULT_VARNAME = "Fileset1: Sv pings T1"


def get_default_cache_dir():
    """Determine the default cache directory."""
    return appdirs.user_cache_dir("echofilter", "DeepSense")


class ListCheckpoints(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        print("Currently available model checkpoints:")
        for checkpoint, props in CHECKPOINT_RESOURCES.items():
            print("    {}".format(checkpoint))
        parser.exit()  # exits the program with no more arg parsing and checking


class ListColors(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        from ..inference import hexcolor2rgb8, get_color_palette

        if values is None:
            include_xkcd = False
        else:
            include_xkcd = values.lower() != "css4"
        colors = get_color_palette(include_xkcd)
        for key, value in colors.items():
            extra = hexcolor2rgb8(value)
            if extra == value:
                extra = ""
            else:
                extra = "  (" + ", ".join(["{:3d}".format(x) for x in extra]) + ")"
            print("{:>31s}: {}{}".format(key, value, extra))
        parser.exit()  # exits the program with no more arg parsing and checking


def cli():
    """
    Run `run_inference` with arguments taken from the command line using
    argparse.
    """
    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py":
        prog = "echofilter"
    parser = argparse.ArgumentParser(
        prog=prog,
        description=__meta__.description,
        formatter_class=formatters.FlexibleHelpFormatter,
        add_help=False,
    )

    # Actions
    group_action = parser.add_argument_group(
        "Actions",
        "These arguments specify special actions to perform. The main action"
        " of this program is supressed if any of these are given.",
    )
    group_action.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit.",
    )
    group_action.add_argument(
        "--version",
        "-V",
        action="version",
        version="%(prog)s {version}".format(version=__meta__.version),
        help="Show program's version number and exit.",
    )
    group_action.add_argument(
        "--list-checkpoints",
        nargs=0,
        action=ListCheckpoints,
        help="Show the available model checkpoints and exit.",
    )
    group_action.add_argument(
        "--list-colors",
        "--list-colours",
        dest="list_colors",
        nargs="?",
        type=str,
        choices=["css4", "full", "xkcd"],
        action=ListColors,
        help="""d|
            Show the available line color names and exit.
            The available color palette can be viewed at
            https://matplotlib.org/gallery/color/named_colors.html.
            The XKCD color palette is also available, but is not
            shown in the output by default due to its size.
            To show the just main palette, run as `--list-colors`
            without argument, or `--list-colors css4`. To show the
            full palette, run as `--list-colors full`.
        """,
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
            --source-dir. For each directory given, the directory
            will be searched recursively for files bearing an
            extension specified by SEARCH_EXTENSION (see the
            --extension argument for details).
            Multiple files and directories can be specified,
            separated by spaces.
            This is a required argument. At least one input file
            or directory must be given, unless one of the
            arguments listed above under "Actions" is given.
            In order to process the directory given by SOURCE_DIR,
            specify "." for this argument, such as:
                echofilter . --source-dir SOURCE_DIR
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
        "-r",
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
        "-R",
        dest="recursive_dir_search",
        action="store_false",
        help="""
            For any directories provided in the FILE_OR_DIRECTORY
            input, only files within the specified directory will
            be included in the files to process. Subfolders within
            the directory will not be included.
        """,
    )
    default_extensions = ["csv"]
    if path.check_if_windows():
        default_extensions.append("ev")
    group_infile.add_argument(
        "--extension",
        "-x",
        dest="extensions",
        metavar="SEARCH_EXTENSION",
        type=str,
        nargs="+",
        default=default_extensions,
        help="""d|
            File extension(s) to process. This argument is used
            when the FILE_OR_DIRECTORY is a directory; files
            within the directory (and all its recursive
            subdirectories) are filtered against this list of
            extensions to identify which files to process.
            Default: %(default)s.
            (Note that the default SEARCH_EXTENSION value is
            OS-specific.)
        """,
    )
    group_infile.add_argument(
        "--skip-existing",
        "--skip",
        "-s",
        dest="skip_existing",
        action="store_true",
        help="""
            Skip processing files for which all outputs already exist
        """,
    )
    group_infile.add_argument(
        "--skip-incompatible",
        action="store_true",
        help="""
            Skip over incompatible input CSV files, without raising an error.
            Default behaviour is to stop if an input CSV file can not be
            processed. This argument is useful if you are processing a
            directory which contains a mixture of CSV files - some are Sv data
            exported from EV files and others are not.
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
        "--overwrite-files",
        dest="overwrite_existing",
        action="store_true",
        help="""
            Overwrite existing files without warning. Default behaviour is to
            stop processing if an output file already exists.
        """,
    )
    group_outfile.add_argument(
        "--overwrite-ev-lines",
        action="store_true",
        help="""
            Overwrite existing lines within the EchoView file without warning.
            Default behaviour is to append the current datetime to the name of
            the line in the event of a collision.
        """,
    )
    group_outfile.add_argument(
        "--force",
        "-f",
        dest="force",
        action="store_true",
        help="""
            Short-hand equivalent to supplying both --overwrite-files and
            --overwrite-ev-lines.
        """,
    )
    group_outfile.add_argument(
        "--no-ev-import",
        dest="import_into_evfile",
        action="store_false",
        help="""
            Do not import lines and regions back into any EV file inputs.
            Default behaviour is to import lines and regions and then
            save the file, overwriting the original EV file.
        """,
    )
    group_outfile.add_argument(
        "--no-turbulence-line",
        dest="generate_turbulence_line",
        action="store_false",
        help="""
            Do not output an evl file for the turbulence line, and do not
            import a turbulence line into the ev file.
        """,
    )
    group_outfile.add_argument(
        "--no-bottom-line",
        dest="generate_bottom_line",
        action="store_false",
        help="""
            Do not output an evl file for the bottom line, and do not import
            a bottom line into the ev file.
        """,
    )
    group_outfile.add_argument(
        "--no-surface-line",
        dest="generate_surface_line",
        action="store_false",
        help="""
            Do not output an evl file for the surface line, and do not import
            a surface line into the ev file.
        """,
    )
    group_outfile.add_argument(
        "--suffix-file",
        "--suffix",
        dest="suffix_file",
        type=str,
        default="",
        help="""
            Suffix to append to output artifacts evl and evr files, between
            the name of the file and the extension.
            If SUFFIX_FILE begins with an alphanumeric character, "-" is
            prepended to it to act as a delimiter.
            The default behavior is to not append a suffix.
        """,
    )
    group_outfile.add_argument(
        "--suffix-var",
        type=str,
        default=None,
        help="""
            Suffix to append to line and region names when  imported back into
            EV file. If SUFFIX_VAR begins with an alphanumeric character, "-"
            is prepended to it to act as a delimiter.
            The default behaviour is to match SUFFIX_FILE if it is set, and use
            "_echofilter" otherwise.
        """,
    )
    group_outfile.add_argument(
        "--color-turbulence",
        type=str,
        default="orangered",
        help="""
            Color to use for the turbulence line when it is imported into
            EchoView. This can either be the name of a supported color (see
            --list-colors for options), or a a hexadecimal string, or a string
            representation of an RGB color to supply directly to EchoView (such
            as "(0,255,0)"). Default: "%(default)s".
        """,
    )
    group_outfile.add_argument(
        "--color-bottom",
        type=str,
        default="orangered",
        help="""
            Color to use for the bottom line when it is imported into EchoView.
            This can either be the name of a supported color (see --list-colors
            for options), or a a hexadecimal string, or a string representation
            of an RGB color to supply directly to EchoView (such as
            "(0,255,0)"). Default: "%(default)s".
        """,
    )
    group_outfile.add_argument(
        "--color-surface",
        type=str,
        default="green",
        help="""
            Color to use for the surface line when it is imported into
            EchoView. This can either be the name of a supported color (see
            --list-colors for options), or a a hexadecimal string, or a string
            representation of an RGB color to supply directly to EchoView (such
            as "(0,255,0)"). Default: "%(default)s".
        """,
    )
    group_outfile.add_argument(
        "--thickness-turbulence",
        type=int,
        default=2,
        help="""
            Thicknesses with which the turbulence line will be displayed in
            EchoView. Default: %(default)s.
        """,
    )
    group_outfile.add_argument(
        "--thickness-bottom",
        type=int,
        default=2,
        help="""
            Thicknesses with which the bottom line will be displayed in
            EchoView. Default: %(default)s.
        """,
    )
    group_outfile.add_argument(
        "--thickness-surface",
        type=int,
        default=1,
        help="""
            Thicknesses with which the surface line will be displayed in
            EchoView. Default: %(default)s.
        """,
    )
    DEFAULT_CACHE_DIR = get_default_cache_dir()
    group_outfile.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="""d|
            Path to checkpoint cache directory.
            Default: "%(default)s".
        """,
    )
    group_outfile.add_argument(
        "--cache-csv",
        nargs="?",
        type=str,
        default=None,
        const="",
        metavar="CSV_DIR",
        help="""
            Path to directory where CSV files generated from EV inputs should
            be cached. If this argument is supplied with an empty string,
            exported CSV files will be saved in the same directory as each
            input EV file. The default behaviour is discard any CSV files
            generated by this program once it has finished running.
        """,
    )
    group_outfile.add_argument(
        "--suffix-csv",
        type=str,
        default="",
        help="""
            Suffix to append to the file names of cached CSV files which are
            exported from EV files. The suffix is inserted between the input
            file name and the new file extension, ".csv".
            If SUFFIX_CSV begins with an alphanumeric character, a delimiter
            is prepended. The delimiter is "-", or "." if --keep-ext is given.
            The default behavior is to not append a suffix.
        """,
    )
    group_outfile.add_argument(
        "--keep-ext",
        action="store_true",
        help="""
            If provided, the output file names (evl, evr, csv) maintain the
            input file extension before their suffix (including a new file
            extension). Default behaviour is to strip the input file name
            extension before constructing the output paths.
        """,
    )

    # Output files
    group_outconfig = parser.add_argument_group(
        "Output configuration arguments",
        "Optional parameters specifying the properties of the output.",
    )
    group_outconfig.add_argument(
        "--line-status",
        type=int,
        default=3,
        help="""d|
            Status value for all the lines which are generated.
            Options are:
              0: none
              1: unverified
              2: bad
              3: good
            Default: %(default)s.
        """,
    )
    group_outconfig.add_argument(
        "--offset",
        type=float,
        default=1.0,
        help="""
            Offset for turbulence, bottom, and surface lines, in metres.
            This will shift turbulence and surface lines downwards and the
            bottom line upwards by the same distance of OFFSET.
            Default: %(default)s.
        """,
    )
    group_outconfig.add_argument(
        "--offset-turbulence",
        type=float,
        default=None,
        help="""
            Offset for the turbulence line, in metres. This shifts the
            turbulence line downards by some distance OFFSET_TURBULENCE.
            If this is set, it overwrites the value provided by --offset.
        """,
    )
    group_outconfig.add_argument(
        "--offset-bottom",
        type=float,
        default=None,
        help="""
            Offset for the bottom line, in metres. This shifts the bottom line
            upwards by some distance OFFSET_BOTTOM. If this is set, it
            overwrites the value provided by --offset.
        """,
    )
    group_outconfig.add_argument(
        "--offset-surface",
        type=float,
        default=None,
        help="""
            Offset for the surface line, in metres. This shifts the surface
            line downards by some distance OFFSET_SURFACE. If this is set, it
            overwrites the value provided by --offset.
        """,
    )
    group_outconfig.add_argument(
        "--nearfield-cutoff",
        type=float,
        nargs="?",
        const=None,
        default=1.7,
        help="""
            Nearest approach distance for line adjacent to echosounder, in
            meters. If the echosounder is downfacing, NEARFIELD_CUTOFF is the
            minimum depth for the turbulence line. If the echosounder is
            upfacing, the maximum depth for the bottom line will be
            NEARFIELD_CUTOFF above the deepest depth in the input data, plus
            one inter-depth interval. If the --nearfield-cutoff argument is
            given without a value, no nearfield cut off will be applied.
            Default: %(default)s.
        """,
    )
    group_outconfig.add_argument(
        "--lines-during-passive",
        type=str,
        default="predict",
        choices=[
            "interpolate-time",
            "interpolate-index",
            "predict",
            "redact",
            "undefined",
        ],
        help="""d|
            Method used to handle line depths during collection
            periods determined to be passive recording instead of
            active recording.
            Options are:
              interpolate-time:
                  depths are linearly interpolated from active
                  recording periods, using the time at which
                  recordings where made.
              interpolate-index:
                  depths are linearly interpolated from active
                  recording periods, using the index of the
                  recording.
              predict:
                  the model's prediction for the lines during
                  passive data collection will be kept; the nature
                  of the prediction depends on how the model was
                  trained.
              redact:
                  no depths are provided during periods determined
                  to be passive data collection.
              undefined:
                  depths are replaced with the placeholder value
                  used by EchoView to denote undefined values,
                  which is {}.
            Default: "%(default)s".
        """,
    )
    group_outconfig.add_argument(
        "--collate-passive-length",
        type=int,
        default=10,
        help="""
            Maximum interval, in ping indices, between detected passive regions
            which will removed to merge consecutive passive regions together
            into a single, collated, region. Default: %(default)s.
        """,
    )
    group_outconfig.add_argument(
        "--collate-removed-length",
        type=int,
        default=10,
        help="""
            Maximum interval, in ping indices, between detected blocks
            (vertical rectangles) marked for removal which will also be removed
            to merge consecutive removed blocks together into a single,
            collated, region. Default: %(default)s.
        """,
    )
    group_outconfig.add_argument(
        "--minimum-passive-length",
        type=int,
        default=10,
        help="""
            Minimum length, in ping indices, which a detected passive region
            must have to be included in the output. Set to -1 to omit all
            detected passive regions from the output. Default: %(default)s.
        """,
    )
    group_outconfig.add_argument(
        "--minimum-removed-length",
        type=int,
        default=10,
        help="""
            Minimum length, in ping indices, which a detected removal block
            (vertical rectangle) must have to be included in the output.
            Set to -1 to omit all detected removal blocks from the output.
            Default: %(default)s.
        """,
    )
    group_outconfig.add_argument(
        "--minimum-patch-area",
        type=int,
        default=25,
        help="""
            Minimum area, in pixels, which a detected removal patch
            (contour/polygon) region must have to be included in the output.
            Set to -1 to omit all detected patches from the output.
            Default: %(default)s.
        """,
    )
    group_outconfig.add_argument(
        "--patch-mode",
        dest="patch_mode",
        type=str,
        default=None,
        help="""d|
            Type of mask patches to use. Must be supported by the
            model checkpoint used. Should be one of:
              merged:
                  Target patches for training were determined
                  after merging as much as possible into the
                  turbulence and bottom lines.
              original:
                  Target patches for training were determined
                  using original lines, before expanding the
                  turbulence and bottom lines.
              ntob:
                  Target patches for training were determined
                  using the original bottom line and the merged
                  turbulence line.
            Default: "merged" is used if downfacing; "ntob" if
            upfacing.
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
        help="""d|
            Name of the EchoView acoustic variable to load from
            EV files.
            Default: "%(default)s".
        """,
    )
    group_inproc.add_argument(
        "--row-len-selector",
        type=str,
        choices=["init", "min", "max", "median", "mode"],
        default="mode",
        help="""
            How to handle inputs with differing number of depth samples across
            time. This method is used to select the "master" number of depth
            samples and minimum and maximum depth. The Sv values for all
            timepoints are interpolated onto this range of depths in order to
            create an input which is sampled in a rectangular manner.
            Default: "%(default)s", the modal number of depths is used, and the
            modal depth range is select amongst time samples which bear this
            number of depths.
        """,
    )
    group_inproc.add_argument(
        "--facing",
        type=str,
        choices=["downward", "upward", "auto"],
        default="auto",
        help="""
            Orientation of echosounder. If this is "auto" (default), the
            orientation is automatically determined from the ordering of the
            depths field in the input (increasing depth values = "downward";
            diminishing depths = "upward").
        """,
    )
    group_inproc.add_argument(
        "--training-standardization",
        dest="use_training_standardization",
        action="store_true",
        help="""
            If this is given, Sv intensities are scaled using the values used
            when the model was trained before being given to the model for
            inference. The default behaviour is to derive the standardization
            values from the Sv statistics of the input instead.
        """,
    )
    group_inproc.add_argument(
        "--crop-min-depth",
        type=float,
        default=None,
        help="""
            Shallowest depth, in metres, to analyse. Data will be truncated at
            this depth, with shallower data removed before the Sv input is
            shown to the model. Default behaviour is not to truncate.
        """,
    )
    group_inproc.add_argument(
        "--crop-max-depth",
        type=float,
        default=None,
        help="""
            Deepest depth, in metres, to analyse. Data will be truncated at
            this depth, with deeper data removed before the Sv input is
            shown to the model. Default behaviour is not to truncate.
        """,
    )
    group_inproc.add_argument(
        "--autocrop-threshold",
        "--autozoom-threshold",
        dest="autocrop_threshold",
        type=float,
        default=0.35,
        help="""
            The inference routine will re-run the model with a zoomed in
            version of the data, if the fraction of the depth which it deems
            irrelevant exceeds the AUTO_CROP_THRESHOLD. The extent of the depth
            which is deemed relevant is from the shallowest point on the
            surface line to the deepest point on the bottom line.
            The data will only be zoomed in and re-analysed at most once.
            To always run the model through once (never auto zoomed), set to 1.
            To always run the model through exactly twice (always one round
            of auto-zoom), set to 0. Default: %(default)s.
        """,
    )
    group_inproc.add_argument(
        "--image-height",
        "--height",
        dest="image_height",
        type=float,
        default=None,
        help="""
            Height to which the Sv image will be rescaled, in pixels, before
            being given to the model. The default behaviour is to use the same
            height as was used when the model was trained.
        """,
    )

    # Model arguments
    group_model = parser.add_argument_group(
        "Model arguments",
        "Optional parameters specifying which model checkpoint will be used"
        " and how it is run.",
    )
    group_model.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="""d|
            Name of checkpoint to load, or path to a checkpoint
            file.
            Default: "%(default)s".
        """,
    )
    group_model.add_argument(
        "--unconditioned",
        "--force-unconditioned",
        dest="force_unconditioned",
        action="store_true",
        help="""
            If this flag is present and a conditional model is loaded,
            it will be run for its unconditioned output. This means the
            model is output is not conditioned on the orientation of
            the echosounder. By default, conditional models are used for
            their conditional output.
        """,
    )
    group_model.add_argument(
        "--logit-smoothing-sigma",
        type=float,
        nargs="+",
        metavar="SIGMA",
        default=[1],
        help="""
            Standard deviation of Gaussian smoothing kernel applied to the
            logits provided as the model's output. The smoothing regularises
            the output to make it smoother.
            Multiple values can be given to use different kernel sizes for
            each dimension, in which case the first value is for the timestamp
            dimension and the second value is for the depth dimension. If a
            single value is given, the kernel is symmetric. Values are relative
            to the pixel space returned by the UNet model.
            Set to 0 to disable.
            Default: %(default)s.
        """,
    )
    group_model.add_argument(
        "--device",
        type=str,
        default=None,
        help="""
            Device to use for running the model for inference.
            Default: use first GPU if available, otherwise use the CPU.
            Note: echofilter.exe is complied without GPU support and can only
            run on the CPU. To use the GPU you must use the source version.
        """,
    )

    # EchoView interaction arguments
    group_evwin = parser.add_argument_group(
        "EchoView window management",
        "Optional parameters specifying how to interact with any EchoView"
        " windows which are used during this process.",
    )
    group_evwin_hiding = group_evwin.add_mutually_exclusive_group()
    group_evwin_hiding.add_argument(
        "--hide-echoview",
        dest="hide_echoview",
        action="store_const",
        const="new",
        help="""
            Hide any EchoView window spawned by this program. If it must use
            an EchoView instance which was already running, that window is not
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
            Don't hide an EchoView window created to run this code. (Disables
            the default behaviour which is equivalent to --hide-echoview.)
        """,
    )
    group_evwin_hiding.add_argument(
        "--always-hide-echoview",
        "--always-hide",
        dest="hide_echoview",
        action="store_const",
        const="always",
        help="""
            Hide the EchoView window while this code runs, even if this
            process is utilising an EchoView window which was already open.
        """,
    )
    group_evwin.add_argument(
        "--minimize-echoview",
        dest="minimize_echoview",
        action="store_true",
        help="""
            Minimize any EchoView window used to runs this code while it runs.
            The window will be restored once the program is finished.
            If this argument is supplied, --show-echoview is implied unless
            --hide-echoview is also given.
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
        default=2,
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

    kwargs = vars(parser.parse_args())

    kwargs.pop("list_checkpoints")
    kwargs.pop("list_colors")

    kwargs["verbose"] -= kwargs.pop("quiet", 0)

    if kwargs.pop("force"):
        kwargs["overwrite_existing"] = True
        kwargs["overwrite_ev_lines"] = True

    default_offset = kwargs.pop("offset")
    if kwargs["offset_turbulence"] is None:
        kwargs["offset_turbulence"] = default_offset
    if kwargs["offset_bottom"] is None:
        kwargs["offset_bottom"] = default_offset
    if kwargs["offset_surface"] is None:
        kwargs["offset_surface"] = default_offset

    if kwargs["hide_echoview"] is None:
        kwargs["hide_echoview"] = "never" if kwargs["minimize_echoview"] else "new"

    from ..inference import run_inference

    run_inference(**kwargs)


def main():
    """
    Run `cli`, with encapsulation for error messages.
    """
    try:
        cli()
    except KeyboardInterrupt as err:
        # Don't show stack traceback when KeyboardInterrupt is given.
        print(
            style.warning_fmt(
                "Interrupted by user while processing: {}".format(
                    style.highlight_fmt(" ".join(sys.argv)),
                )
            )
        )
        try:
            sys.exit(1)
        except SystemExit:
            os._exit(1)
    except:
        # Ensure all other errors are shown in red.
        with style.error_message():
            raise


if __name__ == "__main__":
    main()
