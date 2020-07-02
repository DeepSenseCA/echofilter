#!/usr/bin/env python

import datetime
import os
import pickle
import pprint
import shutil
import sys
import tempfile
import textwrap
import time
import urllib
import warnings

import numpy as np
from matplotlib import colors as mcolors
import pandas as pd
import torch
import torch.nn
import torch.utils.data
import torchvision.transforms
from torchvision.datasets.utils import download_url, download_file_from_google_drive
from torchutils.utils import count_parameters
from torchutils.device import cuda_is_really_available
from tqdm.auto import tqdm

import echofilter.data.transforms
import echofilter.nn
from echofilter.nn.unet import UNet
from echofilter.nn.wrapper import Echofilter
import echofilter.path
import echofilter.raw
from echofilter.raw.manipulate import join_transect, split_transect
import echofilter.ui
import echofilter.utils
import echofilter.win

from echofilter.ui.inference_cli import (
    CHECKPOINT_RESOURCES,
    DEFAULT_CHECKPOINT,
    DEFAULT_VARNAME,
    cli,
    main,
    get_default_cache_dir,
)


EV_UNDEFINED_DEPTH = -10000.99


def run_inference(
    paths,
    source_dir=".",
    recursive_dir_search=True,
    extensions="csv",
    skip_existing=False,
    skip_incompatible=False,
    output_dir="",
    dry_run=False,
    overwrite_existing=False,
    overwrite_ev_lines=False,
    import_into_evfile=True,
    generate_turbulence_line=True,
    generate_bottom_line=True,
    generate_surface_line=True,
    suffix_file="",
    suffix_var=None,
    color_turbulence="orangered",
    color_bottom="orangered",
    color_surface="green",
    thickness_turbulence=2,
    thickness_bottom=2,
    thickness_surface=1,
    cache_dir=None,
    cache_csv=None,
    suffix_csv="",
    keep_ext=False,
    line_status=3,
    offset_turbulence=1.0,
    offset_bottom=1.0,
    offset_surface=1.0,
    nearfield_cutoff=1.7,
    lines_during_passive="predict",
    collate_passive_length=10,
    collate_removed_length=10,
    minimum_passive_length=10,
    minimum_removed_length=10,
    minimum_patch_area=25,
    patch_mode=None,
    variable_name=DEFAULT_VARNAME,
    row_len_selector="mode",
    facing="auto",
    use_training_standardization=False,
    crop_min_depth=None,
    crop_max_depth=None,
    autocrop_threshold=0.35,
    image_height=None,
    checkpoint=None,
    force_unconditioned=False,
    logit_smoothing_sigma=1,
    device=None,
    hide_echoview="new",
    minimize_echoview=False,
    verbose=2,
):
    """
    Perform inference on input files, and write output lines in evl format.

    Parameters
    ----------
    paths : iterable
        Files and folders to be processed. These may be full paths or paths
        relative to `source_dir`. For each folder specified, any files with
        extension `'csv'` within the folder and all its tree of subdirectories
        will be processed.
    source_dir : str, optional
        Path to directory where files are found. Default is `'.'`.
    recursive_dir_search : bool, optional
        How to handle directory inputs in `paths`. If `False`, only files
        (with the correct extension) in the directory will be included.
        If `True`, subdirectories will also be walked through to find input
        files. Default is `True`.
    extensions : iterable or str, optional
        File extensions to detect when running on a directory. Default is
        `'csv'`.
    skip_existing : bool, optional
        Skip processing files which already have all outputs present. Default
        is `False`.
    skip_incompatible : bool, optional
        Skip processing CSV files which do not seem to contain an exported
        echoview transect. If `False`, an error is raised. Default is `False`.
    output_dir : str, optional
        Directory where output files will be written. If this is `''`, outputs
        are written to the same directory as each input file. Otherwise, they
        are written to `output_dir`, preserving their path relative to
        `source_dir` if relative paths were used. Default is `''`.
    dry_run : bool, optional
        If `True`, perform a trial run with no changes made. Default is
        `False`.
    overwrite_existing : bool, optional
        Overwrite existing outputs without producing a warning message. If
        `False`, an error is generated if files would be overwritten.
        Default is `False`.
    overwrite_ev_lines : bool, optional
        Overwrite existing lines within the EchoView file without warning.
        If `False` (default), the current datetime will be appended to line
        variable names in the event of a collision.
    import_into_evfile : bool, optional
        Whether to import the output lines and regions into the EV file,
        whenever the file being processed in an EV file. Default is `True`.
    generate_turbulence_line : bool, optional
        Whether to output an evl file for the turbulence line. If this is
        `False`, the turbulence line is also never imported into EchoView.
        Default is `True`.
    generate_bottom_line : bool, optional
        Whether to output an evl file for the bottom line. If this is `False`,
        the bottom line is also never imported into EchoView.
        Default is `True`.
    generate_surface_line : bool, optional
        Whether to output an evl file for the surface line. If this is `False`,
        the surface line is also never imported into EchoView.
        Default is `True`.
    suffix_file : str, optional
        Suffix to append to output artifacts (evl and evr files), between
        the name of the file and the extension. If `suffix_file` begins with
        an alphanumeric character, `"-"` is prepended. Default is `""`.
    suffix_var : str or None, optional
        Suffix to append to line and region names when imported back into
        EV file. If `suffix_var` begins with an alphanumeric character, `"-"`
        is prepended. If `None` (default), suffix_var will match `suffix_file`
        if it is set, and will be "_echofilter" otherwise.
    color_turbulence : str, optional
        Color to use for the turbulence line when it is imported into EchoView.
        This can either be the name of a supported color from
        matplotlib.colors, or a hexadecimal color, or a string representation
        of an RGB color to supply directly to EchoView (such as "(0,255,0)").
        Default is `"orangered"`.
    color_bottom : str, optional
        Color to use for the bottom line when it is imported into EchoView.
        This can either be the name of a supported color from
        matplotlib.colors, or a hexadecimal color, or a string representation
        of an RGB color to supply directly to EchoView (such as "(0,255,0)").
        Default is `"orangered"`.
    color_surface : str, optional
        Color to use for the surface line when it is imported into EchoView.
        This can either be the name of a supported color from
        matplotlib.colors, or a hexadecimal color, or a string representation
        of an RGB color to supply directly to EchoView (such as "(0,255,0)").
        Default is `"green"`.
    thickness_turbulence : int, optional
        Thicknesses with which the turbulence line will be displayed in EchoView.
        Default is `2`.
    thickness_bottom : int, optional
        Thicknesses with which the turbulence line will be displayed in EchoView.
        Default is `2`.
    thickness_surface : int, optional
        Thicknesses with which the turbulence line will be displayed in EchoView.
        Default is `1`.
    cache_dir : str or None, optional
        Path to directory where downloaded checkpoint files should be cached.
        If `None` (default), an OS-appropriate application-specific default
        cache directory is used.
    cache_csv : str or None, optional
        Path to directory where CSV files generated from EV inputs should be
        cached. If `None` (default), EV files which are exported to CSV files
        are temporary files, deleted after this program has completed. If
        `cache_csv=''`, the CSV files are cached in the same directory as the
        input EV files.
    suffix_csv : str, optional
        Suffix used for cached CSV files which are exported from EV files.
        If `suffix_file` begins with an alphanumeric character, a delimiter
        is prepended. The delimiter is `"."` if `keep_ext=True` or `"-"` if
        `keep_ext=False`. Default is `""`.
    keep_ext : bool, optional
        Whether to preserve the file extension in the input file name when
        generating output file name. Default is `False`, removing the
        extension.
    line_status : int, optional
        Status to use for the lines.
        Must be one of:
            `0` : none
            `1` : unverified
            `2` : bad
            `3` : good
        Default is `3`.
    offset_turbulence : float, optional
        Offset for turbulence line, which moves the turbulence line deeper. Default is `1.0`.
    offset_bottom : float, optional
        Offset for bottom line, which moves the line to become more shallow.
        Default is `1.0`.
    offset_surface : float, optional
        Offset for surface line, which moves the surface line deeper.
        Default is `1.0`.
    nearfield_cutoff : float or None, optional
        Nearest approach distance for line adjacent to echosounder, in meters.
        If the echosounder is downfacing, `nearfield_cutoff` is the minimum
        depth for the turbulence line. If the echosounder is upfacing, the maximum
        depth for the bottom line will be
        ``deepest_input_depth + interdepth_interval - nearfield_cutoff``.
        Set to `None` to disable. Default is `1.7`.
    lines_during_passive : str, optional
        Method used to handle line depths during collection
        periods determined to be passive recording instead of
        active recording.
        Options are:
            `"interpolate-time"`:
                depths are linearly interpolated from active
                recording periods, using the time at which
                recordings where made.
            `"interpolate-index"`:
                depths are linearly interpolated from active
                recording periods, using the index of the
                recording.
            `"predict"`:
                the model's prediction for the lines during
                passive data collection will be kept; the nature
                of the prediction depends on how the model was
                trained.
            `"redact"`:
                no depths are provided during periods determined
                to be passive data collection.
            `"undefined"`:
                depths are replaced with the placeholder value
                used by EchoView to denote undefined values,
                which is `-10000.99`.
        Default: "redact".
    collate_passive_length : int, optional
        Maximum interval, in ping indices, between detected passive regions
        which will removed to merge consecutive passive regions together
        into a single, collated, region. Default is 10.
    collate_passive_length : int, optional
        Maximum interval, in ping indices, between detected blocks
        (vertical rectangles) marked for removal which will also be removed
        to merge consecutive removed blocks together into a single,
        collated, region. Default is 10.
    minimum_passive_length : int, optional
        Minimum length, in ping indices, which a detected passive region
        must have to be included in the output. Set to -1 to omit all
        detected passive regions from the output. Default is 10.
    minimum_removed_length : int, optional
        Minimum length, in ping indices, which a detected removal block
        (vertical rectangle) must have to be included in the output.
        Set to -1 to omit all detected removal blocks from the output.
        Default is 10.
    minimum_patch_area : int, optional
        Minimum area, in pixels, which a detected removal patch
        (contour/polygon) region must have to be included in the output.
        Set to -1 to omit all detected patches from the output.
        Default is 25.
    patch_mode : str or None, optional
        Type of mask patches to use. Must be supported by the
        model checkpoint used. Should be one of:
            `"merged"`:
                Target patches for training were determined
                after merging as much as possible into the
                turbulence and bottom lines.
            `"original"`:
                Target patches for training were determined
                using original lines, before expanding the
                turbulence and bottom lines.
            `"ntob"`:
                Target patches for training were determined
                using the original bottom line and the merged
                turbulence line.
        If `None` (default), `"merged"` is used if downfacing and `"ntob"` is
        used if upfacing.
    variable_name : str, optional
        Name of the EchoView acoustic variable to load from EV files. Default
        is `'Fileset1: Sv pings T1'`.
    row_len_selector : str, optional
        Method used to handle input csv files with different number of Sv
        values across time (i.e. a non-rectangular input). Default is `'mode'`.
        See `echofilter.raw.loader.transect_loader` for options.
    facing : {"downward", "upward", "auto"}, optional
        Orientation in which the echosounder is facing. Default is `"auto"`,
        in which case the orientation is determined from the ordering of the
        depth values in the data (increasing = `"upward"`,
        decreasing = `"downward"`).
    use_training_standardization : bool, optional
        Whether to use the exact normalization center and deviation values as
        used during training. If `False` (default), the center and deviation
        are determined per sample, using the same method methodology as used
        to determine the center and deviation values for training.
    crop_min_depth : float or None, optional
        Minimum depth to include in input. If `None` (default), there is no
        minimum depth.
    crop_max_depth : float or None, optional
        Maxmimum depth to include in input. If `None` (default), there is no
        maximum depth.
    autocrop_threshold : float, optional
        Minimum fraction of input height which must be found to be removable
        for the model to be re-run with an automatically cropped input.
        Default is 0.35.
    image_height : int or None, optional
        Height in pixels of input to model. The data loaded from the csv will
        be resized to this height (the width of the image is unchanged).
        If `None` (default), the height matches that used when the model was
        trained.
    checkpoint : str or None, optional
        A path to a checkpoint file, or name of a checkpoint known to this
        package (listed in `CHECKPOINT_RESOURCES`). If `None` (default),
        the first checkpoint in `CHECKPOINT_RESOURCES` is used.
    force_unconditioned : bool, optional
        Whether to always use unconditioned logit outputs. If `False`
        (default) conditional logits will be used if the checkpoint loaded is
        for a conditional model.
    logit_smoothing_sigma : float, optional
        Standard deviation over which logits will be smoothed before being
        converted into output. Default is `1`.
    device : str or torch.device or None, optional
        Name of device on which the model will be run. If `None`, the first
        available CUDA GPU is used if any are found, and otherwise the CPU is
        used. Set to `'cpu'` to use the CPU even if a CUDA GPU is available.
    hide_echoview : {"never", "new", "always"}, optional
        Whether to hide the EchoView window entirely while the code runs.
        If `hide_echoview="new"`, the application is only hidden if it
        was created by this function, and not if it was already running.
        If `hide_echoview="always"`, the application is hidden even if it was
        already running. In the latter case, the window will be revealed again
        when this function is completed. Default is `"new"`.
    minimize_echoview : bool, optional
        If `True`, the EchoView window being used will be minimized while this
        function is running. Default is `False`.
    verbose : int, optional
        Verbosity level. Default is `2`. Set to `0` to disable print
        statements, or elevate to a higher number to increase verbosity.
    """

    t_start_prog = time.time()

    progress_fmt = (
        echofilter.ui.style.dryrun_fmt if dry_run else echofilter.ui.style.progress_fmt
    )

    existing_file_msg = (
        "Run with overwrite_existing=True (with the command line"
        " interface, use the --force flag) to overwrite existing"
        " outputs."
    )

    if verbose >= 1:
        print(
            progress_fmt(
                "{}Starting inference {}.{} {}".format(
                    echofilter.ui.style.HighlightStyle.start,
                    "dry-run" if dry_run else "routine",
                    echofilter.ui.style.HighlightStyle.reset,
                    datetime.datetime.now().strftime("%A, %B %d, %Y at %H:%M:%S"),
                )
            )
        )

    if device is None:
        device = "cuda" if cuda_is_really_available() else "cpu"
    device = torch.device(device)

    if facing is None:
        facing = "auto"
    if facing.startswith("down"):
        facing = "downward"
    if facing.startswith("up"):
        facing = "upward"

    if suffix_file and suffix_file[0].isalpha():
        suffix_file = "-" + suffix_file

    if suffix_var and suffix_var[0].isalpha():
        suffix_var = "-" + suffix_var
    elif suffix_var is not None:
        pass
    elif suffix_file:
        suffix_var = suffix_file
    else:
        suffix_var = "_echofilter"

    if suffix_csv and suffix_csv[0].isalpha():
        if keep_ext:
            suffix_csv = "." + suffix_csv
        else:
            suffix_csv = "-" + suffix_csv

    line_colors = dict(
        turbulence=color_turbulence, bottom=color_bottom, surface=color_surface
    )
    line_thicknesses = dict(
        turbulence=thickness_turbulence,
        bottom=thickness_bottom,
        surface=thickness_surface,
    )

    if checkpoint is None:
        # Use the first item from the list of checkpoints
        checkpoint = DEFAULT_CHECKPOINT

    ckpt_name = checkpoint

    if os.path.isfile(ckpt_name):
        ckpt_path = ckpt_name
    elif ckpt_name in CHECKPOINT_RESOURCES:
        ckpt_path = download_checkpoint(ckpt_name, cache_dir=cache_dir)
    else:
        msg = "The checkpoint parameter should either be a path to a file or "
        "one of \n{},\nbut {} was provided.".format(
            list(CHECKPOINT_RESOURCES.keys()), ckpt_name
        )
        with echofilter.ui.style.error_message(msg) as msg:
            raise ValueError(msg)

    if not os.path.isfile(ckpt_path):
        msg = "No checkpoint found at '{}'".format(ckpt_path)
        with echofilter.ui.style.error_message(msg) as msg:
            raise EnvironmentError(msg)
    if verbose >= 1:
        print("Loading model from checkpoint:\n  '{}'".format(ckpt_path))

    load_args = {}
    if device is not None:
        # Map model to be loaded to specified single gpu.
        load_args = dict(map_location=device)
    try:
        checkpoint = torch.load(ckpt_path, **load_args)
    except pickle.UnpicklingError:
        if ckpt_name not in CHECKPOINT_RESOURCES or ckpt_name == ckpt_path:
            # Direct path to checkpoint was given, so we shouldn't delete
            # the user's file
            msg = "Error: Unable to load checkpoint {}".format(
                os.path.abspath(ckpt_path)
            )
            with echofilter.ui.style.error_message(msg) as msg:
                print(msg)
                raise
        # Delete the checkpoint and try again, in case it is just a
        # malformed download (interrupted download, etc)
        os.remove(ckpt_path)
        ckpt_path = download_checkpoint(ckpt_name, cache_dir=cache_dir)
        checkpoint = torch.load(ckpt_path, **load_args)

    if image_height is None:
        image_height = checkpoint.get("sample_shape", (128, 512))[1]

    if use_training_standardization:
        center_param = checkpoint.get("data_center", -80.0)
        deviation_param = checkpoint.get("data_deviation", 20.0)
    else:
        center_param = checkpoint.get("center_method", "mean")
        deviation_param = checkpoint.get("deviation_method", "stdev")
    nan_value = checkpoint.get("nan_value", -3)

    if verbose >= 4:
        print("Constructing U-Net model, with arguments:")
        pprint.pprint(checkpoint["model_parameters"])
    unet = UNet(**checkpoint["model_parameters"])
    if hasattr(logit_smoothing_sigma, "__len__"):
        max_logit_smoothing_sigma = max(logit_smoothing_sigma)
    else:
        max_logit_smoothing_sigma = logit_smoothing_sigma
    if max_logit_smoothing_sigma > 0:
        ks = max(11, int(np.round(max_logit_smoothing_sigma * 6)))
        ks += (ks + 1) % 2  # Increment to odd number if even
        model = torch.nn.Sequential(
            unet,
            echofilter.nn.modules.GaussianSmoothing(
                channels=checkpoint["model_parameters"]["out_channels"],
                kernel_size=ks,
                sigma=logit_smoothing_sigma,
            ),
        )
    else:
        model = unet
    model = Echofilter(
        model,
        mapping=checkpoint.get("wrapper_mapping", None),
        **checkpoint.get("wrapper_params", {})
    )
    is_conditional_model = model.params.get("conditional", False)
    if verbose >= 3:
        print(
            "Built {}model with {} trainable parameters".format(
                "conditional " if is_conditional_model else "",
                count_parameters(model, only_trainable=True),
            )
        )
    try:
        unet.load_state_dict(checkpoint["state_dict"])
        if verbose >= 3:
            print(
                "Loaded U-Net state from the checkpoint".format(
                    ckpt_path, checkpoint["epoch"]
                )
            )
    except RuntimeError as err:
        if verbose >= 5:
            print(
                echofilter.ui.style.warning_fmt(
                    "Warning: Checkpoint doesn't seem to be for the UNet."
                    "Trying to load it as the whole model instead."
                )
            )
        try:
            model.load_state_dict(checkpoint["state_dict"])
            if verbose >= 3:
                print(
                    "Loaded model state from the checkpoint".format(
                        ckpt_path, checkpoint["epoch"]
                    )
                )
        except RuntimeError:
            msg = (
                "Could not load the checkpoint state as either the whole model"
                "or the unet component."
            )
            with echofilter.ui.style.error_message(msg) as msg:
                print(msg)
                raise err

    # Ensure model is on correct device
    model.to(device)
    # Put model in evaluation mode
    model.eval()

    files_input = paths
    files = list(
        echofilter.path.parse_files_in_folders(
            paths, source_dir, extensions, recursive=recursive_dir_search
        )
    )
    if verbose >= 1:
        print(
            progress_fmt(
                "Processing {}{} file{}{}...".format(
                    echofilter.ui.style.HighlightStyle.start,
                    len(files),
                    "" if len(files) == 1 else "s",
                    echofilter.ui.style.HighlightStyle.reset,
                )
            )
        )

    if len(extensions) == 1 and "ev" in extensions:
        do_open = True
    else:
        do_open = False
        for file in files:
            if os.path.splitext(file)[1].lower() == ".ev":
                do_open = True
                break

    if dry_run:
        if verbose >= 3:
            print(
                "EchoView application would{} be opened {}.".format(
                    "" if do_open else " not",
                    "to convert EV files to CSV"
                    if do_open
                    else "(no EV files to process)",
                )
            )
        do_open = False

    common_notes = textwrap.dedent(
        """
        Classified by echofilter {ver} at {dt}
        Model checkpoint: {ckpt_name}
        """.format(
            ver=echofilter.__version__,
            dt=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            ckpt_name=os.path.split(ckpt_name)[1],
        )
    )

    if len(files) == 1 or verbose <= 0:
        maybe_tqdm = lambda x: x
    else:
        maybe_tqdm = lambda x: tqdm(x, desc="Files", ascii=True)

    skip_count = 0
    incompatible_count = 0
    error_msgs = []

    # Open EchoView connection
    with echofilter.win.maybe_open_echoview(
        do_open=do_open, minimize=minimize_echoview, hide=hide_echoview,
    ) as ev_app:
        for fname in maybe_tqdm(files):
            if verbose >= 2:
                print(
                    progress_fmt(
                        "Processing {}".format(
                            echofilter.ui.style.highlight_fmt(fname),
                        )
                    )
                )

            # Check what the full path should be
            fname_full = echofilter.path.determine_file_path(fname, source_dir)

            # Determine where destination should be placed
            destination = echofilter.path.determine_destination(
                fname, fname_full, source_dir, output_dir
            )
            if not keep_ext:
                destination = os.path.splitext(destination)[0]

            # Make a list of all the outputs we will produce
            dest_files = {}
            for name in ("turbulence", "bottom", "surface"):
                dest_files[name] = "{}.{}{}.evl".format(destination, name, suffix_file)
            if not generate_turbulence_line:
                dest_files.pop("turbulence")
            if not generate_bottom_line:
                dest_files.pop("bottom")
            if not generate_surface_line:
                dest_files.pop("surface")
            dest_files["regions"] = "{}.{}{}.evr".format(
                destination, "regions", suffix_file
            )

            # Check if any of them exists and if there is any missing
            any_exists = False
            any_missing = False
            first_clobber = None
            for k, dest_file in dest_files.items():
                if os.path.isfile(dest_file):
                    any_exists = True
                    if first_clobber is None:
                        first_clobber = dest_file
                else:
                    any_missing = True

            # Check whether to skip processing this file
            if skip_existing and not any_missing:
                if verbose >= 2:
                    print(echofilter.ui.style.skip_fmt("  Skipping {}".format(fname)))
                skip_count += 1
                continue
            # Check whether we would clobber a file we can't overwrite
            if any_exists and not overwrite_existing:
                msg = "Output {} for {} already exists.".format(first_clobber, fname)
                with echofilter.ui.style.error_message(msg) as msg:
                    if dry_run:
                        error_msgs.append("Error: " + msg + "\n  " + existing_file_msg)
                        print(error_msgs[-1])
                        continue
                    raise EnvironmentError(msg + "\n  " + existing_file_msg)

            # Determine whether we need to run ev2csv on this file
            ext = os.path.splitext(fname)[1]
            if len(ext) > 0:
                ext = ext[1:].lower()
            if ext == "csv":
                process_as_ev = False
            elif ext == "ev":
                process_as_ev = True
            elif len(extensions) == 1 and "csv" in extensions:
                process_as_ev = False
            elif len(extensions) == 1 and "ev" in extensions:
                process_as_ev = True
            else:
                msg = "Unsure how to process file {} with unrecognised extension {}".format(
                    fname, ext
                )
                if not skip_incompatible:
                    with echofilter.ui.style.error_message(msg) as msg:
                        raise EnvironmentError(msg)
                if verbose >= 2:
                    print(
                        echofilter.ui.style.skip_fmt(
                            "  Skipping incompatible file {}".format(fname)
                        )
                    )
                incompatible_count += 1
                continue

            # If we are processing an ev file, we need to export it as a raw
            # csv file. Unless it has already been exported (which we will
            # check for below).
            export_to_csv = process_as_ev

            # Make a temporary directory in case we are not caching generated csvs
            # Directory and all its contents are deleted when we leave this context
            with tempfile.TemporaryDirectory() as tmpdirname:

                # Convert ev file to csv, if necessary
                ev2csv_dir = cache_csv
                if ev2csv_dir is None:
                    ev2csv_dir = tmpdirname

                if not export_to_csv:
                    csv_fname = fname_full
                else:
                    # Determine where exported CSV file should be placed
                    csv_fname = echofilter.path.determine_destination(
                        fname, fname_full, source_dir, ev2csv_dir
                    )
                    if not keep_ext:
                        csv_fname = os.path.splitext(csv_fname)[0]
                    csv_fname += suffix_csv + ".csv"

                if os.path.isfile(csv_fname):
                    # If CSV file is already cached, no need to re-export it
                    export_to_csv = False

                if not export_to_csv:
                    pass
                elif dry_run:
                    if verbose >= 1:
                        print(
                            "  Would export {} as CSV file {}".format(
                                fname_full, csv_fname
                            )
                        )
                else:
                    # Import ev2csv now. We delay this import so Linux users
                    # without pywin32 can run on CSV files.
                    from echofilter.ev2csv import ev2csv

                    # Export the CSV file
                    fname_full = os.path.abspath(fname_full)
                    csv_fname = os.path.abspath(csv_fname)
                    ev2csv(
                        fname_full,
                        csv_fname,
                        variable_name=variable_name,
                        ev_app=ev_app,
                        verbose=verbose - 1,
                    )

                if (dry_run and verbose >= 2) or verbose >= 3:
                    ww = "Would" if dry_run else "Will"
                    print("  {} write files:".format(ww))
                    for key, fname in dest_files.items():
                        if os.path.isfile(fname) and overwrite_existing:
                            over_txt = " " + echofilter.ui.style.overwrite_fmt(
                                "(overwriting existing file)"
                            )
                        else:
                            over_txt = ""
                        tp = "line" if os.path.splitext(fname)[1] == ".evl" else "file"
                        print(
                            "    {} export {} {} to: {}{}".format(
                                ww, key, tp, fname, over_txt,
                            )
                        )
                if dry_run:
                    continue

                # Load the data
                if verbose >= 6:
                    warn_row_overflow = np.inf
                elif verbose >= 5:
                    warn_row_overflow = None
                else:
                    warn_row_overflow = 0
                try:
                    timestamps, depths, signals = echofilter.raw.loader.transect_loader(
                        csv_fname,
                        warn_row_overflow=warn_row_overflow,
                        row_len_selector=row_len_selector,
                    )
                except KeyError:
                    if skip_incompatible and fname not in files_input:
                        if verbose >= 2:
                            print(
                                echofilter.ui.style.skip_fmt(
                                    "  Skipping incompatible file {}".format(fname)
                                )
                            )
                        incompatible_count += 1
                        continue
                    msg = "CSV file {} could not be loaded.".format(fname)
                    with echofilter.ui.style.error_message(msg) as msg:
                        print(msg)
                        raise

            output = inference_transect(
                model,
                timestamps,
                depths,
                signals,
                device,
                image_height,
                facing=facing,
                crop_min_depth=crop_min_depth,
                crop_max_depth=crop_max_depth,
                autocrop_threshold=autocrop_threshold,
                force_unconditioned=force_unconditioned,
                data_center=center_param,
                data_deviation=deviation_param,
                nan_value=nan_value,
                verbose=verbose - 1,
            )
            if verbose >= 5:
                s = "\n    ".join([""] + list(str(k) for k in output.keys()))
                print("  Generated model output with fields:" + s)

            if is_conditional_model and not force_unconditioned:
                if output["is_upward_facing"]:
                    cs = "|upfacing"
                else:
                    cs = "|downfacing"
                if verbose >= 4:
                    print(
                        echofilter.ui.style.aside_fmt(
                            "  Using conditional probability outputs from model:"
                            " p(state{})".format(cs)
                        )
                    )
            else:
                cs = ""
                if is_conditional_model and verbose >= 4:
                    print(
                        echofilter.ui.style.aside_fmt(
                            "Using unconditioned output from conditional model"
                        )
                    )

            # Convert output into lines
            surface_depths = output["depths"][
                echofilter.utils.last_nonzero(
                    output["p_is_above_surface" + cs] > 0.5, -1
                )
            ]
            turbulence_depths = output["depths"][
                echofilter.utils.last_nonzero(
                    output["p_is_above_turbulence" + cs] > 0.5, -1
                )
            ]
            bottom_depths = output["depths"][
                echofilter.utils.first_nonzero(
                    output["p_is_below_bottom" + cs] > 0.5, -1
                )
            ]
            # Offset lines
            turbulence_depths += offset_turbulence
            bottom_depths -= offset_bottom
            surface_depths += offset_surface
            # Redact passive regions
            line_timestamps = output["timestamps"].copy()
            is_passive = output["p_is_passive" + cs] > 0.5
            if lines_during_passive == "predict":
                pass
            elif lines_during_passive == "redact":
                surface_depths = surface_depths[~is_passive]
                turbulence_depths = turbulence_depths[~is_passive]
                bottom_depths = bottom_depths[~is_passive]
                line_timestamps = line_timestamps[~is_passive]
            elif lines_during_passive == "undefined":
                surface_depths[is_passive] = EV_UNDEFINED_DEPTH
                turbulence_depths[is_passive] = EV_UNDEFINED_DEPTH
                bottom_depths[is_passive] = EV_UNDEFINED_DEPTH
            elif lines_during_passive.startswith("interp"):
                if lines_during_passive == "interpolate-time":
                    x = line_timestamps
                elif lines_during_passive == "interpolate-index":
                    x = np.arange(len(line_timestamps))
                else:
                    msg = "Unsupported passive line interpolation method: {}".format(
                        lines_during_passive
                    )
                    with echofilter.ui.style.error_message(msg) as msg:
                        raise ValueError(msg)

                if len(x[~is_passive]) == 0:
                    if verbose >= 0:
                        s = (
                            "Could not interpolate depths for passive data for"
                            " {}, as all data appears to be from passive"
                            " collection. The original model predictions will"
                            " be kept instead.".format(fname)
                        )
                        warnings.warn(s)
                else:
                    surface_depths[is_passive] = np.interp(
                        x[is_passive], x[~is_passive], surface_depths[~is_passive]
                    )
                    turbulence_depths[is_passive] = np.interp(
                        x[is_passive], x[~is_passive], turbulence_depths[~is_passive]
                    )
                    bottom_depths[is_passive] = np.interp(
                        x[is_passive], x[~is_passive], bottom_depths[~is_passive]
                    )
            else:
                msg = "Unsupported passive line method: {}".format(lines_during_passive)
                with echofilter.ui.style.error_message(msg) as msg:
                    raise ValueError(msg)

            if nearfield_cutoff is None:
                pass
            elif output["is_upward_facing"]:
                depth_intv = abs(depths[-1] - depths[-2])
                max_depth = np.max(depths) + depth_intv - nearfield_cutoff
                bottom_depths = np.minimum(max_depth, bottom_depths)
            else:
                turbulence_depths = np.maximum(turbulence_depths, nearfield_cutoff)

            # Export evl files
            destination_dir = os.path.dirname(destination)
            if destination_dir != "":
                os.makedirs(destination_dir, exist_ok=True)
            for line_name, line_depths in (
                ("turbulence", turbulence_depths),
                ("bottom", bottom_depths),
                ("surface", surface_depths),
            ):
                if line_name not in dest_files:
                    continue
                dest_file = dest_files[line_name]
                if verbose >= 3:
                    s = "  Writing output"
                    if not os.path.exists(dest_file):
                        pass
                    elif not overwrite_existing:
                        s = echofilter.ui.style.error_fmt(s)
                    else:
                        s = echofilter.ui.style.overwrite_fmt(s)
                    s += " {}".format(dest_file)
                    print(s)
                if os.path.exists(dest_file) and not overwrite_existing:
                    msg = "Output {} already exists.".format(dest_file)
                    with echofilter.ui.style.error_message(msg) as msg:
                        raise EnvironmentError(msg + "\n  " + existing_file_msg)

                echofilter.raw.loader.evl_writer(
                    dest_file, line_timestamps, line_depths, status=line_status
                )
            # Export evr file
            dest_file = dest_files["regions"]
            if verbose >= 3:
                s = "  Writing output"
                if not os.path.exists(dest_file):
                    pass
                elif not overwrite_existing:
                    s = echofilter.ui.style.error_fmt(s)
                else:
                    s = echofilter.ui.style.overwrite_fmt(s)
                s += " {}".format(dest_file)
                print(s)
            if os.path.exists(dest_file) and not overwrite_existing:
                msg = "Output {} already exists.".format(dest_file)
                with echofilter.ui.style.error_message(msg) as msg:
                    raise EnvironmentError(msg + "\n  " + existing_file_msg)

            patches_key = "p_is_patch"
            if patch_mode is None:
                if output["is_upward_facing"]:
                    patches_key += "-ntob"
            elif patch_mode != "merged":
                patches_key += "-" + patch_mode

            echofilter.raw.loader.write_transect_regions(
                dest_file,
                output,
                depth_range=depths,
                passive_key="p_is_passive" + cs,
                removed_key="p_is_removed" + cs,
                patches_key=patches_key + cs,
                collate_passive_length=collate_passive_length,
                collate_removed_length=collate_removed_length,
                minimum_passive_length=minimum_passive_length,
                minimum_removed_length=minimum_removed_length,
                minimum_patch_area=minimum_patch_area,
                name_suffix=suffix_var,
                common_notes=common_notes,
                verbose=verbose - 2,
                verbose_indent=2,
            )

            if not process_as_ev or not import_into_evfile:
                # Done with processing this file
                continue

            import_lines_regions_to_ev(
                fname_full,
                dest_files,
                target_names={key: key + suffix_var for key in dest_files},
                line_colors=line_colors,
                line_thicknesses=line_thicknesses,
                ev_app=ev_app,
                overwrite=overwrite_ev_lines,
                verbose=verbose,
            )

    if verbose >= 1:
        print(
            progress_fmt(
                "Finished {}processing {}{} file{}{}.".format(
                    "simulating " if dry_run else "",
                    echofilter.ui.style.HighlightStyle.start,
                    len(files),
                    "" if len(files) == 1 else "s",
                    echofilter.ui.style.HighlightStyle.reset,
                )
            )
        )
        skip_total = skip_count + incompatible_count
        if skip_total > 0:
            s = ""
            s += "Of these, {}{}{} file{} skipped{}: {} already processed".format(
                "all " if skip_total == len(files) else "",
                echofilter.ui.style.HighlightStyle.start,
                skip_total,
                " was" if skip_total == 1 else "s were",
                echofilter.ui.style.HighlightStyle.reset,
                skip_count,
            )
            if not dry_run:
                s += ", {} incompatible".format(incompatible_count)
            s += "."
            s = echofilter.ui.style.skip_fmt(s)
            print(s)
        if error_msgs:
            print(
                echofilter.ui.style.error_fmt(
                    "There {} {} error{}:".format(
                        "was" if len(error_msgs) == 1 else "were",
                        len(error_msgs),
                        "" if len(error_msgs) == 1 else "s",
                    )
                )
            )
            for error_msg in error_msgs:
                print(echofilter.ui.style.error_fmt(error_msg))
        print(
            "Total runtime: {}".format(
                datetime.timedelta(seconds=time.time() - t_start_prog)
            )
        )


def inference_transect(
    model,
    timestamps,
    depths,
    signals,
    device,
    image_height,
    facing="auto",
    crop_min_depth=None,
    crop_max_depth=None,
    autocrop_threshold=0.35,
    force_unconditioned=False,
    data_center="mean",
    data_deviation="stdev",
    nan_value=-3,
    dtype=torch.float,
    verbose=0,
):
    """
    Run inference on a single transect.

    Parameters
    ----------
    model : echofilter.wrapper.Echofilter
        A pytorch Module wrapped in an Echofilter UI layer.
    timestamps : array_like
        Sample recording timestamps (in seconds since Unix epoch). Must be a
        vector.
    depths : array_like
        Recording depths from the surface (in metres). Must be a vector.
    signals : array_like
        Echogram Sv data. Must be a matrix shaped
        `(len(timestamps), len(depths))`.
    image_height : int
        Height to resize echogram before passing through model.
    facing : {"downward", "upward", "auto"}, optional
        Orientation in which the echosounder is facing. Default is `"auto"`,
        in which case the orientation is determined from the ordering of the
        depth values in the data (increasing = `"upward"`,
        decreasing = `"downward"`).
    crop_min_depth : float or None, optional
        Minimum depth to include in input. If `None` (default), there is no
        minimum depth.
    crop_max_depth : float or None, optional
        Maxmimum depth to include in input. If `None` (default), there is no
        maximum depth.
    autocrop_threshold : float, optional
        Minimum fraction of input height which must be found to be removable
        for the model to be re-run with an automatically cropped input.
        Default is 0.35.
    force_unconditioned : bool, optional
        Whether to always use unconditioned logit outputs when deteriming the
        new depth range for automatic cropping.
    data_center : float or str, optional
        Center point to use, which will be subtracted from the Sv signals
        (i.e. the overall sample mean).
        If `data_center` is a string, it specifies the method to use to
        determine the center value from the distribution of intensities seen
        in this sample transect. Default is `'mean'`.
    data_deviation : float or str, optional
        Deviation to use to normalise the Sv signals in divisive manner
        (i.e. the overall sample standard deviation).
        If `data_deviation` is a string, it specifies the method to use to
        determine the center value from the distribution of intensities seen
        in this sample transect. Default is `'stdev'`.
    nan_value : float, optional
        Placeholder value to replace NaNs with. Default is `-3`.
    dtype : torch.dtype, optional
        Datatype to use for model input. Default is `torch.float`.
    verbose : int, optional
        Level of verbosity. Default is `0`.

    Returns
    -------
    dict
        Dictionary with fields as output by `echofilter.wrapper.Echofilter`,
        plus `timestamps` and `depths`.
    """
    facing = facing.lower()
    timestamps = np.asarray(timestamps)
    depths = np.asarray(depths)
    signals = np.asarray(signals)
    transect = {
        "timestamps": timestamps,
        "depths": depths,
        "signals": signals,
    }
    if crop_min_depth is not None:
        # Apply minimum depth crop
        depth_crop_mask = transect["depths"] >= crop_min_depth
        transect["depths"] = transect["depths"][depth_crop_mask]
        transect["signals"] = transect["signals"][:, depth_crop_mask]
    if crop_max_depth is not None:
        # Apply maximum depth crop
        depth_crop_mask = transect["depths"] <= crop_max_depth
        transect["depths"] = transect["depths"][depth_crop_mask]
        transect["signals"] = transect["signals"][:, depth_crop_mask]

    # Standardize data distribution
    transect = echofilter.data.transforms.Normalize(data_center, data_deviation)(
        transect
    )

    # Configure data to match what the model expects to see
    # Determine whether depths are ascending or descending
    is_upward_facing = transect["depths"][-1] < transect["depths"][0]
    # Ensure depth is always increasing (which corresponds to descending from
    # the air down the water column)
    if facing[:2] == "up" or (facing == "auto" and is_upward_facing):
        transect["depths"] = transect["depths"][::-1].copy()
        transect["signals"] = transect["signals"][:, ::-1].copy()
        if facing == "auto" and verbose >= 2:
            print(
                echofilter.ui.style.aside_fmt(
                    "  Echogram was autodetected as upward facing, and was flipped"
                    " vertically before being input into the model."
                )
            )
        if not is_upward_facing:
            print(
                echofilter.ui.style.warning_fmt(
                    'Warning: facing = "{}" was provided, but data appears to be'
                    " downward facing".format(facing)
                )
            )
        is_upward_facing = True
    elif facing[:4] != "down" and facing != "auto":
        msg = 'facing should be one of "downward", "upward", and "auto"'
        with echofilter.ui.style.error_message(msg) as msg:
            raise ValueError(msg)
    elif facing[:4] == "down" and is_upward_facing:
        print(
            echofilter.ui.style.warning_fmt(
                'Warning: facing = "{}" was provided, but data appears to be'
                " upward facing".format(facing)
            )
        )
        is_upward_facing = False
    elif facing == "auto" and verbose >= 2:
        print(
            echofilter.ui.style.aside_fmt(
                "  Echogram was autodetected as downward facing."
            )
        )

    # To reduce memory consumption, split into segments whenever the recording
    # interval is longer than normal
    segments = split_transect(**transect)
    if verbose >= 1:
        maybe_tqdm = lambda x: tqdm(list(x), desc="  Segments", position=0, ascii=True)
    else:
        maybe_tqdm = lambda x: x
    outputs = []
    for segment in maybe_tqdm(segments):
        # Preprocessing transform
        transform = torchvision.transforms.Compose(
            [
                echofilter.data.transforms.ReplaceNan(nan_value),
                echofilter.data.transforms.Rescale(
                    (segment["signals"].shape[0], image_height), order=1,
                ),
            ]
        )
        segment = transform(segment)
        input = torch.tensor(segment["signals"]).unsqueeze(0).unsqueeze(0)
        input = input.to(device, dtype).contiguous()
        # Put data through model
        with torch.no_grad():
            output = model(input)
            output = {k: v.squeeze(0).cpu().numpy() for k, v in output.items()}
        output["timestamps"] = segment["timestamps"]
        output["depths"] = segment["depths"]
        outputs.append(output)

    output = join_transect(outputs)
    output["is_upward_facing"] = is_upward_facing

    if autocrop_threshold >= 1:
        # If we are not doing auto-crop, return the current output
        return output

    # See how much of the depth we could crop away
    cs = ""
    if model.params["conditional"] and not force_unconditioned:
        if output["is_upward_facing"]:
            cs = "|upfacing"
        else:
            cs = "|downfacing"
    is_passive = output["p_is_passive" + cs] > 0.5
    depth_intv = abs(transect["depths"][1] - transect["depths"][0])
    if is_upward_facing:
        # Restrict the top based on the observed surface depths
        surface_depths = output["depths"][
            echofilter.utils.last_nonzero(output["p_is_above_surface" + cs] > 0.5, -1)
        ]
        # Redact passive regions
        if len(surface_depths[~is_passive]) > 0:
            surface_depths = surface_depths[~is_passive]
        # Find a good estimator of the maximum. Go for a robust estimate of
        # four sigma from the middle.
        pct = np.percentile(surface_depths, [2.275, 97.725])
        dev = (pct[1] - pct[0]) / 4
        new_crop_min = pct[0] - 2 * dev
        # Don't go higher than the minimum of the predicted surface depths
        new_crop_min = max(new_crop_min, np.min(surface_depths))
        # Offset minimum by at least 2m, to be sure we are capturing enough
        # surrounding content.
        new_crop_min -= max(2, 10 * depth_intv)
        new_crop_max = np.max(transect["depths"])
    else:
        new_crop_min = np.min(transect["depths"])
        # Restrict the bottom based on the observed bottom depths
        bottom_depths = output["depths"][
            echofilter.utils.first_nonzero(output["p_is_below_bottom" + cs] > 0.5, -1)
        ]
        # Redact passive regions
        if len(bottom_depths[~is_passive]) > 0:
            bottom_depths = bottom_depths[~is_passive]
        # Find a good estimator of the maximum. Go for a robust estimate of
        # four sigma from the middle.
        pct = np.percentile(bottom_depths, [2.275, 97.725])
        dev = (pct[1] - pct[0]) / 4
        new_crop_max = pct[1] + 2 * dev
        # Don't go deeper than the maximum of the predicted bottom depths
        new_crop_max = min(new_crop_max, np.max(bottom_depths))
        # Offset maximum by at least 2m, to be sure we are capturing enough
        # surrounding content.
        new_crop_max += max(2, 10 * depth_intv)

    if crop_min_depth is not None:
        new_crop_min = max(new_crop_min, crop_min_depth)
    if crop_max_depth is not None:
        new_crop_max = min(new_crop_max, crop_max_depth)

    current_height = abs(transect["depths"][-1] - transect["depths"][0])
    new_height = abs(new_crop_max - new_crop_min)
    if (current_height - new_height) / current_height <= autocrop_threshold:
        # No need to crop
        return output

    if verbose >= 1:
        print(
            "  Automatically zooming in on the {:.2f}m to {:.2f}m depth range"
            " and re-doing model inference.".format(new_crop_min, new_crop_max)
        )

    # Crop and run again; and don't autocrop a second time!
    return inference_transect(
        model,
        timestamps,
        depths,
        signals,
        device,
        image_height,
        facing=facing,
        crop_min_depth=new_crop_min,
        crop_max_depth=new_crop_max,
        autocrop_threshold=1,  # Don't crop again
        force_unconditioned=force_unconditioned,
        data_center=data_center,
        data_deviation=data_deviation,
        nan_value=nan_value,
        dtype=dtype,
        verbose=verbose,
    )


def import_lines_regions_to_ev(
    ev_fname,
    files,
    target_names={},
    line_colors={},
    line_thicknesses={},
    ev_app=None,
    overwrite=False,
    verbose=1,
):
    """
    Write lines and regions to EV file.

    Parameters
    ----------
    ev_fname : str
        Path to EchoView file to import variables into.
    files : dict
        Mapping from output keys to filenames.
    target_names : dict, optional
        Mapping from output keys to output variable names.
    line_colors : dict, optional
        Mapping from output keys to line colours.
    line_thicknesses : dict, optional
        Mapping from output keys to line thicknesses.
    ev_app : win32com.client.Dispatch object or None, optional
        An object which can be used to interface with the EchoView application,
        as returned by `win32com.client.Dispatch`. If `None` (default), a
        new instance of the application is opened (and closed on completion).
    overwrite : bool, optional
        Whether existing lines with target names should be replaced.
        If a line with the target name already exists and `overwrite=False`,
        the line is named with the current datetime to prevent collisions.
        Default is `False`.
    verbose : int, optional
        Verbosity level. Default is `1`.
    """
    if verbose >= 2:
        print("Importing {} lines/regions into EV file {}".format(len(files), ev_fname))

    # Assemble the color palette
    colors = get_color_palette()

    with echofilter.win.open_ev_file(ev_fname, ev_app) as ev_file:
        for key, fname in files.items():
            # Import the line into the EV file
            fname_full = os.path.abspath(fname)
            if not os.path.isfile(fname_full):
                print(
                    echofilter.ui.style.warning_fmt(
                        "Warning: File '{}' could not be found".format(fname_full)
                    )
                )
                continue
            is_imported = ev_file.Import(fname_full)
            if not is_imported:
                print(
                    echofilter.ui.style.warning_fmt(
                        "Warning: Unable to import file '{}'"
                        "Please consult EchoView for the Import error message.".format(
                            fname
                        )
                    )
                )
                continue

            if os.path.splitext(fname)[1].lower() != ".evl":
                # Further handling is only for lines, so skip if this wasn't
                # a line
                continue

            # Identify line just imported, which is the last variable
            variable = ev_file.Variables[ev_file.Variables.Count - 1]
            lines = ev_file.Lines
            line = lines.FindByName(variable.Name)
            if not line:
                print(
                    echofilter.ui.style.warning_fmt(
                        "Warning: Could not find line which was just imported with"
                        " name '{}'".format(variable.Name)
                    )
                )
                print("Ignoring and continuing processing.")
                continue

            # Check whether we need to change the name of the line
            target_name = target_names.get(key, None)

            # Check if a line with this name already exists
            old_line = None if target_name is None else lines.FindByName(target_name)

            # Maybe try to overwrite existing line with new line
            successful_overwrite = False
            if old_line and overwrite:
                # Overwrite the old line
                if verbose >= 2:
                    print(
                        echofilter.ui.style.overwrite_fmt(
                            "Overwriting existing line '{}' with new {} line"
                            " output".format(target_name, key)
                        )
                    )
                old_line_edit = old_line.AsLineEditable
                if old_line_edit:
                    # Overwrite the old line with the new line
                    old_line_edit.OverwriteWith(line)
                    # Delete the line we imported
                    lines.Delete(line)
                    # Update our reference to the line
                    line = old_line
                    successful_overwrite = True
                elif verbose >= 0:
                    # Line is not editable
                    print(
                        echofilter.ui.style.warning_fmt(
                            "Existing line '{}' is not editable and cannot be"
                            " overwritten.".format(target_name, key)
                        )
                    )

            if old_line and not successful_overwrite:
                # Change the name so there is no collision
                target_name += "_{}".format(
                    datetime.datetime.now().isoformat(timespec="seconds")
                )
                if verbose >= 1:
                    print(
                        "Target line name '{}' already exists. Will save"
                        " new {} line with name '{}' instead.".format(
                            target_names[key], key, target_name
                        )
                    )

            if target_name and not successful_overwrite:
                # Rename the line
                variable.ShortName = target_name
                line.Name = target_name

            # Change the color and thickness of the line
            if key in line_colors or key in line_thicknesses:
                ev_app.Exec(
                    "{} | UseDefaultLineDisplaySettings =| false".format(line.Name)
                )
            if key in line_colors:
                color = line_colors[key]
                if color in colors:
                    color = colors[color]
                elif not isinstance(color, str):
                    pass
                elif "xkcd:" + color in colors:
                    color = colors["xkcd:" + color]
                color = hexcolor2rgb8(color)
                color = repr(color).replace(" ", "")
                ev_app.Exec("{} | CustomGoodLineColor =| {}".format(line.Name, color))
            if key in line_thicknesses:
                ev_app.Exec(
                    "{} | CustomLineDisplayThickness =| {}".format(
                        line.Name, line_thicknesses[key]
                    )
                )

        # Overwrite the EV file now the outputs have been imported
        ev_file.Save()


def get_color_palette(include_xkcd=True):
    """
    Provide a mapping of named colors from matplotlib.

    Parameters
    ----------
    include_xkcd : bool, optional
        Whether to include the XKCD color palette in the output.
        Note that XKCD colors have `"xkcd:"` prepended to their names to
        prevent collisions with official named colors from CSS4.
        Default is `True`.
        See https://xkcd.com/color/rgb/ and
        https://blog.xkcd.com/2010/05/03/color-survey-results/
        for the XKCD colors.

    Returns
    -------
    colors : dict
        Mapping from names of colors as strings to color value, either as
        an RGB tuple (fractional, 0 to 1 range) or a hexadecimal string.
    """
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    if include_xkcd:
        colors.update(**mcolors.XKCD_COLORS)
    return colors


def hexcolor2rgb8(color):
    """
    Utility for mapping hexadecimal colors to uint8 RGB.

    Parameters
    ----------
    color : str
        A hexadecimal color string, with leading "#".
        If the input is not a string beginning with "#", it is returned as-is
        without raising an error.

    Returns
    -------
    tuple
        RGB color tuple, in uint8 format (0-255).
    """
    if color[0] is "#":
        color = mcolors.to_rgba(color)[:3]
        color = tuple(max(0, min(255, int(np.round(c * 255)))) for c in color)
    return color


def download_checkpoint(checkpoint_name, cache_dir=None, verbose=1):
    """
    Download a checkpoint if it isn't already cached.

    Parameters
    ----------
    checkpoint_name : str
        Name of checkpoint to download.
    cache_dir : str or None, optional
        Path to local cache directory. If `None` (default), an OS-appropriate
        application-specific default cache directory is used.
    verbose : int, optional
        Verbosity level. Default is `1`. Set to `0` to disable print
        statements.

    Returns
    -------
    str
        Path to downloaded checkpoint file.
    """
    if cache_dir is None:
        cache_dir = get_default_cache_dir()

    destination = os.path.join(cache_dir, checkpoint_name)

    if os.path.exists(destination):
        return destination

    os.makedirs(cache_dir, exist_ok=True)

    sources = CHECKPOINT_RESOURCES[checkpoint_name]
    success = False
    for key, url_or_id in sources.items():
        if key == "gdrive":
            if verbose >= 1:
                print(
                    "Downloading checkpoint {} from GDrive...".format(checkpoint_name)
                )
            try:
                download_file_from_google_drive(
                    url_or_id, cache_dir, filename=checkpoint_name
                )
                success = True
                continue
            except (pickle.UnpicklingError, urllib.error.URLError):
                if verbose >= 1:
                    print(
                        echofilter.ui.style.error_fmt(
                            "\nCould not download checkpoint {} from GDrive!".format(
                                checkpoint_name
                            )
                        )
                    )
        else:
            if verbose >= 1:
                print(
                    "Downloading checkpoint {} from {}...".format(
                        checkpoint_name, url_or_id
                    )
                )
            try:
                download_url(url_or_id, cache_dir, filename=checkpoint_name)
                success = True
                continue
            except (pickle.UnpicklingError, urllib.error.URLError):
                if verbose >= 1:
                    print(
                        echofilter.ui.style.error_fmt(
                            "\nCould not download checkpoint {} from {}".format(
                                checkpoint_name, url_or_id
                            )
                        )
                    )

    if not success:
        msg = "Unable to download {} from {}".format(checkpoint_name, sources)
        with echofilter.ui.style.error_message(msg) as msg:
            raise OSError(msg)

    if verbose >= 1:
        print("Downloaded checkpoint to {}".format(destination))

    return destination


if __name__ == "__main__":
    main()
