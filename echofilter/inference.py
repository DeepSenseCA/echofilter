#!/usr/bin/env python

from collections import OrderedDict
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

import appdirs
import numpy as np
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
import echofilter.path
from echofilter.nn.unet import UNet
from echofilter.nn.wrapper import Echofilter
import echofilter.raw
from echofilter.raw.manipulate import join_transect, split_transect
import echofilter.utils
import echofilter.win


CHECKPOINT_RESOURCES = OrderedDict(
    [
        (
            "stationary_effunet_block6.xb.2-1_lc32_se2_v2.ckpt.tar",
            {"gdrive": "1Rgr6y7SYEYrAq6tSF7tjqbKoZthKpMCb"},
        ),
        (
            "stationary_effunet_block6.xb.2-1_lc32_se2.ckpt.tar",
            {"gdrive": "114vL-pAxrn9UDhaNG5HxZwjxNy7WMfW_"},
        ),
    ]
)
DEFAULT_CHECKPOINT = next(iter(CHECKPOINT_RESOURCES))

DEFAULT_VARNAME = "Fileset1: Sv pings T1"
EV_UNDEFINED_DEPTH = -10000.99


def run_inference(
    paths,
    source_dir=".",
    extensions="csv",
    skip_existing=False,
    skip_incompatible=False,
    output_dir="",
    dry_run=False,
    overwrite_existing=False,
    cache_dir=None,
    cache_csv=None,
    csv_suffix=".csv",
    keep_ext=False,
    line_status=3,
    offset_top=0.0,
    offset_bottom=0.0,
    lines_during_passive="redact",
    passive_collate_length=10,
    removed_collate_length=10,
    minimum_passive_length=10,
    minimum_removed_length=10,
    minimum_patch_area=25,
    patch_mode=None,
    variable_name=DEFAULT_VARNAME,
    row_len_selector="mode",
    facing="auto",
    use_training_standardization=False,
    crop_depth_min=None,
    crop_depth_max=None,
    image_height=None,
    checkpoint=None,
    device=None,
    hide_echoview="new",
    minimize_echoview=False,
    verbose=1,
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
    csv_suffix : str, optional
        Suffix used for cached CSV files which are exported from EV files.
        Default is `'.csv'` (only the file extension is changed).
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
    offset_top : float, optional
        Offset for top line, which moves the top line deeper. Default is `0`.
    offset_bottom : float, optional
        Offset for bottom line, which moves the line to become more shallow.
         Default is `0`.
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
    passive_collate_length : int, optional
        Maximum interval, in ping indices, between detected passive regions
        which will removed to merge consecutive passive regions together
        into a single, collated, region. Default is 10.
    passive_collate_length : int, optional
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
                top and bottom lines.
            `"original"`:
                Target patches for training were determined
                using original lines, before expanding the
                top and bottom lines.
            `"ntob"`:
                Target patches for training were determined
                using the original bottom line and the merged
                top line.
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
    crop_depth_min : float or None, optional
        Minimum depth to include in input. If `None` (default), there is no
        minimum depth.
    crop_depth_max : float or None, optional
        Maxmimum depth to include in input. If `None` (default), there is no
        maximum depth.
    image_height : int or None, optional
        Height in pixels of input to model. The data loaded from the csv will
        be resized to this height (the width of the image is unchanged).
        If `None` (default), the height matches that used when the model was
        trained.
    checkpoint : str or None, optional
        A path to a checkpoint file, or name of a checkpoint known to this
        package (listed in `CHECKPOINT_RESOURCES`). If `None` (default),
        the first checkpoint in `CHECKPOINT_RESOURCES` is used.
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
        Verbosity level. Default is `1`. Set to `0` to disable print
        statements, or elevate to a higher number to increase verbosity.
    """

    if device is None:
        device = "cuda" if cuda_is_really_available() else "cpu"
    device = torch.device(device)

    if checkpoint is None:
        # Use the first item from the list of checkpoints
        checkpoint = DEFAULT_CHECKPOINT

    ckpt_name = checkpoint

    if os.path.isfile(ckpt_name):
        ckpt_path = ckpt_name
    elif ckpt_name in CHECKPOINT_RESOURCES:
        ckpt_path = download_checkpoint(ckpt_name, cache_dir=cache_dir)
    else:
        raise ValueError(
            "The checkpoint parameter should either be a path to a file or "
            "one of \n{},\nbut {} was provided.".format(
                list(CHECKPOINT_RESOURCES.keys()), ckpt_name
            )
        )

    if not os.path.isfile(ckpt_path):
        raise EnvironmentError("No checkpoint found at '{}'".format(ckpt_path))
    if verbose >= 1:
        print("Loading checkpoint '{}'".format(ckpt_path))

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
            print(
                "Error: Unable to load checkpoint {}".format(os.path.abspath(ckpt_path))
            )
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

    if verbose >= 2:
        print("Constructing U-Net model, with arguments:")
        pprint.pprint(checkpoint["model_parameters"])
    unet = UNet(**checkpoint["model_parameters"])
    model = Echofilter(
        unet,
        mapping=checkpoint.get("wrapper_mapping", None),
        **checkpoint.get("wrapper_params", {})
    )
    if verbose >= 1:
        print(
            "Built model with {} trainable parameters".format(
                count_parameters(model, only_trainable=True)
            )
        )
    try:
        unet.load_state_dict(checkpoint["state_dict"])
        if verbose >= 1:
            print(
                "Loaded UNet state from checkpoint".format(
                    ckpt_path, checkpoint["epoch"]
                )
            )
    except RuntimeError as err:
        if verbose >= 2:
            print(
                "Warning: Checkpoint doesn't seem to be for the UNet."
                "Trying to load it as the whole model instead."
            )
        try:
            model.load_state_dict(checkpoint["state_dict"])
            if verbose >= 1:
                print(
                    "Loaded model state from checkpoint".format(
                        ckpt_path, checkpoint["epoch"]
                    )
                )
        except RuntimeError:
            print(
                "Could not load the checkpoint state as either the whole model"
                "or the unet component."
            )
            raise err

    # Ensure model is on correct device
    model.to(device)
    # Put model in evaluation mode
    model.eval()

    files_input = paths
    files = list(echofilter.path.parse_files_in_folders(paths, source_dir, extensions))
    if verbose >= 1:
        print("Processing {} file{}".format(len(files), "" if len(files) == 1 else "s"))

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
        maybe_tqdm = lambda x: tqdm(x, desc="Files")

    skip_count = 0
    incompatible_count = 0

    # Open EchoView connection
    with echofilter.win.maybe_open_echoview(
        do_open=do_open, minimize=minimize_echoview, hide=hide_echoview,
    ) as ev_app:
        for fname in maybe_tqdm(files):
            if verbose >= 2:
                print("Processing {}".format(fname))

            # Check what the full path should be
            fname_full = echofilter.path.determine_file_path(fname, source_dir)

            # Determine where destination should be placed
            destination = echofilter.path.determine_destination(
                fname, fname_full, source_dir, output_dir
            )
            if not keep_ext:
                destination = os.path.splitext(destination)[0]

            # Check whether to skip processing this file
            if skip_existing:
                any_missing = False
                dest_files = []
                for name in ("top", "bottom", "surface"):
                    dest_files.append("{}.{}.evl".format(destination, name))
                dest_files.append("{}.{}.evr".format(destination, "regions"))
                for dest_file in dest_files:
                    if not os.path.isfile(dest_file):
                        any_missing = True
                        break
                if not any_missing:
                    if verbose >= 2:
                        print("  Skipping {}".format(fname))
                    skip_count += 1
                    continue

            # Determine whether we need to run ev2csv on this file
            ext = os.path.splitext(fname)[1]
            if len(ext) > 0:
                ext = ext[1:].lower()
            if ext == "csv":
                export_to_csv = False
            elif ext == "ev":
                export_to_csv = True
            elif len(extensions) == 1 and "csv" in extensions:
                export_to_csv = False
            elif len(extensions) == 1 and "ev" in extensions:
                export_to_csv = True
            else:
                error_str = "Unsure how to process file {} with unrecognised extension {}".format(
                    fname, ext
                )
                if not skip_incompatible:
                    raise EnvironmentError(error_str)
                if verbose >= 2:
                    print("  Skipping incompatible file {}".format(fname))
                incompatible_count += 1
                continue

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
                    csv_fname += csv_suffix

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

                if dry_run:
                    if verbose >= 1:
                        print("  Would write files to {}.SUFFIX".format(destination))
                    continue

                # Load the data
                if verbose >= 5:
                    warn_row_overflow = np.inf
                elif verbose >= 4:
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
                            print("  Skipping incompatible file {}".format(fname))
                        incompatible_count += 1
                        continue
                    print("CSV file {} could not be loaded.".format(fname))
                    raise

            output = inference_transect(
                model,
                timestamps,
                depths,
                signals,
                device,
                image_height,
                facing=facing,
                crop_depth_min=crop_depth_min,
                crop_depth_max=crop_depth_max,
                data_center=center_param,
                data_deviation=deviation_param,
                nan_value=nan_value,
                verbose=verbose - 1,
            )

            # Convert output into lines
            surface_depths = output["depths"][
                echofilter.utils.last_nonzero(output["p_is_above_surface"] > 0.5, -1)
            ]
            top_depths = output["depths"][
                echofilter.utils.last_nonzero(output["p_is_above_top"] > 0.5, -1)
            ]
            bottom_depths = output["depths"][
                echofilter.utils.first_nonzero(output["p_is_below_bottom"] > 0.5, -1)
            ]
            # Offset lines
            top_depths += offset_top
            bottom_depths -= offset_bottom
            # Redact passive regions
            is_passive = output["p_is_passive"] < 0.5
            if lines_during_passive == "predict":
                pass
            elif lines_during_passive == "redact":
                top_depths = top_depths[~is_passive]
                bottom_depths = bottom_depths[~is_passive]
            elif lines_during_passive == "undefined":
                top_depths[is_passive] = EV_UNDEFINED_DEPTH
                bottom_depths[is_passive] = EV_UNDEFINED_DEPTH
            elif lines_during_passive.startswith("interp"):
                if lines_during_passive == "interpolate-time":
                    x = output["timestamps"]
                elif lines_during_passive == "interpolate-index":
                    x = np.arange(len(output["timestamps"]))
                else:
                    raise ValueError(
                        "Unsupported passive line interpolation method: {}".format(
                            lines_during_passive
                        )
                    )
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
                    top_depths[is_passive] = np.interp(
                        x[is_passive], x[~is_passive], top_depths[~is_passive]
                    )
                    bottom_depths[is_passive] = np.interp(
                        x[is_passive], x[~is_passive], bottom_depths[~is_passive]
                    )
            else:
                raise ValueError(
                    "Unsupported passive line method: {}".format(lines_during_passive)
                )

            # Export evl files
            destination_dir = os.path.dirname(destination)
            if destination_dir != "":
                os.makedirs(destination_dir, exist_ok=True)
            for name, depths in (
                ("top", top_depths),
                ("bottom", bottom_depths),
                ("surface", surface_depths),
            ):
                dest_file = "{}.{}.evl".format(destination, name)
                if verbose >= 2:
                    print("Writing output {}".format(dest_file))
                if os.path.exists(dest_file) and not overwrite_existing:
                    raise EnvironmentError(
                        "Output {} already exists.\n"
                        " Run with overwrite_existing=True (with the command line"
                        " interface, use the --force flag) to overwrite existing"
                        " outputs.".format(dest_file)
                    )
                echofilter.raw.loader.evl_writer(
                    dest_file, timestamps, depths, status=line_status
                )
            # Export evr file
            dest_file = "{}.{}.evr".format(destination, "regions")
            if verbose >= 2:
                print("Writing output {}".format(dest_file))
            if os.path.exists(dest_file) and not overwrite_existing:
                raise EnvironmentError(
                    "Output {} already exists.\n"
                    " Run with overwrite_existing=True (with the command line"
                    " interface, use the --force flag) to overwrite existing"
                    " outputs.".format(dest_file)
                )

            patches_key = "p_is_patch"
            if patch_mode is None:
                if output["is_upward_facing"]:
                    patches_key += "-ntob"
            elif patch_mode != "merged":
                patches_key += "-" + patch_mode

            echofilter.raw.loader.write_transect_regions(
                dest_file,
                output,
                patches_key=patches_key,
                passive_collate_length=passive_collate_length,
                removed_collate_length=removed_collate_length,
                minimum_passive_length=minimum_passive_length,
                minimum_removed_length=minimum_removed_length,
                minimum_patch_area=minimum_patch_area,
                common_notes=common_notes,
                verbose=verbose - 1,
            )

    if verbose >= 1:
        s = "Finished {}processing {} file{}.".format(
            "simulating " if dry_run else "",
            len(files),
            "" if len(files) == 1 else "s",
        )
        skip_total = skip_count + incompatible_count
        if skip_total > 0:
            s += " Of these, {} file{} skipped: {} already processed".format(
                skip_total, " was" if skip_total == 1 else "s were", skip_count,
            )
            if not dry_run:
                s += ", {} incompatible.".format(incompatible_count)
            s += "."
        print(s)


def inference_transect(
    model,
    timestamps,
    depths,
    signals,
    device,
    image_height,
    facing="auto",
    crop_depth_min=None,
    crop_depth_max=None,
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
    crop_depth_min : float or None, optional
        Minimum depth to include in input. If `None` (default), there is no
        minimum depth.
    crop_depth_max : float or None, optional
        Maxmimum depth to include in input. If `None` (default), there is no
        maximum depth.
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
        Level of verbosity. Default is `1`.

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
    if crop_depth_min is not None:
        # Apply minimum depth crop
        depth_crop_mask = transect["depths"] >= crop_depth_min
        transect["depths"] = transect["depths"][depth_crop_mask]
        transect["signals"] = transect["signals"][:, depth_crop_mask]
    if crop_depth_max is not None:
        # Apply maximum depth crop
        depth_crop_mask = transect["depths"] <= crop_depth_max
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
        if facing == "auto" and verbose >= 1:
            print(
                "Data was autodetected as upward facing, and was flipped"
                " vertically before being input into the model."
            )
        is_upward_facing = True
    elif facing[:4] != "down" and facing[:4] != "auto":
        raise ValueError('facing should be one of "downward", "upward", and "auto"')
    elif facing[:4] == "down" and is_upward_facing:
        print(
            'Warning: facing = "{}" was provided, but data appears to be upward facing'.format(
                facing
            )
        )
        is_upward_facing = False

    # To reduce memory consumption, split into segments whenever the recording
    # interval is longer than normal
    segments = split_transect(**transect)
    if verbose >= 1:
        segments = tqdm(list(segments), desc="Segments")
    outputs = []
    for segment in segments:
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

    if verbose >= 1:
        print()

    joined_output = join_transect(outputs)
    joined_output["is_upward_facing"] = is_upward_facing
    return joined_output


def get_default_cache_dir():
    """Determine the default cache directory."""
    return appdirs.user_cache_dir("echofilter", "DeepSense")


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
            if verbose > 0:
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
                if verbose > 0:
                    print(
                        "\nCould not download checkpoint {} from GDrive!".format(
                            checkpoint_name
                        )
                    )
        else:
            if verbose > 0:
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
                if verbose > 0:
                    print(
                        "\nCould not download checkpoint {} from {}".format(
                            checkpoint_name, url_or_id
                        )
                    )

    if not success:
        raise OSError("Unable to download {} from {}".format(checkpoint_name, sources))

    if verbose > 0:
        print("Downloaded checkpoint to {}".format(destination))

    return destination


def main():
    import argparse

    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py":
        prog = "echofilter"
    parser = argparse.ArgumentParser(
        prog=prog,
        description=echofilter.__meta__.description,
        formatter_class=echofilter.utils.FlexibleHelpFormatter,
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
            --source-dir. For each directory given, the directory
            will be searched recursively for files bearing an
            extension specified by SEARCH_EXTENSION (see the
            --extension argument for details).
            Multiple files and directories can be specified,
            separated by spaces.
            This is a required argument. At least one input file
            or directory must be given.
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
            specified by the paths argument. Default: "." (the current
            directory).
        """,
    )
    default_extensions = ["csv"]
    if echofilter.path.check_if_windows():
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
            Default: {}.
            (Note that the default SEARCH_EXTENSION value is
            OS-specific.)
        """.format(
            default_extensions
        ),
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
    group_infile.add_argument(
        "--skip-incompatible",
        dest="skip_incompatible",
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
        "--force",
        "-f",
        dest="overwrite_existing",
        action="store_true",
        help="""
            Overwrite existing files without warning. Default behaviour is to
            stop processing if an output file already exists.
        """,
    )
    DEFAULT_CACHE_DIR = get_default_cache_dir()
    group_outfile.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="""d|
            Path to checkpoint cache directory.
            Default: "{}".
        """.format(
            DEFAULT_CACHE_DIR
        ),
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
        "--csv-suffix",
        type=str,
        default=".csv",
        help="""
            Suffix used for cached CSV files which are exported from EV files.
            This should contain a file extension, including ".", but may
            optionally contain additional text, for instance "_Sv_raw.csv".
            Default: ".csv".
        """,
    )
    group_outfile.add_argument(
        "--keep-ext",
        action="store_true",
        help="""
            If provided, the output file names maintain the input file
            extension before their suffix (including a new file extension).
            Default behaviour is to strip the input file name extension before
            constructing the output path.
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
            Status value for all the lines which are generated. Options are:
              0: none
              1: unverified
              2: bad
              3: good
            Default: 3.
        """,
    )
    group_outconfig.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="""
            Offset for both top and bottom lines, in metres. This will shift
            both lines towards each other by the same distance of OFFSET.
            Default is 0.
        """,
    )
    group_outconfig.add_argument(
        "--offset-top",
        type=int,
        default=None,
        help="""
            Offset for the top line, in metres. This shifts the top line
            downards by some distance OFFSET_TOP. If this is set, it overwrites
            the value provided by --offset.
        """,
    )
    group_outconfig.add_argument(
        "--offset-bottom",
        type=int,
        default=None,
        help="""
            Offset for the bottom line, in metres. This shifts the bottom line
            upwards by some distance OFFSET_BOTTOM. If this is set, it
            overwrites the value provided by --offset.
        """,
    )
    group_outconfig.add_argument(
        "--lines-during-passive",
        type=str,
        default="redact",
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
            Default: "redact".
        """.format(
            EV_UNDEFINED_DEPTH
        ),
    )
    group_outconfig.add_argument(
        "--passive-collate",
        "--passive-collate-length",
        dest="passive_collate_length",
        type=int,
        default=10,
        help="""
            Maximum interval, in ping indices, between detected passive regions
            which will removed to merge consecutive passive regions together
            into a single, collated, region. Default is 10.
        """,
    )
    group_outconfig.add_argument(
        "--removed-collate",
        "--removed-collate-length",
        dest="passive_collate_length",
        type=int,
        default=10,
        help="""
            Maximum interval, in ping indices, between detected blocks
            (vertical rectangles) marked for removal which will also be removed
            to merge consecutive removed blocks together into a single,
            collated, region. Default is 10.
        """,
    )
    group_outconfig.add_argument(
        "--minimum-passive",
        "--minimum-passive-length",
        dest="minimum_passive_length",
        type=int,
        default=10,
        help="""
            Minimum length, in ping indices, which a detected passive region
            must have to be included in the output. Set to -1 to omit all
            detected passive regions from the output. Default is 10.
        """,
    )
    group_outconfig.add_argument(
        "--minimum-removed",
        "--minimum-removed-length",
        dest="minimum_removed_length",
        type=int,
        default=10,
        help="""
            Minimum length, in ping indices, which a detected removal block
            (vertical rectangle) must have to be included in the output.
            Set to -1 to omit all detected removal blocks from the output.
            Default is 10.
        """,
    )
    group_outconfig.add_argument(
        "--minimum-patch",
        "--minimum-patch-area",
        dest="minimum_patch_area",
        type=int,
        default=25,
        help="""
            Minimum area, in pixels, which a detected removal patch
            (contour/polygon) region must have to be included in the output.
            Set to -1 to omit all detected patches from the output.
            Default is 25.
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
                  top and bottom lines.
              original:
                  Target patches for training were determined
                  using original lines, before expanding the
                  top and bottom lines.
              ntob:
                  Target patches for training were determined
                  using the original bottom line and the merged
                  top line.
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
            Default: "{}".
        """.format(
            DEFAULT_VARNAME
        ),
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
            Default: "mode", the modal number of depths is used, and the modal
            depth range is select amongst time samples which bear this number
            of depths.
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
        "--crop-depth-min",
        type=float,
        default=None,
        help="""
            Shallowest depth, in metres, to analyse. Data will be truncated at
            this depth, with shallower data removed before the Sv input is
            shown to the model. Default behaviour is not to truncate.
        """,
    )
    group_inproc.add_argument(
        "--crop-depth-max",
        type=float,
        default=None,
        help="""
            Deepest depth, in metres, to analyse. Data will be truncated at
            this depth, with deeper data removed before the Sv input is
            shown to the model. Default behaviour is not to truncate.
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
            Default: "{}".
        """.format(
            DEFAULT_CHECKPOINT
        ),
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
        default=1,
        help="""
            Increase the level of verbosity of the program. This can be
            specified multiple times, each will increase the amount of detail
            printed to the terminal. The default verbosity level is 1.
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

    kwargs["verbose"] -= kwargs.pop("quiet", 0)

    default_offset = kwargs.pop("offset")
    if kwargs["offset_top"] is None:
        kwargs["offset_top"] = default_offset
    if kwargs["offset_bottom"] is None:
        kwargs["offset_bottom"] = default_offset

    if kwargs["hide_echoview"] is None:
        kwargs["hide_echoview"] = "never" if kwargs["minimize_echoview"] else "new"

    run_inference(**kwargs)


if __name__ == "__main__":
    main()
