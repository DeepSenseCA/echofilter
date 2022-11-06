"""
Interacting with the list of available checkpoints.
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
from collections import OrderedDict
import os
import pickle

import appdirs
import yaml

from . import style


PACKAGE_DIR = os.path.dirname(os.path.dirname(__file__))
REPO_DIR = os.path.dirname(PACKAGE_DIR)
CHECKPOINT_FILE = os.path.join(PACKAGE_DIR, "checkpoints.yaml")
CHECKPOINT_FILE_ALT = os.path.join(REPO_DIR, "checkpoints.yaml")
CHECKPOINT_EXT = ".pt"


def get_checkpoint_list():
    """
    List the currently available checkpoints, as stored in a local file.

    Returns
    -------
    checkpoints : OrderedDict
        Dictionary with a key for each checkpoint. Each key maps to a dictionary
        whose elements describe the checkpoint.
    """
    checkpoint_file_use = None
    if os.path.isfile(CHECKPOINT_FILE):
        checkpoint_file_use = CHECKPOINT_FILE
    elif os.path.isfile(CHECKPOINT_FILE_ALT):
        checkpoint_file_use = CHECKPOINT_FILE_ALT
    else:
        raise EnvironmentError(f"No such file: '{CHECKPOINT_FILE}'")
    with open(checkpoint_file_use, "r") as hf:
        checkpoints = OrderedDict(yaml.safe_load(hf)["checkpoints"])
    return checkpoints


def get_default_checkpoint():
    """
    Get the name of the current default checkpoint.

    Returns
    -------
    checkpoint_name : str
        Name of current checkpoint.
    """
    return next(iter(get_checkpoint_list()))


def cannonise_checkpoint_name(name):
    """
    Cannonises checkpoint name by removing extension.

    Parameters
    ----------
    name : str
        Name of checkpoint, possibly including extension.

    Returns
    -------
    name : str
        Name of checkpoint, with extension removed it matches a possible
        checkpoint file extension.
    """
    for possible_ext in [
        ".ckpt.pth.tar",
        ".checkpoint.pth.tar",
        ".pth.tar",
        ".ckpt.tar",
        ".checkpoint.tar",
        CHECKPOINT_EXT,
        ".pt",
        ".pth",
        ".ckpt",
        ".tar",
    ]:
        if name.lower().endswith(possible_ext):
            name = name[: -len(possible_ext)]
            return name
    return name


class ListCheckpoints(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        print("Currently available model checkpoints:")
        for checkpoint, props in get_checkpoint_list().items():
            if checkpoint == get_default_checkpoint():
                print("  * " + style.progress_fmt(checkpoint))
            else:
                print("    " + checkpoint)
        parser.exit()  # exits the program with no more arg parsing and checking


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

    checkpoint_name = cannonise_checkpoint_name(checkpoint_name)
    destination = os.path.join(cache_dir, checkpoint_name + CHECKPOINT_EXT)

    if os.path.exists(destination):
        return destination

    # Import packages needed for downloading files
    import requests, urllib
    from torchvision.datasets.utils import download_url, download_file_from_google_drive

    os.makedirs(cache_dir, exist_ok=True)

    sources = get_checkpoint_list()[checkpoint_name]
    success = False
    for key, url_or_id in sources.items():
        if key == "gdrive":
            if verbose >= 1:
                print(
                    "Downloading checkpoint {} from GDrive...".format(checkpoint_name)
                )
            try:
                download_file_from_google_drive(
                    url_or_id,
                    os.path.dirname(destination),
                    filename=os.path.basename(destination),
                )
                success = True
                continue
            except pickle.UnpicklingError:
                if verbose >= 1:
                    print(
                        style.error_fmt(
                            "\nCould not download checkpoint {} from GDrive!".format(
                                checkpoint_name
                            )
                        )
                    )
            except (requests.exceptions.ConnectionError, urllib.error.URLError):
                msg = "Could not connect to Google Drive. Please check your Internet connection."
                with style.error_message(msg) as msg:
                    raise EnvironmentError(msg)
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
            except pickle.UnpicklingError:
                if verbose >= 1:
                    print(
                        style.error_fmt(
                            "\nCould not download checkpoint {} from {}".format(
                                checkpoint_name, url_or_id
                            )
                        )
                    )
            except (requests.exceptions.ConnectionError, urllib.error.URLError):
                msg = "Could not connect to file server to download {}. Please check your Internet connection.".format(
                    url_or_id
                )
                with style.error_message(msg) as msg:
                    raise EnvironmentError(msg)

    if not success:
        msg = "Unable to download {} from {}".format(checkpoint_name, sources)
        with style.error_message(msg) as msg:
            raise OSError(msg)

    if verbose >= 1:
        print("Downloaded checkpoint to {}".format(destination))

    return destination


def load_checkpoint(
    ckpt_name=None, cache_dir=None, device="cpu", return_name=False, verbose=1
):
    """
    Load a checkpoint, either from absolute path or the cache.

    Parameters
    ----------
    checkpoint_name : str or None, optional
        Path to checkpoint file, or name of checkpoint to download.
        Default is `None`.
    cache_dir : str or None, optional
        Path to local cache directory. If `None` (default), an OS-appropriate
        application-specific default cache directory is used.
    device : str or torch.device or None, optional
        Device onto which weight tensors will be mapped. If `None`, no mapping
        is performed and tensors will be loaded onto the same device as they
        were on when saved (which will result in an error if the device is not
        present). Default is `"cpu"`.
    return_name : bool, optional
        If `True`, a tuple is returned indicting the name of the checkpoint
        which was loaded. This is useful if the default checkpoint was loaded.
        Default is `False`.
    verbose : int, optional
        Verbosity level. Default is `1`. Set to `0` to disable print
        statements.

    Returns
    -------
    checkpoint : dict
        Loaded checkpoint.
    checkpoint_name : str, optional
        If `return_name` is `True`, the name of the checkpoint is also
        returned.
    """
    import torch

    if ckpt_name is None:
        ckpt_name = get_default_checkpoint()

    if cache_dir is None:
        cache_dir = get_default_cache_dir()

    ckpt_name_cannon = cannonise_checkpoint_name(ckpt_name)
    checkpoint_resources = get_checkpoint_list()
    builtin_ckpt_path_a = os.path.join(
        PACKAGE_DIR,
        "checkpoints",
        os.path.split(ckpt_name)[1],
    )
    builtin_ckpt_path_b = os.path.join(
        PACKAGE_DIR,
        "checkpoints",
        ckpt_name_cannon + CHECKPOINT_EXT,
    )

    using_cache = False
    if os.path.isfile(ckpt_name):
        ckpt_path = ckpt_name
        ckpt_dscr = "local"
    elif os.path.isfile(ckpt_name + CHECKPOINT_EXT):
        ckpt_path = ckpt_name + CHECKPOINT_EXT
        ckpt_dscr = "local"
    elif os.path.isfile(builtin_ckpt_path_a):
        ckpt_path = builtin_ckpt_path_a
        ckpt_dscr = "builtin"
    elif os.path.isfile(builtin_ckpt_path_b):
        ckpt_path = builtin_ckpt_path_b
        ckpt_dscr = "builtin"
    elif ckpt_name_cannon in checkpoint_resources:
        using_cache = True
        ckpt_path = download_checkpoint(ckpt_name_cannon, cache_dir=cache_dir)
        ckpt_dscr = "cached"
    else:
        msg = style.error_fmt(
            "The checkpoint parameter should either be a path to a file or one of"
        )
        msg += "\n  ".join([""] + list(checkpoint_resources.keys()))
        msg += style.error_fmt("\nbut {} was provided.".format(ckpt_name))
        with style.error_message():
            raise ValueError(msg)

    if not os.path.isfile(ckpt_path):
        msg = "No checkpoint found at '{}'".format(ckpt_path)
        with style.error_message(msg) as msg:
            raise EnvironmentError(msg)
    if verbose >= 1:
        print("Loading model from {} checkpoint:\n  '{}'".format(ckpt_dscr, ckpt_path))

    load_args = {}
    if device is not None:
        # Map model to be loaded to specified single gpu.
        load_args = dict(map_location=device)
    try:
        checkpoint = torch.load(ckpt_path, **load_args)
    except pickle.UnpicklingError:
        if not using_cache:
            # Direct path to checkpoint was given, so we shouldn't delete
            # the user's file
            msg = "Error: Unable to load checkpoint {}".format(
                os.path.abspath(ckpt_path)
            )
            with style.error_message(msg) as msg:
                print(msg)
                raise
        else:
            # Delete the checkpoint and try again, in case it is just a
            # malformed download (interrupted download, etc)
            os.remove(ckpt_path)
            ckpt_path = download_checkpoint(ckpt_name, cache_dir=cache_dir)
            try:
                checkpoint = torch.load(ckpt_path, **load_args)

            except pickle.UnpicklingError:

                msg = "Error: Unable to load checkpoint {}.".format(
                    os.path.abspath(ckpt_path)
                )
                with style.error_message(msg) as msg:
                    print(msg)

                # Check if there was an error because the file was missing
                # and we downloaded a 404 error page instead.
                with open(ckpt_path) as myfile:
                    contents = myfile.read()
                    if "Not Found" in contents or "Error 404" in contents:
                        msg = (
                            "The file you are trying to download is not available"
                            " at this web address."
                            " The download produced a 404 Error (File Not Found)."
                            "\nThe original source file may have been moved or deleted"
                            " by its host."
                        )
                        with style.error_message(msg) as msg:
                            print(msg + "\n")
                            raise EnvironmentError(msg) from None

                # Check if the user ran out of storage space part-way through
                # the download.
                import shutil

                _, _, free_B = shutil.disk_usage("/")
                free_MiB = free_B // 10 ** 6

                if free_MiB < 64:
                    msg = (
                        "You only have {}MB of free space on your hard disk."
                        " Please free up 100MB of space on your hard disk and"
                        " then try again.\n"
                    ).format(free_MiB)
                    with style.error_message(msg) as msg:
                        print(msg + "\n")
                        raise EnvironmentError(msg) from None

                msg = "There was an unknown issue opening the downloaded file."
                with style.error_message(msg) as msg:
                    print(msg)
                    raise

    if return_name:
        return checkpoint, ckpt_name

    return checkpoint
