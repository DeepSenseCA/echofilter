"""
Interacting with the list of available checkpoints.
"""

import argparse
from collections import OrderedDict
import os
import pickle
import urllib

import appdirs
import requests
from torchvision.datasets.utils import download_url, download_file_from_google_drive
import yaml

from . import style


FILE = os.path.dirname(__file__)
CHECKPOINT_FILE = os.path.join(os.path.dirname(FILE), "checkpoints.yaml")
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
    with open(CHECKPOINT_FILE, "r") as hf:
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
    destination = os.path.join(cache_dir, checkpoint_name + CHECKPOINT_EXT,)

    if os.path.exists(destination):
        return destination

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
