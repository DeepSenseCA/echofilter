"""
Interacting with the list of available checkpoints.
"""

import argparse
from collections import OrderedDict
import os
import yaml


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
            name = name[:-len(possible_ext)]
            return name
    return name


class ListCheckpoints(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        print("Currently available model checkpoints:")
        CHECKPOINT_RESOURCES = get_checkpoint_list()
        DEFAULT_CHECKPOINT = get_default_checkpoint()
        for checkpoint, props in CHECKPOINT_RESOURCES.items():
            print(
                "  {} {}".format(
                    "*" if checkpoint == DEFAULT_CHECKPOINT else " ", checkpoint
                )
            )
        parser.exit()  # exits the program with no more arg parsing and checking
