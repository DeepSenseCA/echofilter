"""
Utility functions for dataset.
"""

import random

import numpy as np
import torch

from ..utils import first_nonzero, last_nonzero


def worker_seed_fn(worker_id):
    """
    A worker initialization function for `torch.utils.data.DataLoader` objects
    which seeds builtin `random` and `numpy` with `torch.randint` (which is
    stable if torch is manually seeded in the main program).

    Parameters
    ----------
    worker_id : int
        The ID of the worker.
    """
    np.random.seed((torch.randint(0, 4294967296, (1,)).item() + worker_id) % 4294967296)
    random.seed(torch.randint(0, 4294967296, (1,)).item() + worker_id)


def worker_staticseed_fn(worker_id):
    """
    A worker initialization function for `torch.utils.data.DataLoader` objects
    which produces the same seed for builtin `random`, `numpy`, and `torch`
    every time, so it is the same for every epoch.

    Parameters
    ----------
    worker_id : int
        The ID of the worker.
    """
    random.seed(worker_id)
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)
