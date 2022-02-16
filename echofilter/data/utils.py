"""
Utility functions for dataset.
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

import random

import numpy as np
import torch

from ..utils import first_nonzero, last_nonzero


def worker_seed_fn(worker_id):
    """
    A worker initialization function for `torch.utils.data.DataLoader` objects
    which seeds builtin `random` and `numpy` with :meth:`torch.randint` (which is
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
    A worker initialization function for :class:`torch.utils.data.DataLoader`
    objects which produces the same seed for builtin `random`, `numpy`, and
    `torch` every time, so it is the same for every epoch.

    Parameters
    ----------
    worker_id : int
        The ID of the worker.
    """
    random.seed(worker_id)
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)
