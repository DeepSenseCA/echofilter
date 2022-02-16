"""
Utility functions for interacting with optimizers.
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


def get_current_lr(optimizer):
    """
    Get the learning rate of an optimizer.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        An optimizer, with a learning rate common to all parameter groups.

    Returns
    -------
    float
        The learning rate of the first parameter group.
    """
    return optimizer.param_groups[0]["lr"]


def get_current_momentum(optimizer):
    """
    Get the momentum of an optimizer.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        An optimizer which implements momentum or betas (where momentum is the
        first beta, c.f. :class:`torch.optim.Adam`) with a momentum common to all
        parameter groups.

    Returns
    -------
    float
        The momentum of the first parameter group.
    """
    if "momentum" not in optimizer.defaults and "betas" not in optimizer.defaults:
        raise ValueError(
            "optimizer {} does not support momentum".format(optimizer.__class__)
        )

    group = optimizer.param_groups[0]
    if "momentum" in group:
        return group["momentum"]
    else:
        return group["betas"][0]
