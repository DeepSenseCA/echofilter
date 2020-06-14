"""
Utility functions for interacting with optimizers.
"""


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
        first beta, c.f. torch.optim.Adam) with a momentum common to all
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
