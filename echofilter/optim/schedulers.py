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

import math
import warnings

from torch.optim.optimizer import Optimizer

from .torch_backports import OneCycleLR


class MesaOneCycleLR(OneCycleLR):
    r"""
    A variant on the 1cycle learning rate policy which features a flat
    region at maximum learning rate between warm-up and warm-down.

    Sets the learning rate of each parameter group according to the
    1cycle learning rate policy. The 1cycle policy anneals the learning
    rate from an initial learning rate to some maximum learning rate and then
    from that maximum learning rate to some minimum learning rate much lower
    than the initial learning rate.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.

    The 1cycle learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This scheduler is not chainable.

    Note also that the total number of steps in the cycle can be determined in one
    of two ways (listed in order of precedence):

    #. A value for total_steps is explicitly provided.
    #. A number of epochs (epochs) and a number of steps per epoch
       (steps_per_epoch) are provided.
       In this case, the number of total steps is inferred by
       total_steps = epochs * steps_per_epoch

    You must either provide a value for total_steps or provide a value for both
    epochs and steps_per_epoch.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group.
        total_steps (int): The total number of steps in the cycle. Note that
            if a value is provided here, then it must be inferred by providing
            a value for epochs and steps_per_epoch.
            Default: None
        epochs (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle
            if a value for total_steps is not provided.
            Default: None
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the
            cycle if a value for total_steps is not provided.
            Default: None
        pct_start (float): The percentage of the cycle (in number of steps) spent
            increasing the learning rate.
            Default: 0.25
        pct_end (float): The percentage of the cycle (in number of steps) spent
            before decreasing the learning rate.
            Default: 0.75
        anneal_strategy (str): {"cos", "linear"}
            Specifies the annealing strategy: "cos" for cosine annealing, "linear" for
            linear annealing.
            Default: "cos".
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between "base_momentum" and "max_momentum".
            Default: True
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            "base_momentum" and learning rate is "max_lr".
            Default: 0.85
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is "max_momentum"
            and learning rate is "base_lr"
            Default: 0.95
        div_factor (float): Determines the initial learning rate via
            initial_lr = max_lr/div_factor
            Default: 25
        final_div_factor (float): Determines the minimum learning rate via
            min_lr = initial_lr/final_div_factor
            Default: 1e4
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1

    Example:
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = MesaOneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(data_loader), epochs=10)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()


    .. _Super-Convergence\: Very Fast Training of Neural Networks Using Large Learning Rates:
        https://arxiv.org/abs/1708.07120
    """

    def __init__(
        self,
        optimizer,
        max_lr,
        total_steps=None,
        pct_start=0.25,
        pct_end=0.75,
        **kwargs,
    ):

        # Validate pct_start
        if not isinstance(pct_start, float) or pct_start < 0 or pct_start > 1:
            raise ValueError(
                "Expected pct_start to be a float between 0 and 1, but got {}".format(
                    pct_start
                )
            )

        # Validate pct_end
        if pct_end < pct_start or pct_end > 1 or not isinstance(pct_start, float):
            raise ValueError(
                "Expected pct_end to be a float between pct_start={} and 1, but got {}".format(
                    pct_start, pct_end
                )
            )

        super(MesaOneCycleLR, self).__init__(
            optimizer, max_lr, total_steps=total_steps, pct_start=pct_start, **kwargs
        )

        self.step_size_up = float(pct_start * self.total_steps) - 1
        self.step_size_down_start = float(pct_end * self.total_steps) - 1
        self.step_size_down = float(self.total_steps - self.step_size_down_start) - 1

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                DeprecationWarning,
            )

        lrs = []
        step_num = self.last_epoch

        if step_num > self.total_steps:
            raise ValueError(
                "Tried to step {} times. The specified number of total steps is {}".format(
                    step_num + 1, self.total_steps
                )
            )

        for group in self.optimizer.param_groups:
            if step_num <= self.step_size_up:
                computed_lr = self.anneal_func(
                    group["initial_lr"], group["max_lr"], step_num / self.step_size_up
                )
                if self.cycle_momentum:
                    computed_momentum = self.anneal_func(
                        group["max_momentum"],
                        group["base_momentum"],
                        step_num / self.step_size_up,
                    )
            elif step_num <= self.step_size_down_start:
                computed_lr = group["max_lr"]
                if self.cycle_momentum:
                    computed_momentum = group["base_momentum"]
            else:
                down_step_num = step_num - self.step_size_down_start
                computed_lr = self.anneal_func(
                    group["max_lr"],
                    group["min_lr"],
                    down_step_num / self.step_size_down,
                )
                if self.cycle_momentum:
                    computed_momentum = self.anneal_func(
                        group["base_momentum"],
                        group["max_momentum"],
                        down_step_num / self.step_size_down,
                    )

            lrs.append(computed_lr)
            if self.cycle_momentum:
                if self.use_beta1:
                    _, beta2 = group["betas"]
                    group["betas"] = (computed_momentum, beta2)
                else:
                    group["momentum"] = computed_momentum

        return lrs
