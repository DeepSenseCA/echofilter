"""
Model wrapper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from .utils import TensorDict, logavgexp


class Echofilter(nn.Module):
    """
    Echofilter logit mapping wrapper.

    Parameters
    ----------
    model : `torch.nn.Module`
        The model backbone, which converts inputs to logits.
    top : str, optional
        Type of output for top line and surface line. If `"mask"`, the top
        output corresponds to logits, which are converted into probabilities
        with sigmoid. If `"boundary"` (default), the output corresponds to
        logits for the location of the line, which is converted into a
        probability mask using softmax and cumsum.
    bottom : str, optional
        As for `top`, but for the bottom line. Default is `"boundary"`.
    mapping : dict or None, optional
        Mapping from logit names to output channels provided by `model`.
        If `None`, a default mapping is used. The mapping is stored as
        `self.mapping`.
    reduction_ispassive : str, optional
        Method used to reduce the depths dimension for the `"logit_is_passive"`
        output. Default is `"mean"`.
    reduction_isremoved : str, optional
        Method used to reduce the depths dimension for the `"logit_is_removed"`
        output. Default is `"mean"`.
    """

    def __init__(
        self,
        model,
        top="boundary",
        bottom="boundary",
        mapping=None,
        reduction_ispassive="logavgexp",
        reduction_isremoved="logavgexp",
    ):
        super(Echofilter, self).__init__()
        self.model = model
        self.params = {
            "top": top,
            "bottom": bottom,
            "reduction_ispassive": reduction_ispassive,
            "reduction_isremoved": reduction_isremoved,
        }
        if mapping is None:
            mapping = {
                "logit_is_above_top": 0,
                "logit_is_boundary_top": 0,
                "logit_is_below_bottom": 1,
                "logit_is_boundary_bottom": 1,
                "logit_is_removed": 2,
                "logit_is_passive": 3,
                "logit_is_patch": 4,
                "logit_is_above_surface": 5,
                "logit_is_boundary_surface": 5,
                "logit_is_above_top-original": 6,
                "logit_is_boundary_top-original": 6,
                "logit_is_below_bottom-original": 7,
                "logit_is_boundary_bottom-original": 7,
                "logit_is_patch-original": 8,
                "logit_is_patch-ntob": 9,
            }
            if top == "boundary":
                mapping.pop("logit_is_above_top")
                mapping.pop("logit_is_above_top-original")
                mapping.pop("logit_is_above_surface")
            else:
                mapping.pop("logit_is_boundary_top")
                mapping.pop("logit_is_boundary_top-original")
                mapping.pop("logit_is_boundary_surface")
            if bottom == "boundary":
                mapping.pop("logit_is_below_bottom")
                mapping.pop("logit_is_below_bottom-original")
            else:
                mapping.pop("logit_is_boundary_bottom")
                mapping.pop("logit_is_boundary_bottom-original")
        self.mapping = mapping

    def forward(self, x):
        logits = self.model(x)
        outputs = TensorDict()

        # Include raw logits in output
        for key, index in self.mapping.items():
            outputs[key] = logits[:, index]

        # Flatten some outputs which are vectors not arrays
        if self.params["reduction_isremoved"] == "mean":
            outputs["logit_is_removed"] = torch.mean(
                outputs["logit_is_removed"], dim=-1
            )
        elif self.params["reduction_isremoved"] in {"logavgexp", "lae"}:
            outputs["logit_is_removed"] = logavgexp(outputs["logit_is_removed"], dim=-1)
        else:
            raise ValueError(
                "Unsupported reduction_isremoved value: {}".format(
                    self.params["reduction_isremoved"]
                )
            )

        if self.params["reduction_ispassive"] == "mean":
            outputs["logit_is_passive"] = torch.mean(
                outputs["logit_is_passive"], dim=-1
            )
        elif self.params["reduction_ispassive"] in {"logavgexp", "lae"}:
            outputs["logit_is_passive"] = logavgexp(outputs["logit_is_passive"], dim=-1)
        else:
            raise ValueError(
                "Unsupported reduction_ispassive value: {}".format(
                    self.params["reduction_ispassive"]
                )
            )

        # Convert logits to probabilities
        outputs["p_is_removed"] = torch.sigmoid(outputs["logit_is_removed"])
        outputs["p_is_passive"] = torch.sigmoid(outputs["logit_is_passive"])

        for sfx in ("top", "top-original", "surface"):
            if self.params["top"] == "mask":
                outputs["p_is_above_" + sfx] = torch.sigmoid(
                    outputs["logit_is_above_" + sfx]
                )
                outputs["p_is_below_" + sfx] = 1 - outputs["p_is_above_" + sfx]
            elif self.params["top"] == "boundary":
                outputs["p_is_boundary_" + sfx] = F.softmax(
                    outputs["logit_is_boundary_" + sfx], dim=-1
                )
                outputs["p_is_above_" + sfx] = torch.flip(
                    torch.cumsum(
                        torch.flip(outputs["p_is_boundary_" + sfx], dims=(-1,)), dim=-1
                    ),
                    dims=(-1,),
                )
                outputs["p_is_below_" + sfx] = torch.cumsum(
                    outputs["p_is_boundary_" + sfx], dim=-1
                )
                # Due to floating point precision, max value can exceed 1.
                # Fix this by clipping the values to the appropriate range.
                outputs["p_is_above_" + sfx].clamp_(0, 1)
                outputs["p_is_below_" + sfx].clamp_(0, 1)
            else:
                raise ValueError(
                    'Unsupported "top" parameter: {}'.format(self.params["top"])
                )

        for sfx in ("", "-original"):
            if self.params["bottom"] == "mask":
                outputs["p_is_below_bottom" + sfx] = torch.sigmoid(
                    outputs["logit_is_below_bottom" + sfx]
                )
                outputs["p_is_above_bottom" + sfx] = (
                    1 - outputs["p_is_below_bottom" + sfx]
                )
            elif self.params["bottom"] == "boundary":
                outputs["p_is_boundary_bottom" + sfx] = F.softmax(
                    outputs["logit_is_boundary_bottom" + sfx], dim=-1
                )
                outputs["p_is_below_bottom" + sfx] = torch.cumsum(
                    outputs["p_is_boundary_bottom" + sfx], dim=-1
                )
                outputs["p_is_above_bottom" + sfx] = torch.flip(
                    torch.cumsum(
                        torch.flip(outputs["p_is_boundary_bottom" + sfx], dims=(-1,)),
                        dim=-1,
                    ),
                    dims=(-1,),
                )
                # Due to floating point precision, max value can exceed 1.
                # Fix this by clipping the values to the appropriate range.
                outputs["p_is_below_bottom" + sfx].clamp_(0, 1)
                outputs["p_is_above_bottom" + sfx].clamp_(0, 1)
            else:
                raise ValueError(
                    'Unsupported "bottom" parameter: {}'.format(self.params["bottom"])
                )

        for sfx in ("", "-original", "-ntob"):
            outputs["p_is_patch" + sfx] = torch.sigmoid(outputs["logit_is_patch" + sfx])

        outputs["p_keep_pixel"] = (
            1.0
            * 0.5
            * ((1 - outputs["p_is_above_top"]) + outputs["p_is_below_top"])
            * 0.5
            * ((1 - outputs["p_is_below_bottom"]) + outputs["p_is_above_bottom"])
            * (1 - outputs["p_is_removed"].unsqueeze(-1))
            * (1 - outputs["p_is_passive"].unsqueeze(-1))
            * (1 - outputs["p_is_patch"])
        ).clamp_(0, 1)
        outputs["mask_keep_pixel"] = (
            1.0
            * (outputs["p_is_above_top"] < 0.5)
            * (outputs["p_is_below_bottom"] < 0.5)
            * (outputs["p_is_removed"].unsqueeze(-1) < 0.5)
            * (outputs["p_is_passive"].unsqueeze(-1) < 0.5)
            * (outputs["p_is_patch"] < 0.5)
        )
        return outputs


class EchofilterLoss(_Loss):
    """
    Evaluate loss for an Echofilter model.

    Parameters
    ----------
    reduction : `"mean"` or `"sum"`, optional
        The reduction method, which is used to collapse batch and timestamp
        dimensions. Default is `"mean"`.
    top_mask : float, optional
        Weighting for top line/mask loss term. Default is `1.0`.
    bottom_mask : float, optional
        Weighting for bottom line/mask loss term. Default is `1.0`.
    removed_segment : float, optional
        Weighting for `is_removed` loss term. Default is `1.0`.
    passive : float, optional
        Weighting for `is_passive` loss term. Default is `1.0`.
    patch : float, optional
        Weighting for `mask_patch` loss term. Default is `1.0`.
    overall : float, optional
        Weighting for overall mask loss term. Default is `0.0`.
    surface : float, optional
        Weighting for surface line/mask loss term. Default is `1.0`.
    auxiliary : float, optional
        Weighting for auxiliary loss terms `"top-original"`,
        `"bottom-original"`, `"mask_patches-original"`, and
        `"mask_patches-ntob"`. Default is `1.0`.
    ignore_lines_during_passive : bool, optional
        Whether targets for lines should be excluded from the loss during
        passive data collection. Default is `True`.
    ignore_lines_during_removed : bool, optional
        Whether targets for lines should be excluded from the loss during
        entirely removed sections. Default is `True`.
    """

    __constants__ = ["reduction"]

    def __init__(
        self,
        reduction="mean",
        top_mask=1.0,
        bottom_mask=1.0,
        removed_segment=1.0,
        passive=1.0,
        patch=1.0,
        overall=0.0,
        surface=1.0,
        auxiliary=1.0,
        ignore_lines_during_passive=True,
        ignore_lines_during_removed=True,
    ):
        super(EchofilterLoss, self).__init__(None, None, reduction)
        self.top_mask = top_mask
        self.bottom_mask = bottom_mask
        self.removed_segment = removed_segment
        self.passive = passive
        self.patch = patch
        self.overall = overall
        self.surface = surface
        self.auxiliary = auxiliary
        self.ignore_lines_during_passive = ignore_lines_during_passive
        self.ignore_lines_during_removed = ignore_lines_during_removed

    def forward(self, input, target):
        """
        Construct loss term.

        Parameters
        ----------
        input : dict
            Output from `echofilter.wrapper.Echofilter` layer.
        target : dict
            A transect, as provided by `TransectDataset`.
        """
        loss = 0

        target["is_passive"] = target["is_passive"].to(
            input["logit_is_passive"].device,
            input["logit_is_passive"].dtype,
            non_blocking=True,
        )
        target["is_removed"] = target["is_removed"].to(
            input["logit_is_removed"].device,
            input["logit_is_removed"].dtype,
            non_blocking=True,
        )
        apply_loss_inclusion = False
        inner_reduction = self.reduction
        loss_inclusion_mask = 1
        if self.ignore_lines_during_passive:
            apply_loss_inclusion = True
            inner_reduction = "none"
            loss_inclusion_mask *= 1 - target["is_passive"]
        if self.ignore_lines_during_removed:
            apply_loss_inclusion = True
            inner_reduction = "none"
            loss_inclusion_mask *= 1 - target["is_removed"]
        loss_inclusion_sum = torch.sum(loss_inclusion_mask)
        # Prevent division by zero
        loss_inclusion_sum = torch.max(
            loss_inclusion_sum, torch.ones_like(loss_inclusion_sum)
        )

        for sfx in ("top", "top-original", "surface"):
            if sfx == "surface":
                weight = self.surface
                target_key = "mask_surf"
                target_i_key = "index_surf"
            else:
                weight = self.top_mask
                target_key = "mask_" + sfx
                target_i_key = "index_" + sfx
                if sfx != "top":
                    weight *= self.auxiliary
            if not weight:
                continue
            elif "logit_is_above_" + sfx in input:
                loss_term = F.binary_cross_entropy_with_logits(
                    input["logit_is_above_" + sfx],
                    target[target_key].to(
                        input["logit_is_above_" + sfx].device,
                        input["logit_is_above_" + sfx].dtype,
                    ),
                    reduction=inner_reduction,
                )
                if apply_loss_inclusion:
                    loss_term = loss_term * (loss_inclusion_mask.unsqueeze(-1))
                    loss_term = torch.sum(loss_term)
                    if self.reduction == "mean":
                        loss_term = loss_term / loss_inclusion_sum
            elif "logit_is_boundary_" + sfx in input:
                # Load cross-entropy class target
                C = target[target_i_key].to(
                    device=input["logit_is_boundary_" + sfx].device
                )
                loss_term = F.cross_entropy(
                    input["logit_is_boundary_" + sfx].transpose(-2, -1),
                    C,
                    reduction=inner_reduction,
                )
                if apply_loss_inclusion:
                    loss_term = loss_term * loss_inclusion_mask
                    loss_term = torch.sum(loss_term)
                    if self.reduction == "mean":
                        loss_term = loss_term / loss_inclusion_sum
            else:
                loss_term = F.binary_cross_entropy(
                    input["p_is_above_" + sfx],
                    target[target_key],
                    reduction=inner_reduction,
                )
                if apply_loss_inclusion:
                    loss_term = loss_term * (loss_inclusion_mask.unsqueeze(-1))
                    loss_term = torch.sum(loss_term)
                    if self.reduction == "mean":
                        loss_term = loss_term / loss_inclusion_sum
            if torch.isnan(loss_term).any():
                print("Loss term {} is NaN".format(target_key))
            else:
                loss += weight * loss_term

        for sfx in ("", "-original"):
            weight = self.bottom_mask
            if sfx != "":
                weight *= self.auxiliary
            if not weight:
                continue
            elif "logit_is_below_bottom" + sfx in input:
                loss_term = F.binary_cross_entropy_with_logits(
                    input["logit_is_below_bottom" + sfx],
                    target["mask_bot" + sfx].to(
                        input["logit_is_below_bottom" + sfx].device,
                        input["logit_is_below_bottom" + sfx].dtype,
                    ),
                    reduction=inner_reduction,
                )
                if apply_loss_inclusion:
                    loss_term = loss_term * (loss_inclusion_mask.unsqueeze(-1))
                    loss_term = torch.sum(loss_term)
                    if self.reduction == "mean":
                        loss_term = loss_term / loss_inclusion_sum
            elif "logit_is_boundary_bottom" + sfx in input:
                # Load cross-entropy class target
                C = target["index_bot" + sfx].to(
                    device=input["logit_is_boundary_bottom" + sfx].device
                )
                loss_term = F.cross_entropy(
                    input["logit_is_boundary_bottom" + sfx].transpose(-2, -1),
                    C,
                    reduction=inner_reduction,
                )
                if apply_loss_inclusion:
                    loss_term = loss_term * loss_inclusion_mask
                    loss_term = torch.sum(loss_term)
                    if self.reduction == "mean":
                        loss_term = loss_term / loss_inclusion_sum
            else:
                loss_term = F.binary_cross_entropy(
                    input["p_is_below_bottom" + sfx],
                    target["mask_bot" + sfx].to(
                        input["p_is_below_bottom" + sfx].device,
                        input["p_is_below_bottom" + sfx].dtype,
                    ),
                    reduction=inner_reduction,
                )
                if apply_loss_inclusion:
                    loss_term = loss_term * (loss_inclusion_mask.unsqueeze(-1))
                    loss_term = torch.sum(loss_term)
                    if self.reduction == "mean":
                        loss_term = loss_term / loss_inclusion_sum
            if torch.isnan(loss_term).any():
                print("Loss term mask_bot{} is NaN".format(sfx))
            else:
                loss += weight * loss_term

        if self.removed_segment:
            loss_term = F.binary_cross_entropy_with_logits(
                input["logit_is_removed"],
                target["is_removed"].to(
                    input["logit_is_removed"].device, input["logit_is_removed"].dtype
                ),
                reduction=self.reduction,
            )
            if torch.isnan(loss_term).any():
                print("Loss term is_removed is NaN")
            else:
                loss += self.removed_segment * loss_term

        if self.passive:
            loss_term = self.passive * F.binary_cross_entropy_with_logits(
                input["logit_is_passive"],
                target["is_passive"].to(
                    input["logit_is_passive"].device, input["logit_is_passive"].dtype
                ),
                reduction=self.reduction,
            )
            if torch.isnan(loss_term).any():
                print("Loss term is_passive is NaN")
            else:
                loss += self.passive * loss_term

        for sfx in ("", "-original", "-ntob"):
            weight = self.patch
            if sfx != "":
                weight *= self.auxiliary
            if not weight:
                continue
            loss_term = F.binary_cross_entropy_with_logits(
                input["logit_is_patch" + sfx],
                target["mask_patches" + sfx].to(
                    input["logit_is_patch" + sfx].device,
                    input["logit_is_patch" + sfx].dtype,
                ),
                reduction=self.reduction,
            )
            if torch.isnan(loss_term).any():
                print("Loss term mask_patches{} is NaN".format(sfx))
            else:
                loss += weight * loss_term

        if self.overall:
            loss_term = self.overall * F.binary_cross_entropy(
                input["p_keep_pixel"],
                target["mask"].to(
                    input["p_keep_pixel"].device, input["p_keep_pixel"].dtype
                ),
                reduction=self.reduction,
            )
            if torch.isnan(loss_term).any():
                print("Loss term overall is NaN")
            else:
                loss += self.overall * loss_term

        return loss
