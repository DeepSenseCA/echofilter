"""
Model wrapper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from .utils import TensorDict


class Echofilter(nn.Module):
    def __init__(self, model, top="boundary", bottom="boundary", mapping=None):
        super(Echofilter, self).__init__()
        self.model = model
        self.params = {
            "top": top,
            "bottom": bottom,
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
        outputs["logit_is_removed"] = torch.mean(outputs["logit_is_removed"], dim=-1)
        outputs["logit_is_passive"] = torch.mean(outputs["logit_is_passive"], dim=-1)

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
        auxillary=1.0,
        ignore_lines_during_passive=False,
    ):
        super(EchofilterLoss, self).__init__(None, None, reduction)
        self.top_mask = top_mask
        self.bottom_mask = bottom_mask
        self.removed_segment = removed_segment
        self.passive = passive
        self.patch = patch
        self.overall = overall
        self.surface = surface
        self.auxillary = auxillary
        self.ignore_lines_during_passive = ignore_lines_during_passive

    def forward(self, input, target):
        loss = 0

        target["is_passive"] = target["is_passive"].to(
            input["logit_is_passive"].device,
            input["logit_is_passive"].dtype,
            non_blocking=True,
        )
        inner_reduction = "none" if self.ignore_lines_during_passive else self.reduction

        for sfx in ("top", "top-original", "surface"):
            if sfx == "surface":
                weight = self.surface
                target_key = "mask_surf"
            else:
                weight = self.top_mask
                target_key = "mask_" + sfx
                if sfx != "top":
                    weight *= self.auxillary
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
                if self.ignore_lines_during_passive:
                    loss_term = loss_term * (1 - target["is_passive"].unsqueeze(-1))
                    loss_term = torch.sum(loss_term)
                    if self.reduction == "mean":
                        loss_term = loss_term / torch.sum(1 - target["is_passive"])
                loss += weight * loss_term
            elif "logit_is_boundary_" + sfx in input:
                X = target[target_key]
                shp = list(X.shape)
                shp[-1] = 1
                X = torch.cat(
                    [
                        torch.ones(shp, dtype=X.dtype, device=X.device),
                        X,
                        torch.zeros(shp, dtype=X.dtype, device=X.device),
                    ],
                    dim=-1,
                )
                X = X.float()
                X = X.narrow(-1, 0, X.shape[-1] - 1) - X.narrow(-1, 1, X.shape[-1] - 1)
                C = torch.argmax(X, dim=-1)
                Cmax = torch.tensor(
                    [input["logit_is_boundary_" + sfx].shape[-1] - 1],
                    device=C.device,
                    dtype=C.dtype,
                )
                C = torch.min(C, Cmax)
                loss_term = F.cross_entropy(
                    input["logit_is_boundary_" + sfx].transpose(-2, -1),
                    C,
                    reduction=inner_reduction,
                )
                if self.ignore_lines_during_passive:
                    loss_term = loss_term * (1 - target["is_passive"])
                    loss_term = torch.sum(loss_term)
                    if self.reduction == "mean":
                        loss_term = loss_term / torch.sum(1 - target["is_passive"])
                loss += weight * loss_term
            else:
                loss_term = F.binary_cross_entropy(
                    input["p_is_above_" + sfx],
                    target[target_key],
                    reduction=inner_reduction,
                )
                if self.ignore_lines_during_passive:
                    loss_term = loss_term * (1 - target["is_passive"].unsqueeze(-1))
                    loss_term = torch.sum(loss_term)
                    if self.reduction == "mean":
                        loss_term = loss_term / torch.sum(1 - target["is_passive"])
                loss += weight * loss_term

        for sfx in ("", "-original"):
            weight = self.bottom_mask
            if sfx != "":
                weight *= self.auxillary
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
                if self.ignore_lines_during_passive:
                    loss_term = loss_term * (1 - target["is_passive"].unsqueeze(-1))
                    loss_term = torch.sum(loss_term)
                    if self.reduction == "mean":
                        loss_term = loss_term / torch.sum(1 - target["is_passive"])
                loss += weight * loss_term
            elif "logit_is_boundary_bottom" + sfx in input:
                X = target["mask_bot" + sfx]
                shp = list(X.shape)
                shp[-1] = 1
                X = torch.cat(
                    [
                        torch.zeros(shp, dtype=X.dtype, device=X.device),
                        X,
                        torch.ones(shp, dtype=X.dtype, device=X.device),
                    ],
                    dim=-1,
                )
                X = X.float()
                X = X.narrow(-1, 0, X.shape[-1] - 1) - X.narrow(-1, 1, X.shape[-1] - 1)
                C = torch.argmin(X, dim=-1)
                Cmax = torch.tensor(
                    [input["logit_is_boundary_bottom" + sfx].shape[-1] - 1],
                    device=C.device,
                    dtype=C.dtype,
                )
                C = torch.min(C, Cmax)
                loss_term = F.cross_entropy(
                    input["logit_is_boundary_bottom" + sfx].transpose(-2, -1),
                    C,
                    reduction=inner_reduction,
                )
                if self.ignore_lines_during_passive:
                    loss_term = loss_term * (1 - target["is_passive"])
                    loss_term = torch.sum(loss_term)
                    if self.reduction == "mean":
                        loss_term = loss_term / torch.sum(1 - target["is_passive"])
                loss += weight * loss_term
            else:
                loss_term = F.binary_cross_entropy(
                    input["p_is_below_bottom" + sfx],
                    target["mask_bot" + sfx].to(
                        input["p_is_below_bottom" + sfx].device,
                        input["p_is_below_bottom" + sfx].dtype,
                    ),
                    reduction=inner_reduction,
                )
                if self.ignore_lines_during_passive:
                    loss_term = loss_term * (1 - target["is_passive"].unsqueeze(-1))
                    loss_term = torch.sum(loss_term)
                    if self.reduction == "mean":
                        loss_term = loss_term / torch.sum(1 - target["is_passive"])
                loss += weight * loss_term

        if self.removed_segment:
            loss += self.removed_segment * F.binary_cross_entropy_with_logits(
                input["logit_is_removed"],
                target["is_removed"].to(
                    input["logit_is_removed"].device, input["logit_is_removed"].dtype
                ),
                reduction=self.reduction,
            )

        if self.passive:
            loss += self.passive * F.binary_cross_entropy_with_logits(
                input["logit_is_passive"],
                target["is_passive"].to(
                    input["logit_is_passive"].device, input["logit_is_passive"].dtype
                ),
                reduction=self.reduction,
            )

        for sfx in ("", "-original", "-ntob"):
            weight = self.patch
            if sfx != "":
                weight *= self.auxillary
            if not weight:
                continue
            loss += weight * F.binary_cross_entropy_with_logits(
                input["logit_is_patch" + sfx],
                target["mask_patches" + sfx].to(
                    input["logit_is_patch" + sfx].device,
                    input["logit_is_patch" + sfx].dtype,
                ),
                reduction=self.reduction,
            )

        if self.overall:
            loss += self.overall * F.binary_cross_entropy(
                input["p_keep_pixel"],
                target["mask"].to(
                    input["p_keep_pixel"].device, input["p_keep_pixel"].dtype
                ),
                reduction=self.reduction,
            )

        return loss
