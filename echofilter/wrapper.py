"""
Model wrapper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from .utils import TensorDict


class Echofilter(nn.Module):
    def __init__(self, model, top="boundary", bottom="boundary", surface="boundary"):
        super(Echofilter, self).__init__()
        self.model = model
        self.params = {
            "top": top,
            "bottom": bottom,
            "surface": surface,
        }

    def forward(self, x):
        logits = self.model(x)
        outputs = TensorDict()
        i = 0

        if self.params["top"] == "mask":
            outputs["logit_is_above_top"] = logits[:, i]
            outputs["p_is_above_top"] = torch.sigmoid(outputs["logit_is_above_top"])
            outputs["p_is_below_top"] = 1 - outputs["p_is_above_top"]
            i += 1
        elif self.params["top"] == "boundary":
            outputs["logit_is_boundary_top"] = logits[:, i]
            outputs["p_is_boundary_top"] = F.softmax(
                outputs["logit_is_boundary_top"], dim=-1
            )
            outputs["p_is_above_top"] = torch.flip(
                torch.cumsum(
                    torch.flip(outputs["p_is_boundary_top"], dims=(-1,)), dim=-1
                ),
                dims=(-1,),
            )
            outputs["p_is_below_top"] = torch.cumsum(
                outputs["p_is_boundary_top"], dim=-1
            )
            # Due to floating point precision, max value can exceed 1.
            # Fix this by clipping the values to the appropriate range.
            outputs["p_is_above_top"].clamp_(0, 1)
            outputs["p_is_below_top"].clamp_(0, 1)
            i += 1
        else:
            raise ValueError(
                'Unsupported "top" parameter: {}'.format(self.params["top"])
            )

        if self.params["bottom"] == "mask":
            outputs["logit_is_below_bottom"] = logits[:, i]
            outputs["p_is_below_bottom"] = torch.sigmoid(
                outputs["logit_is_below_bottom"]
            )
            outputs["p_is_above_bottom"] = 1 - outputs["p_is_below_bottom"]
            i += 1
        elif self.params["bottom"] == "boundary":
            outputs["logit_is_boundary_bottom"] = logits[:, i]
            outputs["p_is_boundary_bottom"] = F.softmax(
                outputs["logit_is_boundary_bottom"], dim=-1
            )
            outputs["p_is_below_bottom"] = torch.cumsum(
                outputs["p_is_boundary_bottom"], dim=-1
            )
            outputs["p_is_above_bottom"] = torch.flip(
                torch.cumsum(
                    torch.flip(outputs["p_is_boundary_bottom"], dims=(-1,)), dim=-1
                ),
                dims=(-1,),
            )
            # Due to floating point precision, max value can exceed 1.
            # Fix this by clipping the values to the appropriate range.
            outputs["p_is_below_bottom"].clamp_(0, 1)
            outputs["p_is_above_bottom"].clamp_(0, 1)
            i += 1
        else:
            raise ValueError(
                'Unsupported "bottom" parameter: {}'.format(self.params["bottom"])
            )

        outputs["logit_is_removed"] = torch.mean(logits[:, i], dim=-1)
        outputs["p_is_removed"] = torch.sigmoid(outputs["logit_is_removed"])
        i += 1

        outputs["logit_is_passive"] = torch.mean(logits[:, i], dim=-1)
        outputs["p_is_passive"] = torch.sigmoid(outputs["logit_is_passive"])
        i += 1

        outputs["logit_is_patch"] = logits[:, i]
        outputs["p_is_patch"] = torch.sigmoid(outputs["logit_is_patch"])
        i += 1

        if self.params["surface"] == "mask":
            outputs["logit_is_above_surface"] = logits[:, i]
            outputs["p_is_above_surface"] = torch.sigmoid(
                outputs["logit_is_above_surface"]
            )
            outputs["p_is_below_surface"] = 1 - outputs["p_is_above_surface"]
            i += 1
        elif self.params["surface"] == "boundary":
            outputs["logit_is_boundary_surface"] = logits[:, i]
            outputs["p_is_boundary_surface"] = F.softmax(
                outputs["logit_is_boundary_surface"], dim=-1
            )
            outputs["p_is_above_surface"] = torch.flip(
                torch.cumsum(
                    torch.flip(outputs["p_is_boundary_surface"], dims=(-1,)), dim=-1
                ),
                dims=(-1,),
            )
            outputs["p_is_below_surface"] = torch.cumsum(
                outputs["p_is_boundary_surface"], dim=-1
            )
            # Due to floating point precision, max value can exceed 1.
            # Fix this by clipping the values to the appropriate range.
            outputs["p_is_above_surface"].clamp_(0, 1)
            outputs["p_is_below_surface"].clamp_(0, 1)
            i += 1
        else:
            raise ValueError(
                'Unsupported "surface" parameter: {}'.format(self.params["surface"])
            )

        if self.params["top"] == "mask":
            outputs["logit_is_above_top-original"] = logits[:, i]
            outputs["p_is_above_top-original"] = torch.sigmoid(
                outputs["logit_is_above_top-original"]
            )
            outputs["p_is_below_top-original"] = 1 - outputs["p_is_above_top-original"]
            i += 1
        elif self.params["top"] == "boundary":
            outputs["logit_is_boundary_top-original"] = logits[:, i]
            outputs["p_is_boundary_top-original"] = F.softmax(
                outputs["logit_is_boundary_top-original"], dim=-1
            )
            outputs["p_is_above_top-original"] = torch.flip(
                torch.cumsum(
                    torch.flip(outputs["p_is_boundary_top-original"], dims=(-1,)),
                    dim=-1,
                ),
                dims=(-1,),
            )
            outputs["p_is_below_top-original"] = torch.cumsum(
                outputs["p_is_boundary_top-original"], dim=-1
            )
            # Due to floating point precision, max value can exceed 1.
            # Fix this by clipping the values to the appropriate range.
            outputs["p_is_above_top-original"].clamp_(0, 1)
            outputs["p_is_below_top-original"].clamp_(0, 1)
            i += 1
        else:
            raise ValueError(
                'Unsupported "top" parameter: {}'.format(self.params["top"])
            )

        if self.params["bottom"] == "mask":
            outputs["logit_is_below_bottom-original"] = logits[:, i]
            outputs["p_is_below_bottom-original"] = torch.sigmoid(
                outputs["logit_is_below_bottom-original"]
            )
            outputs["p_is_above_bottom-original"] = (
                1 - outputs["p_is_below_bottom-original"]
            )
            i += 1
        elif self.params["bottom"] == "boundary":
            outputs["logit_is_boundary_bottom-original"] = logits[:, i]
            outputs["p_is_boundary_bottom-original"] = F.softmax(
                outputs["logit_is_boundary_bottom-original"], dim=-1
            )
            outputs["p_is_below_bottom-original"] = torch.cumsum(
                outputs["p_is_boundary_bottom-original"], dim=-1
            )
            outputs["p_is_above_bottom-original"] = torch.flip(
                torch.cumsum(
                    torch.flip(outputs["p_is_boundary_bottom-original"], dims=(-1,)),
                    dim=-1,
                ),
                dims=(-1,),
            )
            # Due to floating point precision, max value can exceed 1.
            # Fix this by clipping the values to the appropriate range.
            outputs["p_is_below_bottom-original"].clamp_(0, 1)
            outputs["p_is_above_bottom-original"].clamp_(0, 1)
            i += 1
        else:
            raise ValueError(
                'Unsupported "bottom" parameter: {}'.format(self.params["bottom"])
            )

        outputs["logit_is_patch-original"] = logits[:, i]
        outputs["p_is_patch-original"] = torch.sigmoid(
            outputs["logit_is_patch-original"]
        )
        i += 1

        outputs["logit_is_patch-ntob"] = logits[:, i]
        outputs["p_is_patch-ntob"] = torch.sigmoid(outputs["logit_is_patch-ntob"])
        i += 1

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

    def forward(self, input, target):
        loss = 0

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
                loss += weight * F.binary_cross_entropy_with_logits(
                    input["logit_is_above_" + sfx],
                    target[target_key].to(
                        input["logit_is_above_" + sfx].device,
                        input["logit_is_above_" + sfx].dtype,
                    ),
                    reduction=self.reduction,
                )
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
                loss += weight * F.cross_entropy(
                    input["logit_is_boundary_" + sfx].transpose(-2, -1),
                    C,
                    reduction=self.reduction,
                )
            else:
                loss += weight * F.binary_cross_entropy(
                    input["p_is_above_" + sfx],
                    target[target_key],
                    reduction=self.reduction,
                )

        for sfx in ("", "-original"):
            weight = 1 if sfx == "" else self.auxillary
            if not self.bottom_mask:
                pass
            elif "logit_is_below_bottom" + sfx in input:
                loss += (
                    weight
                    * self.bottom_mask
                    * F.binary_cross_entropy_with_logits(
                        input["logit_is_below_bottom" + sfx],
                        target["mask_bot" + sfx].to(
                            input["logit_is_below_bottom" + sfx].device,
                            input["logit_is_below_bottom" + sfx].dtype,
                        ),
                        reduction=self.reduction,
                    )
                )
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
                loss += (
                    weight
                    * self.bottom_mask
                    * F.cross_entropy(
                        input["logit_is_boundary_bottom" + sfx].transpose(-2, -1),
                        C,
                        reduction=self.reduction,
                    )
                )
            else:
                loss += (
                    weight
                    * self.bottom_mask
                    * F.binary_cross_entropy(
                        input["p_is_below_bottom" + sfx],
                        target["mask_bot" + sfx].to(
                            input["p_is_below_bottom" + sfx].device,
                            input["p_is_below_bottom" + sfx].dtype,
                        ),
                        reduction=self.reduction,
                    )
                )

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
