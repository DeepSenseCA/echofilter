"""
Model wrapper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class Echofilter(nn.Module):
    def __init__(self, model, top='mask', bottom='mask'):
        super(Echofilter, self).__init__()
        self.model = model
        self.params = {
            'top': top,
            'bottom': bottom,
        }

    def forward(self, x):
        logits = self.model(x)
        outputs = {}
        i = 0

        if self.params['top'] == 'mask':
            outputs['logit_is_above_top'] = logits[:, i]
            outputs['p_is_above_top'] = torch.sigmoid(outputs['logit_is_above_top'])
            i += 1
        elif self.params['top'] == 'boundary':
            outputs['logit_is_boundary_top'] = logits[:, i]
            outputs['p_is_boundary_top'] = F.softmax(outputs['logit_is_boundary_top'], dim=-1)
            outputs['p_is_above_top'] = torch.flip(
                torch.cumsum(torch.flip(outputs['p_is_boundary_top'], dims=(-1, )), dim=-1),
                dims=(-1, ),
            )
            # Due to floating point precision, max value can exceed 1.
            # Fix this by clipping the values to the appropriate range.
            outputs['p_is_above_top'].clamp_(0, 1)
            i += 1
        else:
            raise ValueError('Unsupported "top" parameter: {}'.format(self.params['top']))

        if self.params['bottom'] == 'mask':
            outputs['logit_is_below_bottom'] = logits[:, i]
            outputs['p_is_below_bottom'] = torch.sigmoid(outputs['logit_is_below_bottom'])
            i += 1
        elif self.params['bottom'] == 'boundary':
            outputs['logit_is_boundary_bottom'] = logits[:, i]
            outputs['p_is_boundary_bottom'] = F.softmax(outputs['logit_is_boundary_bottom'], dim=-1)
            outputs['p_is_below_bottom'] = torch.cumsum(outputs['p_is_boundary_bottom'], dim=-1)
            # Due to floating point precision, max value can exceed 1.
            # Fix this by clipping the values to the appropriate range.
            outputs['p_is_below_bottom'].clamp_(0, 1)
            i += 1
        else:
            raise ValueError('Unsupported "bottom" parameter: {}'.format(self.params['bottom']))

        outputs['logit_is_removed'] = torch.mean(logits[:, i], dim=-1)
        outputs['p_is_removed'] = torch.sigmoid(outputs['logit_is_removed'])
        i += 1

        outputs['logit_is_passive'] = torch.mean(logits[:, i], dim=-1)
        outputs['p_is_passive'] = torch.sigmoid(outputs['logit_is_passive'])
        i += 1

        outputs['logit_is_patch'] = logits[:, i]
        outputs['p_is_patch'] = torch.sigmoid(outputs['logit_is_patch'])
        i += 1

        outputs['p_keep_pixel'] = (
            1.
            * (1 - outputs['p_is_above_top'])
            * (1 - outputs['p_is_below_bottom'])
            * (1 - outputs['p_is_removed'].unsqueeze(-1))
            * (1 - outputs['p_is_passive'].unsqueeze(-1))
            * (1 - outputs['p_is_patch'])
        ).clamp_(0, 1)
        outputs['mask_keep_pixel'] = (
            1.
            * (outputs['p_is_above_top'] < 0.5)
            * (outputs['p_is_below_bottom'] < 0.5)
            * (outputs['p_is_removed'].unsqueeze(-1) < 0.5)
            * (outputs['p_is_passive'].unsqueeze(-1) < 0.5)
            * (outputs['p_is_patch'] < 0.5)
        )
        return outputs


class EchofilterLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(
        self,
        reduction='mean',
        top_mask=1.,
        bottom_mask=1.,
        removed_segment=1.,
        passive=1.,
        patch=1.,
        overall=1.,
    ):
        super(EchofilterLoss, self).__init__(None, None, reduction)
        self.top_mask = top_mask
        self.bottom_mask = bottom_mask
        self.removed_segment = removed_segment
        self.passive = passive
        self.patch = patch
        self.overall = overall

    def forward(self, input, target):
        loss = 0

        if not self.top_mask:
            pass
        elif 'logit_is_above_top' in input:
            loss += self.top_mask * F.binary_cross_entropy_with_logits(
                input['logit_is_above_top'], target['mask_top'], reduction=self.reduction,
            )
        elif 'logit_is_boundary_top' in input:
            X = target['mask_top']
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
                [input['logit_is_boundary_top'].shape[-1] - 1],
                device=C.device,
                dtype=C.dtype,
            )
            C = torch.min(C, Cmax)
            loss += self.top_mask * F.cross_entropy(
                input['logit_is_boundary_top'].transpose(-2, -1), C, reduction=self.reduction,
            )
        else:
            loss += self.top_mask * F.binary_cross_entropy(
                input['p_is_above_top'], target['mask_top'], reduction=self.reduction,
            )

        if not self.bottom_mask:
            pass
        elif 'logit_is_below_bottom' in input:
            loss += self.bottom_mask * F.binary_cross_entropy_with_logits(
                input['logit_is_below_bottom'], target['mask_bot'], reduction=self.reduction,
            )
        elif 'logit_is_boundary_bottom' in input:
            X = target['mask_bot']
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
                [input['logit_is_boundary_bottom'].shape[-1] - 1],
                device=C.device,
                dtype=C.dtype,
            )
            C = torch.min(C, Cmax)
            loss += self.bottom_mask * F.cross_entropy(
                input['logit_is_boundary_bottom'].transpose(-2, -1), C, reduction=self.reduction,
            )
        else:
            loss += self.bottom_mask * F.binary_cross_entropy(
                input['p_is_below_bottom'], target['mask_bot'], reduction=self.reduction,
            )

        if self.removed_segment:
            loss += self.removed_segment * F.binary_cross_entropy_with_logits(
                input['logit_is_removed'], target['is_removed'], reduction=self.reduction,
            )

        if self.passive:
            loss += self.passive * F.binary_cross_entropy_with_logits(
                input['logit_is_passive'], target['is_passive'], reduction=self.reduction,
            )

        if self.patch:
            loss += self.patch * F.binary_cross_entropy_with_logits(
                input['logit_is_patch'], target['mask_patches'], reduction=self.reduction,
            )

        if self.overall:
            loss += self.overall * F.binary_cross_entropy(
                input['p_keep_pixel'], target['mask'], reduction=self.reduction,
            )

        return loss
