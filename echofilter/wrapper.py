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
        else:
            raise ValueError('Unsupported "top" parameter: {}'.format(self.params['top']))
        if self.params['bottom'] == 'mask':
            outputs['logit_is_below_bottom'] = logits[:, i]
            outputs['p_is_below_bottom'] = torch.sigmoid(outputs['logit_is_below_bottom'])
            i += 1
        else:
            raise ValueError('Unsupported "bottom" parameter: {}'.format(self.params['bottom']))
        outputs['p_keep_pixel'] = (
            1.
            * (1 - outputs['p_is_above_top'])
            * (1 - outputs['p_is_below_bottom'])
        )
        outputs['mask_keep_pixel'] = (
            1.
            * (outputs['p_is_above_top'] < 0.5)
            * (outputs['p_is_below_bottom'] < 0.5)
        )
        return outputs


class EchofilterLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction='mean', top_mask=1., bottom_mask=1.):
        super(EchofilterLoss, self).__init__(None, None, reduction)
        self.top_mask = top_mask
        self.bottom_mask = bottom_mask

    def forward(self, input, target):
        loss = 0

        if not self.top_mask:
            pass
        elif 'logit_is_above_top' in input:
            loss += self.top_mask * F.binary_cross_entropy_with_logits(
                input['logit_is_above_top'], target['mask_top'], reduction=self.reduction
            )
        else:
            loss += self.top_mask * F.binary_cross_entropy(
                input['p_is_above_top'], target['mask_top'], reduction=self.reduction
            )

        if not self.bottom_mask:
            pass
        elif 'logit_is_below_bottom' in input:
            loss += self.bottom_mask * F.binary_cross_entropy_with_logits(
                input['logit_is_below_bottom'], target['mask_bot'], reduction=self.reduction
            )
        else:
            loss += self.bottom_mask * F.binary_cross_entropy(
                input['p_is_below_bottom'], target['mask_bot'], reduction=self.reduction
            )

        return loss
