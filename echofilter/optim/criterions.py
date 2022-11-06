"""
Evaluation criterions.
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

import torch


def _binarise_and_reshape(arg, threshold=0.5, ndim=None):
    """
    Binarise and partially flatten a tensor.

    Parameters
    ----------
    arg : array_like
        Input tensor or array.
    threshold : float, optional
        Threshold which entries in `arg` must exceed. Default is `0.5`.
    ndim : int or None
        Number of dimensions to keep. If `None`, only the first (batch)
        dimension is kept and the rest are flattened. Default is `None`.

    Returns
    -------
    array_like
        A :class:`numpy.ndarray` or :class:`torch.Tensor` (corresponding to
        the type of `arg`), but partially flattened and binarised.
    """
    # Binarise mask
    arg = arg > threshold
    # Reshape so pixels are vectorised by batch
    if ndim is None:
        shape = [arg.shape[0], -1]
    else:
        shape = list(arg.shape)
        shape = shape[:-ndim] + [-1]
    arg = arg.reshape(shape)
    return arg


def mask_active_fraction(input, threshold=0.5, ndim=None, reduction="mean"):
    """
    Measure the fraction of input which exceeds a threshold.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor.
    threshold : float, optional
        Threshold which entries in `input` must exceed. Default is `0.5`.
    ndim : int or None
        Number of dimensions to keep. If `None`, only the first (batch)
        dimension is kept and the rest are flattened. Default is `None`.
    reduction : `"none"` or `"mean"` or `"sum"`, optional
        Specifies the reduction to apply to the output:
        `"none"` | `"mean"` | `"sum"`.
        `"none"`: no reduction will be applied,
        `"mean"`: the sum of the output will be divided by the number of
        elements in the output,
        `"sum"`: the output will be summed.
        Default: `"mean"`.

    Returns
    -------
    torch.Tensor
        The fraction of `input` which exceeds `threshold`, with shaped
        corresponding to `reduction`.
    """
    # Binarise and reshape mask
    input = _binarise_and_reshape(input, threshold=threshold, ndim=ndim)

    # Measure hit rate and number of samples
    output = input.sum(-1).float() / input.size(-1)

    # Apply reduction
    if reduction == "none":
        return output
    elif reduction == "mean":
        return output.mean()
    elif reduction == "sum":
        return output.sum()
    else:
        raise ValueError("Unsupported reduction value: {}".format(reduction))


def mask_active_fraction_with_logits(input, *args, **kwargs):
    """
    Convert logits to probabilities with sigmoid, then measure the fraction
    of the tensor which exceeds a threshold.

    See also
    --------
    mask_active_fraction
    """
    return mask_active_fraction(torch.sigmoid(input), *args, **kwargs)


def mask_accuracy(input, target, threshold=0.5, ndim=None, reduction="mean"):
    """
    Measure the fraction of input which exceeds a threshold.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor.
    target : torch.Tensor
        Target tensor, the same shape as `input`.
    threshold : float, optional
        Threshold which entries in `input` and `target` must exceed to be
        binarised as the positive class. Default is `0.5`.
    ndim : int or None
        Number of dimensions to keep. If `None`, only the first (batch)
        dimension is kept and the rest are flattened. Default is `None`.
    reduction : `"none"` or `"mean"` or `"sum"`, optional
        Specifies the reduction to apply to the output:
        `"none"` | `"mean"` | `"sum"`.
        `"none"`: no reduction will be applied,
        `"mean"`: the sum of the output will be divided by the number of
        elements in the output,
        `"sum"`: the output will be summed.
        Default: `"mean"`.

    Returns
    -------
    torch.Tensor
        The fraction of `input` which has the same class as `target` after
        thresholding.
    """
    # Binarise and reshape masks
    input = _binarise_and_reshape(input, threshold=threshold, ndim=ndim)
    target = _binarise_and_reshape(target, threshold=threshold, ndim=ndim)

    # Measure hit rate and number of samples
    hits = (input == target).sum(-1)
    count = input.size(-1)
    output = hits.float() / count

    # Apply reduction
    if reduction == "none":
        return output
    elif reduction == "mean":
        return output.mean()
    elif reduction == "sum":
        return output.sum()
    else:
        raise ValueError("Unsupported reduction value: {}".format(reduction))


def mask_accuracy_with_logits(input, *args, **kwargs):
    """
    Measure the accuracy between input and target, after passing `input`
    through a sigmoid function.

    See also
    --------
    mask_accuracy
    """
    return mask_accuracy(torch.sigmoid(input), *args, **kwargs)


def mask_precision(input, target, threshold=0.5, ndim=None, reduction="mean"):
    """
    Measure the precision of the input as compared to a ground truth target,
    after binarising with a threshold.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor.
    target : torch.Tensor
        Target tensor, the same shape as `input`.
    threshold : float, optional
        Threshold which entries in `input` and `target` must exceed to be
        binarised as the positive class. Default is `0.5`.
    ndim : int or None
        Number of dimensions to keep. If `None`, only the first (batch)
        dimension is kept and the rest are flattened. Default is `None`.
    reduction : `"none"` or `"mean"` or `"sum"`, optional
        Specifies the reduction to apply to the output:
        `"none"` | `"mean"` | `"sum"`.
        `"none"`: no reduction will be applied,
        `"mean"`: the sum of the output will be divided by the number of
        elements in the output,
        `"sum"`: the output will be summed.
        Default: `"mean"`.

    Returns
    -------
    torch.Tensor
        The precision of `input` as compared to `target` after thresholding.
        The fraction of predicted positive cases, `input > 0.5`, which are
        true positive cases (`input > 0.5 and `target > 0.5`).
        If there are no predicted positives, the output is `0` if there are
        any positives to predict and `1` if there are none.
    """
    # Binarise and reshape masks
    input = _binarise_and_reshape(input, threshold=threshold, ndim=ndim)
    target = _binarise_and_reshape(target, threshold=threshold, ndim=ndim)

    # Measure true positives and total predicted positives
    true_p = (input & target).sum(-1)
    predicted_p = input.sum(-1)
    output = true_p.float() / predicted_p.float()
    # Handle division by 0: If there were no positives predicted, check whether
    # there were any to find.
    output[predicted_p == 0] = (true_p[predicted_p == 0] == 0).float()

    # Apply reduction
    if reduction == "none":
        return output
    elif reduction == "mean":
        return output.mean()
    elif reduction == "sum":
        return output.sum()
    else:
        raise ValueError("Unsupported reduction value: {}".format(reduction))


def mask_precision_with_logits(input, *args, **kwargs):
    """
    Convert logits to probabilities with sigmoid, apply a threshold, then
    measure the precision of the tensor as compared to ground truth.

    See also
    --------
    mask_precision
    """
    return mask_precision(torch.sigmoid(input), *args, **kwargs)


def mask_recall(input, target, threshold=0.5, ndim=None, reduction="mean"):
    """
    Measure the recall of the input as compared to a ground truth target,
    after binarising with a threshold.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor.
    target : torch.Tensor
        Target tensor, the same shape as `input`.
    threshold : float, optional
        Threshold which entries in `input` and `target` must exceed to be
        binarised as the positive class. Default is `0.5`.
    ndim : int or None
        Number of dimensions to keep. If `None`, only the first (batch)
        dimension is kept and the rest are flattened. Default is `None`.
    reduction : `"none"` or `"mean"` or `"sum"`, optional
        Specifies the reduction to apply to the output:
        `"none"` | `"mean"` | `"sum"`.
        `"none"`: no reduction will be applied,
        `"mean"`: the sum of the output will be divided by the number of
        elements in the output,
        `"sum"`: the output will be summed.
        Default: `"mean"`.

    Returns
    -------
    torch.Tensor
        The recall of `input` as compared to `target` after thresholding.
        The fraction of true positive cases, `target > 0.5`, which are
        true positive cases (`input > 0.5 and `target > 0.5`).
        If there are no true positives, the output is `1`.
    """
    # Binarise and reshape masks
    input = _binarise_and_reshape(input, threshold=threshold, ndim=ndim)
    target = _binarise_and_reshape(target, threshold=threshold, ndim=ndim)

    # Measure true positives and actual positives
    true_p = (input & target).sum(-1)
    ground_truth_p = target.sum(-1)
    output = true_p.float() / ground_truth_p.float()
    # Handle division by 0: If there were no positives to find, all were found.
    output[ground_truth_p == 0] = 1

    # Apply reduction
    if reduction == "none":
        return output
    elif reduction == "mean":
        return output.mean()
    elif reduction == "sum":
        return output.sum()
    else:
        raise ValueError("Unsupported reduction value: {}".format(reduction))


def mask_recall_with_logits(input, *args, **kwargs):
    """
    Convert logits to probabilities with sigmoid, apply a threshold, then
    measure the recall of the tensor as compared to ground truth.

    See also
    --------
    mask_recall
    """
    return mask_recall(torch.sigmoid(input), *args, **kwargs)


def mask_f1_score(input, target, reduction="mean", **kwargs):
    """
    Measure the F1-score of the input as compared to a ground truth target,
    after binarising with a threshold.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor.
    target : torch.Tensor
        Target tensor, the same shape as `input`.
    threshold : float, optional
        Threshold which entries in `input` and `target` must exceed to be
        binarised as the positive class. Default is `0.5`.
    ndim : int or None
        Number of dimensions to keep. If `None`, only the first (batch)
        dimension is kept and the rest are flattened. Default is `None`.
    reduction : `"none"` or `"mean"` or `"sum"`, optional
        Specifies the reduction to apply to the output:
        `"none"` | `"mean"` | `"sum"`.
        `"none"`: no reduction will be applied,
        `"mean"`: the sum of the output will be divided by the number of
        elements in the output,
        `"sum"`: the output will be summed.
        Default: `"mean"`.

    Returns
    -------
    torch.Tensor
        The F1-score of `input` as compared to `target` after thresholding.
        The F1-score is the harmonic mean of precision and recall.

    See also
    --------
    mask_precision
    mask_recall
    """
    precision = mask_precision(input, target, reduction="none", **kwargs)
    recall = mask_recall(input, target, reduction="none", **kwargs)
    sum_pr = precision + recall
    output = 2 * precision * recall / sum_pr
    # Handle division by 0: If both precision and recall are 0, f1 is 0 too.
    output[sum_pr == 0] = 0

    # Apply reduction
    if reduction == "none":
        return output
    elif reduction == "mean":
        return output.mean()
    elif reduction == "sum":
        return output.sum()
    else:
        raise ValueError("Unsupported reduction value: {}".format(reduction))


def mask_f1_score_with_logits(input, *args, **kwargs):
    """
    Convert logits to probabilities with sigmoid, apply a threshold, then
    measure the F1-score of the tensor as compared to ground truth.

    See also
    --------
    mask_f1_score
    """
    return mask_f1_score(torch.sigmoid(input), *args, **kwargs)


def mask_jaccard_index(input, target, threshold=0.5, ndim=None, reduction="mean"):
    """
    Measure the Jaccard Index (intersection over union) of the input as
    compared to a ground truth target, after binarising with a threshold.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor.
    target : torch.Tensor
        Target tensor, the same shape as `input`.
    threshold : float, optional
        Threshold which entries in `input` and `target` must exceed to be
        binarised as the positive class. Default is `0.5`.
    ndim : int or None
        Number of dimensions to keep. If `None`, only the first (batch)
        dimension is kept and the rest are flattened. Default is `None`.
    reduction : `"none"` or `"mean"` or `"sum"`, optional
        Specifies the reduction to apply to the output:
        `"none"` | `"mean"` | `"sum"`.
        `"none"`: no reduction will be applied,
        `"mean"`: the sum of the output will be divided by the number of
        elements in the output,
        `"sum"`: the output will be summed.
        Default: `"mean"`.

    Returns
    -------
    torch.Tensor
        The Jaccard Index of `input` as compared to `target`.
        The Jaccard Index is the number of elements where both `input` and
        `target` exceed `threshold`, divided by the number of elements where
        at least one of `input` and `target` exceeds `threshold`.
    """
    # Binarise and reshape masks
    input = _binarise_and_reshape(input, threshold=threshold, ndim=ndim)
    target = _binarise_and_reshape(target, threshold=threshold, ndim=ndim)

    # Use bitwise operators to determine intersection and union of masks
    intersect = input & target
    union = input | target
    # Use number of pixels at which intersect and union are activated
    intersect = intersect.sum(-1)
    union = union.sum(-1)
    output = intersect.float() / union.float()
    # Handle division by 0: If there is no union, the two masks match completely.
    output[union == 0] = 1

    # Apply reduction
    if reduction == "none":
        return output
    elif reduction == "mean":
        return output.mean()
    elif reduction == "sum":
        return output.sum()
    else:
        raise ValueError("Unsupported reduction value: {}".format(reduction))


def mask_jaccard_index_with_logits(input, *args, **kwargs):
    """
    Convert logits to probabilities with sigmoid, apply a threshold, then
    measure the Jaccard Index (intersection over union) of the tensor as
    compared to ground truth.

    See also
    --------
    mask_jaccard_index
    """
    return mask_jaccard_index(torch.sigmoid(input), *args, **kwargs)
