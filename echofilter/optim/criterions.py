import torch


def _binarise_and_reshape(arg, threshold=0.5, ndim=None):
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
    return mask_active_fraction(torch.sigmoid(input), *args, **kwargs)


def mask_accuracy(input, target, threshold=0.5, ndim=None, reduction="mean"):
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
    return mask_accuracy(torch.sigmoid(input), *args, **kwargs)


def mask_precision(input, target, threshold=0.5, ndim=None, reduction="mean"):
    # Binarise and reshape masks
    input = _binarise_and_reshape(input, threshold=threshold, ndim=ndim)
    target = _binarise_and_reshape(target, threshold=threshold, ndim=ndim)

    # Measure true positives and total predicted positives
    true_p = (input & target).sum(-1)
    predicted_p = input.sum(-1)
    output = true_p.float() / predicted_p.float()
    # Handle division by 0: If there were no positives predicted, use 0.5.
    output[predicted_p == 0] = 0.5

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
    return mask_precision(torch.sigmoid(input), *args, **kwargs)


def mask_recall(input, target, threshold=0.5, ndim=None, reduction="mean"):
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
    return mask_recall(torch.sigmoid(input), *args, **kwargs)


def mask_f1_score(input, target, reduction="mean", **kwargs):
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
    return mask_f1_score(torch.sigmoid(input), *args, **kwargs)


def mask_jaccard_index(input, target, threshold=0.5, ndim=None, reduction="mean"):
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
    return mask_jaccard_index(torch.sigmoid(input), *args, **kwargs)