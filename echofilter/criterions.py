import torch


def mask_accuracy(input, target, threshold=0.5, has_batch_dim=True, reduction='mean'):
    # Binarise masks
    input = (input > 0.5)
    target = (target > 0.5)
    # Reshape so pixels are vectorised by batch
    shape = [-1]
    if has_batch_dim:
        n_batch = input.shape[0]
        shape = [n_batch] + shape
    input = input.reshape(shape)
    target = target.reshape(shape)

    # Measure hit rate and number of samples
    hits = (input == target).sum(-1)
    count = input.size(-1)
    output = hits.float() / count

    # Apply reduction
    if not has_batch_dim:
        return output
    if reduction == 'none':
        return output
    elif reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    else:
        raise ValueError('Unsupported reduction value: {}'.format(reduction))


def mask_accuracy_with_logits(input, *args, **kwargs):
    return mask_accuracy(torch.sigmoid(input), *args, **kwargs)


def mask_precision(input, target, threshold=0.5, has_batch_dim=True, reduction='mean'):
    # Binarise masks
    input = (input > 0.5)
    target = (target > 0.5)
    # Reshape so pixels are vectorised by batch
    shape = [-1]
    if has_batch_dim:
        n_batch = input.shape[0]
        shape = [n_batch] + shape
    input = input.reshape(shape)
    target = target.reshape(shape)

    # Measure true positives and total predicted positives
    true_p = (input & target).sum(-1)
    predicted_p = input.sum(-1)
    output = true_p.float() / predicted_p.float()
    # Handle division by 0: If there were positives predicted, all were wrong.
    output[predicted_p == 0] = 0

    # Apply reduction
    if not has_batch_dim:
        return output
    if reduction == 'none':
        return output
    elif reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    else:
        raise ValueError('Unsupported reduction value: {}'.format(reduction))


def mask_precision_with_logits(input, *args, **kwargs):
    return mask_precision(torch.sigmoid(input), *args, **kwargs)


def mask_recall(input, target, threshold=0.5, has_batch_dim=True, reduction='mean'):
    # Binarise masks
    input = (input > 0.5)
    target = (target > 0.5)
    # Reshape so pixels are vectorised by batch
    shape = [-1]
    if has_batch_dim:
        n_batch = input.shape[0]
        shape = [n_batch] + shape
    input = input.reshape(shape)
    target = target.reshape(shape)

    # Measure true positives and actual positives
    true_p = (input & target).sum(-1)
    ground_truth_p = target.sum(-1)
    output = true_p.float() / ground_truth_p.float()
    # Handle division by 0: If there were no positives to find, all were found.
    output[ground_truth_p == 0] = 1

    # Apply reduction
    if not has_batch_dim:
        return output
    if reduction == 'none':
        return output
    elif reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    else:
        raise ValueError('Unsupported reduction value: {}'.format(reduction))


def mask_recall_with_logits(input, *args, **kwargs):
    return mask_recall(torch.sigmoid(input), *args, **kwargs)


def mask_f1_score(input, target, reduction='mean', **kwargs):
    precision = mask_precision(input, target, reduction='none', **kwargs)
    recall = mask_recall(input, target, reduction='none', **kwargs)
    sum_pr = precision + recall
    output = 2 * precision * recall / sum_pr
    # Handle division by 0: If both precision and recall are 0, f1 is 0 too.
    output[sum_pr == 0] = 0

    # Apply reduction
    if reduction == 'none':
        return output
    elif reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    else:
        raise ValueError('Unsupported reduction value: {}'.format(reduction))


def mask_f1_score_with_logits(input, *args, **kwargs):
    return mask_f1_score(torch.sigmoid(input), *args, **kwargs)


def mask_jaccard_index(input, target, threshold=0.5, has_batch_dim=True, reduction='mean'):
    # Binarise masks
    input = (input > 0.5)
    target = (target > 0.5)
    # Reshape so pixels are vectorised by batch
    shape = [-1]
    if has_batch_dim:
        n_batch = input.shape[0]
        shape = [n_batch] + shape
    input = input.reshape(shape)
    target = target.reshape(shape)

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
    if not has_batch_dim:
        return output
    if reduction == 'none':
        return output
    elif reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    else:
        raise ValueError('Unsupported reduction value: {}'.format(reduction))


def mask_jaccard_index_with_logits(input, *args, **kwargs):
    return mask_jaccard_index(torch.sigmoid(input), *args, **kwargs)
