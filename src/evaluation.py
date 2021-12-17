import torch


def iou(outputs: torch.Tensor, labels: torch.Tensor):

    safety = 1e-6

    intersection = (outputs & labels).float().sum((1, 2, 3))
    union = (outputs | labels).float().sum((1, 2, 3))

    iou = (intersection + safety) / (union + safety)

    return iou
