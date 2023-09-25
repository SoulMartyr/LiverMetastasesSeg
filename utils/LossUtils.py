import torch
import torch.nn.functional as F
from typing import Union, List, Tuple


def bce_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(preds, targets, reduction='mean')


def ce_loss(preds: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(preds, targets, weight, reduction="mean")


def dice_loss(preds: torch.Tensor, targets: torch.Tensor, tgt_channels: List[int] = [0], is_softmax: bool = False, weights: List[float] = None, in_detail: bool = False) -> Union[Tuple[List[torch.FloatTensor], torch.FloatTensor], torch.FloatTensor]:
    if is_softmax:
        preds = torch.softmax(preds, dim=1)
    else:
        preds = torch.sigmoid(preds)

    preds = torch.clamp(preds, min=1e-7, max=1).to(torch.float32)

    eps = 1e-5
    B, C = preds.size(0), preds.size(1)
    num_tgt_channels = len(tgt_channels)

    axis_order = (1, 0) + tuple(range(2, preds.dim()))

    preds = preds.permute(axis_order).reshape(C, B, -1)
    targets = targets.permute(axis_order).reshape(C, B, -1)

    preds, targets = preds[tgt_channels], targets[tgt_channels]

    assert preds.size() == targets.size()

    assert not weights or len(weights) == preds.size(0)

    intersection = (preds * targets).sum(-1)

    union = (preds + targets).sum(-1)

    loss_channel = (1 - (2. * intersection + eps) / (union + eps)).sum(-1) / B

    if weights is not None:
        loss_channel *= torch.FloatTensor(weights)

    loss = loss_channel.sum(-1) / num_tgt_channels

    if in_detail:
        loss_list = [loss_channel[i].item() for i in range(num_tgt_channels)]
        return loss_list, loss
    else:
        return loss


def generalized_dice_loss(preds: torch.Tensor, targets: torch.Tensor, tgt_channels: List[int] = [0], is_softmax: bool = False,
                          weight_type: str = 'square', in_detail: bool = False) -> Union[Tuple[List[torch.FloatTensor], torch.FloatTensor], torch.FloatTensor]:
    if is_softmax:
        preds = torch.softmax(preds, dim=1)
    else:
        preds = torch.sigmoid(preds)

    preds = torch.clamp(preds, min=1e-7, max=1).to(torch.float32)

    eps = 1e-5

    B, C = preds.size(0), preds.size(1)
    num_tgt_channels = len(tgt_channels)

    axis_order = (1, 0) + tuple(range(2, preds.dim()))

    preds = preds.permute(axis_order).reshape(C, B, -1)
    targets = targets.permute(axis_order).reshape(C, B, -1)

    preds, targets = preds[tgt_channels], targets[tgt_channels]

    assert preds.size() == targets.size()

    target_sum = targets.sum(-1)

    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :', weight_type)

    intersection = (preds * targets).sum(-1) * class_weights
    union = (preds + targets).sum(-1) * class_weights + eps

    loss_channel = (1 - 2. * (2. * intersection + eps) /
                    (union + eps)).sum(-1) / B
    loss = loss_channel.sum() / num_tgt_channels

    if in_detail:
        loss_list = [loss_channel[i].item() for i in range(num_tgt_channels)]
        return loss_list, loss
    else:
        return loss


def softmax_binary_torch(tensor: torch.Tensor) -> torch.Tensor:
    num_classes = tensor.size(1)
    tensor = torch.argmax(tensor, dim=1)
    one_hot = [(tensor == i).type(
        torch.uint8) for i in range(num_classes)]
    return torch.stack(one_hot, dim=1)


def dice_with_norm_binary(preds: torch.Tensor, targets: torch.Tensor, tgt_channel: int,  is_softmax: bool = False, threshold: float = 0.5) -> float:
    # assert preds.size() == targets.size(), "the size of predict and target must be equal."
    if is_softmax:
        preds = softmax_binary_torch(preds)
    else:
        preds = torch.sigmoid(preds)
        preds = torch.clamp(preds, min=1e-7, max=1).to(torch.float32)

        preds[preds >= threshold] = 1
        preds[preds < threshold] = 0

    eps = 1e-5

    B = preds.size(0)
    dice = 0.
    for i in range(B):

        pred = preds[i, tgt_channel].flatten()
        target = targets[i, tgt_channel].flatten()

        intersection = (pred * target).sum()
        union = (pred + target).sum() + eps

        dice += (2. * intersection / union).item()
    return dice / B


def dice_with_binary(preds: torch.Tensor, targets: torch.Tensor, tgt_channel: int, is_softmax: bool = False, threshold: float = 0.5) -> float:
    # assert preds.size() == targets.size(), "the size of predict and target must be equal."
    if is_softmax:
        preds = softmax_binary_torch(preds)
    else:
        preds[preds >= threshold] = 1
        preds[preds < threshold] = 0

    eps = 1e-5

    B = preds.size(0)
    dice = 0.
    for i in range(B):

        pred = preds[i, tgt_channel].flatten()
        target = targets[i, tgt_channel].flatten()

        intersection = (pred * target).sum()
        union = (pred + target).sum() + eps

        dice += (2. * intersection / union).item()
    return dice / B


def VOE(preds: torch.Tensor, targets: torch.Tensor, tgt_channel: int, is_softmax: bool = False, threshold: float = 0.5) -> float:
    if is_softmax:
        preds = softmax_binary_torch(preds)
    else:
        preds = torch.sigmoid(preds)
        preds = torch.clamp(preds, min=1e-7, max=1).to(torch.float32)

        preds[preds >= threshold] = 1
        preds[preds < threshold] = 0

    if threshold is not None:
        if preds.size(1) == 1:
            preds = torch.sigmoid(preds)
        else:
            preds = torch.softmax(preds, dim=1)
        preds = torch.clamp(preds, min=1e-7, max=1).to(torch.float32)

        preds[preds >= threshold] = 1
        preds[preds < threshold] = 0

    eps = 1e-5
    B = preds.size(0)

    pred = preds[:, tgt_channel].flatten()
    target = targets[:, tgt_channel].flatten()

    intersection = (pred * target).sum()
    union = (pred + target).sum()

    voe = (intersection / (union - intersection + eps)).item()
    return voe / B


if __name__ == "__main__":
    pred = torch.Tensor([[[
        [[1, 1, 1, 0],
         [1, 0, 0, 0],
         [1, 0, 0, 0],
         [0, 1, 1, 0]]],
        [[[0, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 1, 0, 0],
          [0, 1, 1, 0]]]],
        [[
            [[0, 1, 1, 0],
             [0.8, 0, 0, 0],
             [0.8, 0, 0, 0],
             [0, 1, 1, 0]]],
            [[[0, 0, 0, 0],
              [0.1, 0.9, 0, 0],
              [0.1, 0.9, 0, 0],
              [0, 1, 1, 0]]]]])

    gt = torch.Tensor([[[
        [[0, 1, 1, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]]],
        [[[1, 0, 0, 1],
          [0, 1, 1, 0],
          [0, 1, 1, 0],
          [1, 0, 0, 1]]]],
        [[
            [[0, 1, 1, 0],
             [1, 0, 0, 1],
             [1, 0, 0, 1],
             [0, 1, 1, 0]]],
            [[[1, 0, 0, 1],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [1, 0, 0, 1]]]]])
    # pred = torch.randn([2, 4, 4, 4, 4])
    # pred2 = torch.randn([2, 4, 4, 4, 4])
    # target = torch.zeros([2, 1, 4, 4, 4])
    # target[:, :, 1, 1, 1] = 1
    # target[:, :, 2, 2, 2] = 1
    # target[:, :, 0, 0, 0] = 1
    # target[:, :, 1, 3, 3] = 1
    # print(pred.shape, target.shape)
    print(pred.shape, gt.shape)
    print(dice_loss(pred, gt, tgt_channels=[1], in_detail=True))
    # rand = torch.ones([12, 224, 224])
    # r_rand = rotate(rand, interpolation=InterpolationMode.BILINEAR,
    #                 angle=0., expand=0., fill=False)

    # print(r_rand == rand)
