import torch
import torch.nn.functional as F
from torch.nn import Module
from typing import Union, List, Tuple, Dict

from .AuxUtils import AvgOutput


class Loss(Module):
    def __init__(self, num_classes: int, include_background: bool,  is_softmax: bool, loss_types: List[str] = None, class_weight: Union[List[float], None] = None, loss_weight: Union[List[float], None] = None):
        super(Loss, self).__init__()

        self.loss_func_dict = {"mse": mse_loss,
                               "bce": bce_loss,
                               "ce": ce_loss,
                               "dice": dice_loss,
                               "gdice": generalized_dice_loss
                               }

        for loss_type in loss_types:
            if loss_type not in self.loss_func_dict:
                raise ValueError("error loss type:{}".format(loss_type))
        self.loss_types = loss_types

        start_channel = 0 if include_background or not is_softmax else 1
        end_channel = num_classes + 1 if is_softmax else num_classes
        self.channel_range = [i for i in range(
            start_channel, end_channel)]

        assert len(loss_types) == len(loss_weight)

        self.num_classes = num_classes
        self.include_background = include_background
        self.is_softmax = is_softmax
        self.class_weight = class_weight
        self.loss_weight = loss_weight

    def __len__(self) -> int:
        return len(self.loss_types)

    def forward(self, pred, gt) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_list = []

        for loss_type in self.loss_types:
            if loss_type == "bce" or loss_type == "mse":
                loss_list.append(self.loss_func_dict[loss_type](pred, gt))
            elif loss_type == "ce":
                loss_list.append(self.loss_func_dict[loss_type](
                    pred, gt, self.class_weight))
            elif loss_type == "dice":
                loss_list.append(self.loss_func_dict[loss_type](
                    pred, gt, self.channel_range, self.is_softmax, self.class_weight))
            elif loss_type == "gdice":
                loss_list.append(self.loss_func_dict[loss_type](
                    pred, gt, self.channel_range, self.is_softmax, self.class_weight))

        loss = torch.sum(torch.stack(loss_list) * self.loss_weight)
        loss_dict = dict(zip(self.loss_types, loss_list))
        return loss, loss_dict


def mse_loss(preds: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(preds, gts, reduction='mean')


def bce_loss(preds: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(preds, gts, reduction='mean')


def ce_loss(preds: torch.Tensor, gts: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(preds, gts, weight, reduction="mean")


def dice_loss(preds: torch.Tensor, gts: torch.Tensor, tgt_channels: List[int] = [0], is_softmax: bool = False,
              class_weights: Union[torch.Tensor, None] = None, in_detail: bool = False) -> Union[Tuple[List[torch.FloatTensor], torch.FloatTensor], torch.FloatTensor]:
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
    gts = gts.permute(axis_order).reshape(C, B, -1)

    preds, gts = preds[tgt_channels], gts[tgt_channels]

    assert preds.size() == gts.size()

    intersection = (preds * gts).sum(-1)

    union = (preds + gts).sum(-1)

    loss_channel = (1 - (2. * intersection + eps) / (union + eps)).sum(-1) / B

    if class_weights is None:
        loss_channel /= num_tgt_channels
    else:
        loss_channel *= class_weights[tgt_channels]

    loss = loss_channel.sum(-1)

    if in_detail:
        loss_list = [loss_channel[i].item() for i in range(num_tgt_channels)]
        return loss_list, loss
    else:
        return loss


def generalized_dice_loss(preds: torch.Tensor, gts: torch.Tensor, tgt_channels: List[int] = [0], is_softmax: bool = False, class_weights: Union[torch.Tensor, None] = None,
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
    gts = gts.permute(axis_order).reshape(C, B, -1)

    preds, gts = preds[tgt_channels], gts[tgt_channels]

    assert preds.size() == gts.size()

    target_sum = gts.sum(-1)

    if weight_type == 'square':
        weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :', weight_type)

    intersection = (preds * gts).sum(-1) * weights
    union = (preds + gts).sum(-1) * weights + eps

    loss_channel = (1 - (2. * intersection + eps) /
                    (union + eps)).sum(-1) / B

    if class_weights is None:
        loss_channel /= num_tgt_channels
    else:
        loss_channel *= class_weights[tgt_channels]

    loss = loss_channel.sum(-1)

    if in_detail:
        loss_list = [loss_channel[i].item() for i in range(num_tgt_channels)]
        return loss_list, loss
    else:
        return loss


def sigmoid_binary_torch(tensor: torch.Tensor, threshold: List[float] = [0.5]) -> torch.Tensor:
    for idx, thres in enumerate(threshold):
        tensor[:, idx][tensor[:, idx] >= thres] = 1
        tensor[:, idx][tensor[:, idx] < thres] = 0
    return tensor


def softmax_binary_torch(tensor: torch.Tensor) -> torch.Tensor:
    num_classes = tensor.size(1)
    tensor = torch.argmax(tensor, dim=1)
    one_hot = [(tensor == i).type(
        torch.uint8) for i in range(num_classes)]
    return torch.stack(one_hot, dim=1)


def dice_with_norm_binary(preds: torch.Tensor, gts: torch.Tensor, tgt_channel: int,  is_softmax: bool = False, threshold: List[float] = [0.5]) -> float:
    # assert preds.size() == gts.size(), "the size of predict and gt must be equal."
    if is_softmax:
        preds = softmax_binary_torch(preds)
    else:
        preds = torch.sigmoid(preds)
        preds = torch.clamp(preds, min=1e-7, max=1).to(torch.float32)

        preds = sigmoid_binary_torch(preds, threshold)

    eps = 1e-5
    B = preds.size(0)
    dice = 0.

    for i in range(B):

        pred = preds[i, tgt_channel].flatten()
        gt = gts[i, tgt_channel].flatten()

        intersection = (pred * gt).sum()
        union = (pred + gt).sum() + eps

        dice += (2. * intersection / union).item()

    return dice / B


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
    # gt = torch.zeros([2, 1, 4, 4, 4])
    # gt[:, :, 1, 1, 1] = 1
    # gt[:, :, 2, 2, 2] = 1
    # gt[:, :, 0, 0, 0] = 1
    # gt[:, :, 1, 3, 3] = 1
    # print(pred.shape, gt.shape)
    print(pred.shape, gt.shape)
    print(dice_loss(pred, gt, tgt_channels=[1], in_detail=True))
    # rand = torch.ones([12, 224, 224])
    # r_rand = rotate(rand, interpolation=InterpolationMode.BILINEAR,
    #                 angle=0., expand=0., fill=False)

    # print(r_rand == rand)
