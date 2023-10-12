import torch
import numpy as np
from typing import List, Tuple
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate


def rotate3D(tensor: torch.Tensor, axis: int, theta: int) -> torch.Tensor:
    expand = False
    fill = 0.0

    if axis == 0:
        tensor = rotate(tensor, interpolation=InterpolationMode.BILINEAR,
                        angle=theta, expand=expand, fill=fill)
    elif axis == 1:
        tensor = tensor.permute((1, 0, 2))
        tensor = rotate(tensor, interpolation=InterpolationMode.BILINEAR,
                        angle=theta, expand=expand, fill=fill)
        tensor = tensor.permute((1, 0, 2))
    elif axis == 2:
        tensor = tensor.permute((2, 1, 0))
        tensor = rotate(tensor, interpolation=InterpolationMode.BILINEAR,
                        angle=-theta, expand=expand, fill=fill)
        tensor = tensor.permute((2, 1, 0))
    else:
        raise Exception('Not invalid axis')
    return tensor


def projection_loss_s(preds: torch.Tensor, targets: torch.Tensor, proj_direct: int, tgt_channels: List[int] = [0, 1, 2]) -> torch.Tensor:
    B = preds.size(0)
    num_tgt_channels = len(tgt_channels)

    if preds.size(1) == 1:
        preds = torch.sigmoid(preds)
    else:
        preds = torch.softmax(preds, dim=1)
    preds = torch.clamp(preds, min=1e-7, max=1-1e-7).to(torch.float32)

    if targets.dtype == torch.bool:
        targets = targets.to(torch.float32)

    loss = 0.

    for i in range(B):
        for channel in tgt_channels:
            pred_proj = torch.sum(preds[i, channel], axis=proj_direct)
            target_proj = torch.sum(targets[i, channel], axis=proj_direct)
            loss += F.mse_loss(pred_proj, target_proj)

    return loss / (B * num_tgt_channels)


def projection_loss_rotate_s(preds: torch.Tensor, targets: torch.Tensor, tgt_channels: List[int] = [0],
                             proj_direct: int = 0, rtt_dir: int = 1, theta: int = 30) -> torch.Tensor:
    B = preds.size(0)
    num_tgt_channels = len(tgt_channels)

    if preds.size(1) == 1:
        preds = torch.sigmoid(preds)
    else:
        preds = torch.softmax(preds, dim=1)
    preds = torch.clamp(preds, min=1e-7, max=1-1e-7).to(torch.float32)

    if targets.dtype == torch.bool:
        targets = targets.to(torch.float32)

    loss = 0.

    for i in range(B):
        for channel in tgt_channels:
            assert proj_direct != rtt_dir
            pred_rtt = rotate3D(preds[i, channel], rtt_dir, theta)
            target_rtt = rotate3D(targets[i, channel], rtt_dir, theta)

            pred_proj = torch.sum(pred_rtt, axis=proj_direct)
            target_proj = torch.sum(target_rtt, axis=proj_direct)
            loss += F.mse_loss(pred_proj, target_proj)

    return loss / (B * num_tgt_channels)


def projection_loss_rotate_m(preds: torch.Tensor, targets: torch.Tensor, tgt_channels: List[int] = [0],
                             proj_dirs: List[int] = [0], rtt_dirs: List[int] = [1], thetas: List[int] = [0, 30, 60]) -> Tuple[List[float], torch.Tensor]:
    B = preds.size(0)
    num_tgt_channels = len(tgt_channels)

    if preds.size(1) == 1:
        preds = torch.sigmoid(preds)
    else:
        preds = torch.softmax(preds, dim=1)
    preds = torch.clamp(preds, min=1e-7, max=1).to(torch.float32)

    if targets.dtype == torch.bool:
        targets = targets.to(torch.float32)

    loss = 0.
    num_loss = 0
    for proj_idx, _ in enumerate(proj_dirs):
        num_loss += len(rtt_dirs[proj_idx]) * len(thetas[proj_idx])

    loss_list = [0. for _ in range(num_loss)]

    start_idx = 0
    for i in range(B):
        for channel in tgt_channels:
            pred = preds[i, channel]
            target = targets[i, channel]
            for proj_idx, proj_dir in enumerate(proj_dirs):
                for rtt_idx, rtt_dir in enumerate(rtt_dirs[proj_idx]):
                    assert proj_dir != rtt_dir, "projection axis should not equal to rotate axis"
                    num_rtt_dirs = len(rtt_dirs[proj_idx])
                    num_thetas = len(thetas[proj_idx])
                    for theta_idx, theta in enumerate(thetas[proj_idx]):
                        pred_rtt = rotate3D(pred, rtt_dir, theta)
                        target_rtt = rotate3D(target, rtt_dir, theta)

                        pred_proj = torch.sum(pred_rtt, axis=proj_dir)
                        target_proj = torch.sum(target_rtt, axis=proj_dir)
                        _loss = F.mse_loss(pred_proj, target_proj)

                        loss_list[start_idx + rtt_idx *
                                  num_thetas + theta_idx] += _loss
                    start_idx += num_rtt_dirs * num_thetas

    loss_list = [l.item() / (B * num_tgt_channels) for l in loss_list]
    return loss_list, loss / (B * num_tgt_channels)


def projection_loss_rotate_dym(preds: torch.Tensor, targets: torch.Tensor, epoch: int, avg_cost: np.ndarray, tgt_channels: List[int] = [0],
                               proj_dirs: List[int] = [0], rtt_dirs: List[List[int]] = [[1, 2]], thetas: List[List[int]] = [[0, 30, 60]]) -> Tuple[List[float], torch.Tensor]:
    B = preds.size(0)
    num_tgt_channels = len(tgt_channels)

    if preds.size(1) == 1:
        preds = torch.sigmoid(preds)
    else:
        preds = torch.softmax(preds, dim=1)
    preds = torch.clamp(preds, min=1e-7, max=1).to(torch.float32)

    if targets.dtype == torch.bool:
        targets = targets.to(torch.float32)

    loss = 0.
    num_loss = 0
    for proj_idx, _ in enumerate(proj_dirs):
        num_loss += len(rtt_dirs[proj_idx]) * len(thetas[proj_idx])

    loss_list = [0. for _ in range(num_loss)]

    start_idx = 0
    for i in range(B):
        for channel in tgt_channels:
            pred = preds[i, channel]
            target = targets[i, channel]
            for proj_idx, proj_dir in enumerate(proj_dirs):
                for rtt_idx, rtt_dir in enumerate(rtt_dirs[proj_idx]):
                    assert proj_dir != rtt_dir, "projection axis should not equal to rotate axis"
                    num_rtt_dirs = len(rtt_dirs[proj_idx])
                    num_thetas = len(thetas[proj_idx])
                    for theta_idx, theta in enumerate(thetas[proj_idx]):
                        pred_rtt = rotate3D(pred, rtt_dir, theta)
                        target_rtt = rotate3D(target, rtt_dir, theta)

                        pred_proj = torch.sum(pred_rtt, axis=proj_dir)
                        target_proj = torch.sum(target_rtt, axis=proj_dir)
                        _loss = F.mse_loss(pred_proj, target_proj)

                        loss_list[start_idx + rtt_idx *
                                  num_thetas + theta_idx] += _loss
                    start_idx += num_rtt_dirs * num_thetas

    loss_list = [l / (B * num_tgt_channels) for l in loss_list]

    lambda_weight = [1. for _ in range(num_loss)]
    T = 2.0

    if epoch > 1:
        tmp_weight = [0. for _ in range(num_loss)]
        loss_sum = 0.
        for i in range(num_loss):
            tmp_weight[i] = np.exp(
                avg_cost[epoch - 1, i] / avg_cost[epoch - 2, i] / T)
            loss_sum += tmp_weight[i]
        for i in range(num_loss):
            lambda_weight[i] = num_loss * tmp_weight[i] / loss_sum

    for i in range(num_loss):
        loss += lambda_weight[i] * loss_list[i]

    loss_list = [_loss.item() for _loss in loss_list]
    return loss_list, loss
