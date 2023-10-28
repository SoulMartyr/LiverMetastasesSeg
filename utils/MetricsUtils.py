import numpy as np
import torch
from scipy.ndimage import morphology, zoom
from typing import List, Tuple

from .LossUtils import sigmoid_binary_torch, softmax_binary_torch


class Metrics:
    def __init__(self, metrics_types: List[str] = ["dice", "voe", "rvd", "asd"]) -> None:
        super()
        self.metrics_types = metrics_types
        self.metrics_dict = {"dice": dice_torch,
                             "voe": voe_torch,
                             "rvd": rvd_torch,
                             "asd": asd_torch,
                             "msd": msd_torch,
                             "rmsd": rmsd_torch}

    def compute_metrics(self, pred: torch.tensor, gt: torch.tensor, tgt_channel: int, spacing: Tuple[float]) -> str:
        metrics_info = ""
        metrics_result = []

        for metrics_type in self.metrics_types:
            if metrics_type not in self.metrics_dict:
                raise ValueError("error metrics_type:{}".format(metrics_type))
            if metrics_type.find("sd") != -1:
                metric = self.metrics_dict[metrics_type](
                    pred, gt, tgt_channel, spacing)
            else:
                metric = self.metrics_dict[metrics_type](
                    pred, gt, tgt_channel)
            metrics_info += "{}: {:.4f} ".format(metrics_type.upper(), metric)
            metrics_result.append(metric)

        return metrics_info, metrics_result


def binary_torch(preds: torch.Tensor, is_softmax: bool = False, threshold: List[float] = [0.5]) -> torch.Tensor:
    if is_softmax:
        preds = softmax_binary_torch(preds)
    else:
        preds = sigmoid_binary_torch(preds, threshold)
    return preds


def dice_torch(preds: torch.Tensor, gts: torch.Tensor, tgt_channel: int) -> float:
    assert preds.dim() == 4 and gts.dim(
    ) == 4, "preds and gts shape should be [C,D,H,W]"
    assert torch.unique(preds).numel() <= 2 and torch.unique(
        gts).numel() <= 2, "preds and gts unique number should be less than 2"

    eps = 1e-5

    pred = preds[tgt_channel].flatten()
    gt = gts[tgt_channel].flatten()

    intersection = (pred * gt).sum()
    union = (pred + gt).sum()

    dice = ((2 * intersection + eps) / (union + eps)).item()
    return dice


def voe_torch(preds: torch.Tensor, gts: torch.Tensor, tgt_channel: int) -> float:
    assert preds.dim() == 4 and gts.dim(
    ) == 4, "preds and gts shape should be [C,D,H,W]"
    assert torch.unique(preds).numel() <= 2 and torch.unique(
        gts).numel() <= 2, "preds and gts unique number should be less than 2"

    eps = 1e-5

    pred = preds[tgt_channel].flatten()
    gt = gts[tgt_channel].flatten()

    intersection = (pred * gt).sum()
    union = (pred + gt).sum()

    voe = 1 - ((intersection + eps) / (union - intersection + eps)).item()
    return voe


def rvd_torch(preds: torch.Tensor, gts: torch.Tensor, tgt_channel: int) -> float:
    assert preds.dim() == 4 and gts.dim(
    ) == 4, "preds and gts shape should be [C,D,H,W]"
    assert torch.unique(preds).numel() <= 2 and torch.unique(
        gts).numel() <= 2, "preds and gts unique number should be less than 2"

    eps = 1e-5

    pred = preds[tgt_channel].flatten()
    gt = gts[tgt_channel].flatten()

    volume_pred = pred.sum()
    volume_gt = gt.sum()

    rvd = (torch.abs(volume_pred - volume_gt + eps) /
           (volume_gt + eps)).item()
    return rvd


def compute_bounding_box(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    num_dims = len(mask.shape)
    bbox_min = np.zeros(num_dims, np.int64)
    bbox_max = np.zeros(num_dims, np.int64)

    proj_0 = np.amax(mask, axis=tuple(range(num_dims))[1:])
    idx_nonzero_0 = np.nonzero(proj_0)[0]
    if len(idx_nonzero_0) == 0:
        return None, None

    bbox_min[0] = np.min(idx_nonzero_0)
    bbox_max[0] = np.max(idx_nonzero_0)

    for axis in range(1, num_dims):
        max_over_axes = list(range(num_dims))
        max_over_axes.pop(axis)
        max_over_axes = tuple(max_over_axes)
        proj = np.amax(mask, axis=max_over_axes)
        idx_nonzero = np.nonzero(proj)[0]
        bbox_min[axis] = np.min(idx_nonzero)
        bbox_max[axis] = np.max(idx_nonzero)

    return bbox_min, bbox_max


def resize_mask_spacing_numpy(array: np.ndarray, spacing: Tuple[float]) -> np.ndarray:
    if spacing != (1., 1., 1.):
        array = zoom(
            array, (spacing[2], spacing[1], spacing[0]), order=0
        )

    return array


def crop_to_bounding_box(mask: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    slices = tuple([slice(s, e+1)
                    for s, e in zip(bbox_min.tolist(), bbox_max.tolist())])
    cropmask = mask[slices]
    return cropmask


def get_mask_edges(pred: np.ndarray, gt: np.ndarray, crop: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if crop:
        if not np.any(pred | gt):
            return np.zeros_like(pred), np.zeros_like(gt)

        bbox_min, bbox_max = compute_bounding_box(pred | gt)
        pred = crop_to_bounding_box(pred, bbox_min, bbox_max)
        gt = crop_to_bounding_box(gt, bbox_min, bbox_max)

    edges_pred = morphology.binary_erosion(pred) ^ pred
    edges_gt = morphology.binary_erosion(gt) ^ gt

    return edges_pred, edges_gt


def get_surface_distance(src: np.ndarray, dest: np.ndarray, distance_metric: str = "euclidean") -> np.ndarray:
    if not np.any(dest):
        dist = np.inf * np.ones_like(dest)
    else:
        if not np.any(src):
            dist = np.inf * np.ones_like(dest)
            return np.asarray(dist[dest])
        if distance_metric == "euclidean":
            dist = morphology.distance_transform_edt(~dest)
        elif distance_metric in {"chessboard", "taxicab"}:
            dist = morphology.distance_transform_cdt(
                ~dest, metric=distance_metric)
        else:
            raise ValueError(
                f"distance_metric {distance_metric} is not implemented.")

    return np.asarray(dist[src])


def asd_torch(preds: torch.Tensor, gts: torch.Tensor, tgt_channel: int, spacing: Tuple[float], distance_metric: str = "euclidean") -> float:
    assert preds.dim() == 4 and gts.dim(
    ) == 4, "preds and gts shape should be [C,D,H,W]"
    assert torch.unique(preds).numel() <= 2 and torch.unique(
        gts).numel() <= 2, "preds and gts unique number should be less than 2"

    pred = preds[tgt_channel].detach().cpu().numpy()
    gt = gts[tgt_channel].detach().cpu().numpy()

    pred = resize_mask_spacing_numpy(pred, spacing)
    gt = resize_mask_spacing_numpy(gt, spacing)

    edges_pred, edges_gt = get_mask_edges(pred, gt)

    surface_distance_pred2gt = get_surface_distance(
        edges_pred, edges_gt, distance_metric=distance_metric)
    surface_distance_gt2pred = get_surface_distance(
        edges_gt, edges_pred, distance_metric=distance_metric)
    surface_distance = np.concatenate(
        [surface_distance_pred2gt, surface_distance_gt2pred])

    if surface_distance.shape == (0,):
        return np.nan
    else:
        return surface_distance.mean()


def msd_torch(preds: torch.Tensor, gts: torch.Tensor, tgt_channel: int, spacing: Tuple[float], distance_metric: str = "euclidean") -> float:
    assert preds.dim() == 4 and gts.dim(
    ) == 4, "preds and gts shape should be [C,D,H,W]"
    assert torch.unique(preds).numel() <= 2 and torch.unique(
        gts).numel() <= 2, "preds and gts unique number should be less than 2"

    pred = preds[tgt_channel].detach().cpu().numpy()
    gt = gts[tgt_channel].detach().cpu().numpy()

    pred = resize_mask_spacing_numpy(pred, spacing)
    gt = resize_mask_spacing_numpy(gt, spacing)

    edges_pred, edges_gt = get_mask_edges(pred, gt)
    surface_distance_pred2gt = get_surface_distance(
        edges_pred, edges_gt, distance_metric=distance_metric)
    surface_distance_gt2pred = get_surface_distance(
        edges_gt, edges_pred, distance_metric=distance_metric)

    if surface_distance_pred2gt.shape == (0,) or surface_distance_gt2pred.shape == (0,):
        return np.nan
    else:
        return max(surface_distance_pred2gt.sum(), surface_distance_gt2pred.sum())


def rmsd_torch(preds: torch.Tensor, gts: torch.Tensor, tgt_channel: int, spacing: Tuple[float], distance_metric: str = "euclidean") -> float:
    assert preds.dim() == 4 and gts.dim(
    ) == 4, "preds and gts shape should be [C,D,H,W]"
    assert torch.unique(preds).numel() <= 2 and torch.unique(
        gts).numel() <= 2, "preds and gts unique number should be less than 2"

    pred = preds[tgt_channel].detach().cpu().numpy()
    gt = gts[tgt_channel].detach().cpu().numpy()

    pred = resize_mask_spacing_numpy(pred, spacing)
    gt = resize_mask_spacing_numpy(gt, spacing)

    edges_pred, edges_gt = get_mask_edges(pred, gt)
    surface_distance_pred2gt = get_surface_distance(
        edges_pred, edges_gt, distance_metric=distance_metric)
    surface_distance_gt2pred = get_surface_distance(
        edges_gt, edges_pred, distance_metric=distance_metric)
    surface_distance = np.concatenate(
        [surface_distance_pred2gt * surface_distance_pred2gt, surface_distance_gt2pred * surface_distance_gt2pred])

    if surface_distance_pred2gt.shape == (0,) or surface_distance_gt2pred.shape == (0,):
        return np.nan
    else:
        return np.sqrt(surface_distance.mean())


if __name__ == "__main__":
    A = torch.tensor(
        [[[
            [0, 0, 1, 1, 1],
            [1, 0, 0, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]]]
    )
    B = torch.tensor(
        [[[
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]]],
    )
    a = get_surface_distance(A[0, 0].numpy(), B[0, 0].numpy())
