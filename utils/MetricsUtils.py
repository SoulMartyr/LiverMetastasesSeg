import numpy as np
import torch
from scipy.ndimage import morphology, zoom
from typing import List, Tuple, Union, Callable

from .LossUtils import sigmoid_binary_torch, softmax_binary_torch
from .AuxUtils import AvgOutput


class Metrics:
    def __init__(self, num_samples, num_classes, is_softmax, thres, metrics_types: List[str] = ["dice_per_case", "dice_global", "voe", "rvd", "asd"]) -> None:
        super()
        self.metrics_func_dict = {"dice_per_case": dice_per_case_torch,
                                  "voe_per_case": voe_per_case_torch,
                                  "rvd_per_case": rvd_per_case_torch,
                                  "dice_global": dice_global_torch,
                                  "voe_global": voe_global_torch,
                                  "rvd_global": rvd_global_torch,
                                  "asd": asd_torch,
                                  "msd": msd_torch,
                                  "rmsd": rmsd_torch}

        self.per_case_metrics_type = []
        self.global_metrics_type = []

        for metrics_type in metrics_types:
            if metrics_type not in self.metrics_func_dict:
                raise ValueError("error metrics type:{}".format(metrics_type))

            if metrics_type.find("global") == -1:
                self.per_case_metrics_type.append(metrics_type)
            else:
                self.global_metrics_type.append(metrics_type)

        self.channel_range = [i if not is_softmax else i +
                              1 for i in range(0, num_classes)]
        self.per_case_metric = [AvgOutput(length=len(self.per_case_metrics_type))
                                for _ in range(0, num_classes)]
        self.is_softmax = is_softmax
        self.thres = thres
        self.num_samples = num_samples
        self.num_classes = num_classes

        self.preds = []
        self.gts = []

    def __convert_pred_and_gt(self, pred: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = binary_torch(pred, self.is_softmax, self.thres)
        pred, gt = pred.squeeze(0), gt.squeeze(0)
        return pred, gt

    def compute_per_case_metrics(self, pred: torch.Tensor, gt: torch.Tensor, spacing: Tuple[float] = (1., 1., 1.)) -> List[List[float]]:
        pred, gt = self.__convert_pred_and_gt(pred, gt)

        self.preds.append(pred)
        self.gts.append(gt)

        metrics_result = []

        for i, channel in enumerate(self.channel_range):
            channel_metric_result = []

            for metric_type in self.per_case_metrics_type:
                metrics_func = self.metrics_func_dict[metric_type]

                if metric_type.find('sd') != -1:
                    metric = metrics_func(pred, gt, channel, spacing)
                else:
                    metric = metrics_func(pred, gt, channel)

                channel_metric_result.append(metric)

            self.per_case_metric[i].add(channel_metric_result)
            metrics_result.append(channel_metric_result)

        return metrics_result

    def info_per_case_metrics(self, index, pred: torch.Tensor, gt: torch.Tensor, spacing: Tuple[float] = (1., 1., 1.), info_func: Callable = print) -> None:
        metric_result = self.compute_per_case_metrics(pred, gt, spacing)

        metrics_info = "Index:{} ".format(index)

        for i in range(self.num_classes):
            metrics_info += '-'

            if not self.is_softmax:
                metrics_info += "Thres:{} ".format(self.thres[i])

            for idx, metric_type in enumerate(self.per_case_metrics_type):
                metric_type = metric_type.split('_')[0]
                metrics_info += "{}{}: {:.4f} ".format(
                    metric_type.upper(), i, metric_result[i][idx])

        info_func(metrics_info)

    def compute_mean_per_case_metrics(self) -> List[List[float]]:
        mean_metric_result = []

        for mean_metric in self.per_case_metric:
            mean_metric_result.append(mean_metric.avg())

        return mean_metric_result

    def info_mean_per_case_metrics(self, info_func: Callable = print) -> None:
        mean_metric_result = self.compute_mean_per_case_metrics()

        metrics_info = "Mean: "

        for i in range(self.num_classes):
            metrics_info += '-'

            for idx, metric_type in enumerate(self.per_case_metrics_type):
                metric_type = metric_type.split('_')[0]
                metrics_info += "{}{}: {:.4f} ".format(
                    metric_type.upper(), i, mean_metric_result[i][idx])

        info_func(metrics_info)

    def compute_global_metrics(self) -> List[List[float]]:
        assert len(self.preds) == len(
            self.gts) == self.num_samples, "samples num error"

        global_metric_result = []

        for channel in self.channel_range:
            channel_global_metric_result = []

            for metric_type in self.global_metrics_type:
                metrics_func = self.metrics_func_dict[metric_type]
                channel_global_metric_result.append(
                    metrics_func(self.preds, self.gts, channel))

            global_metric_result.append(channel_global_metric_result)

        return global_metric_result

    def info_global_metrics(self, info_func: Callable = print) -> None:
        global_metric_result = self.compute_global_metrics()

        metrics_info = "Global: "

        for i in range(self.num_classes):
            metrics_info += '-'

            for idx, metric_type in enumerate(self.global_metrics_type):
                metric_type = metric_type.split('_')[0]
                metrics_info += "{}{}: {:.4f} ".format(
                    metric_type.upper(), i, global_metric_result[i][idx])
        info_func(metrics_info)


def binary_torch(pred: torch.Tensor, is_softmax: bool = False, threshold: List[float] = [0.5]) -> torch.Tensor:
    if is_softmax:
        pred = softmax_binary_torch(pred)
    else:
        pred = sigmoid_binary_torch(pred, threshold)
    return pred


def dice_per_case_torch(pred: torch.Tensor, gt: torch.Tensor, tgt_channel: int) -> float:
    assert pred.dim() == 4 and gt.dim(
    ) == 4, "pred and gt shape should be [C,D,H,W]"
    assert torch.unique(pred).numel() <= 2 and torch.unique(
        gt).numel() <= 2, "pred and gt unique number should be less than 2"

    eps = 1e-5

    pred = pred[tgt_channel].flatten()
    gt = gt[tgt_channel].flatten()

    intersection = (pred * gt).sum()
    union = (pred + gt).sum()

    dice = ((2 * intersection + eps) / (union + eps)).item()
    return dice


def dice_global_torch(preds: List[torch.Tensor], gts: List[torch.Tensor], tgt_channel: int) -> float:
    assert isinstance(preds, list) and isinstance(
        gts, list), "preds and gts should be list of tensor"
    assert preds[0].dim() == 4 and gts[0].dim(
    ) == 4, "pred and gt shape should be [C,D,H,W]"
    assert torch.unique(preds[0]).numel() <= 2 and torch.unique(
        gts[0]).numel() <= 2, "pred and gt unique number should be less than 2"

    preds = [pred[tgt_channel].flatten() for pred in preds]
    gts = [gt[tgt_channel].flatten() for gt in gts]

    eps = 1e-5

    intersections = [(pred * gt).sum() for pred, gt in zip(preds, gts)]
    unions = [(pred + gt).sum() for pred, gt in zip(preds, gts)]

    intersection = sum(intersections)
    union = sum(unions)

    dice = ((2 * intersection + eps) / (union + eps)).item()
    return dice


def voe_per_case_torch(pred: torch.Tensor, gt: torch.Tensor, tgt_channel: int) -> float:
    assert pred.dim() == 4 and gt.dim(
    ) == 4, "pred and gt shape should be [C,D,H,W]"
    assert torch.unique(pred).numel() <= 2 and torch.unique(
        gt).numel() <= 2, "pred and gt unique number should be less than 2"

    eps = 1e-5

    pred = pred[tgt_channel].flatten()
    gt = gt[tgt_channel].flatten()

    intersection = (pred * gt).sum()
    union = (pred + gt).sum()

    voe = 1 - ((intersection + eps) / (union - intersection + eps)).item()
    return voe


def voe_global_torch(preds: List[torch.Tensor], gts: List[torch.Tensor], tgt_channel: int) -> float:
    assert isinstance(preds, list) and isinstance(
        gts, list), "preds and gts should be list of tensor"
    assert preds[0].dim() == 4 and gts[0].dim(
    ) == 4, "pred and gt shape should be [C,D,H,W]"
    assert torch.unique(preds[0]).numel() <= 2 and torch.unique(
        gts[0]).numel() <= 2, "pred and gt unique number should be less than 2"

    preds = [pred[tgt_channel].flatten() for pred in preds]
    gts = [gt[tgt_channel].flatten() for gt in gts]

    eps = 1e-5

    intersections = [(pred * gt).sum() for pred, gt in zip(preds, gts)]
    unions = [(pred + gt).sum() for pred, gt in zip(preds, gts)]

    intersection = sum(intersections)
    union = sum(unions)

    voe = 1 - ((intersection + eps) / (union - intersection + eps)).item()
    return voe


def rvd_per_case_torch(pred: torch.Tensor, gt: torch.Tensor, tgt_channel: int) -> float:
    assert pred.dim() == 4 and gt.dim(
    ) == 4, "pred and gt shape should be [C,D,H,W]"
    assert torch.unique(pred).numel() <= 2 and torch.unique(
        gt).numel() <= 2, "pred and gt unique number should be less than 2"

    eps = 1e-5

    pred = pred[tgt_channel].flatten()
    gt = gt[tgt_channel].flatten()

    volume_pred = pred.sum()
    volume_gt = gt.sum()

    rvd = (torch.abs(volume_pred - volume_gt + eps) /
           (volume_gt + eps)).item()
    return rvd


def rvd_global_torch(preds: List[torch.Tensor], gts: List[torch.Tensor], tgt_channel: int) -> float:
    assert isinstance(preds, list) and isinstance(
        gts, list), "preds and gts should be list of tensor"
    assert preds[0].dim() == 4 and gts[0].dim(
    ) == 4, "pred and gt shape should be [C,D,H,W]"
    assert torch.unique(preds[0]).numel() <= 2 and torch.unique(
        gts[0]).numel() <= 2, "pred and gt unique number should be less than 2"

    preds = [pred[tgt_channel].flatten() for pred in preds]
    gts = [gt[tgt_channel].flatten() for gt in gts]

    eps = 1e-5

    volume_pred = sum(preds)
    volume_gt = sum(gts)

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


def get_surface_distance(src: np.ndarray, dest: np.ndarray, sampling: Union[float, List[float]], distance_metric: str = "euclidean") -> np.ndarray:
    if not np.any(dest):
        dest = resize_mask_spacing_numpy(dest, sampling)
        dist = np.inf * np.ones_like(dest)
    else:
        if not np.any(src):
            dest = resize_mask_spacing_numpy(dest, sampling)
            dist = np.inf * np.ones_like(dest)
            return np.asarray(dist[dest])
        if distance_metric == "euclidean":
            dist = morphology.distance_transform_edt(~dest, sampling=sampling)
        elif distance_metric in {"chessboard", "taxicab"}:
            dist = morphology.distance_transform_cdt(
                ~dest, sampling=sampling, metric=distance_metric)
        else:
            raise ValueError(
                f"distance_metric {distance_metric} is not implemented.")

    return np.asarray(dist[src])


def asd_torch(pred: torch.Tensor, gt: torch.Tensor, tgt_channel: int, spacing: Tuple[float], distance_metric: str = "euclidean") -> float:
    assert pred.dim() == 4 and gt.dim(
    ) == 4, "pred and gt shape should be [C,D,H,W]"
    assert torch.unique(pred).numel() <= 2 and torch.unique(
        gt).numel() <= 2, "pred and gt unique number should be less than 2"

    pred = pred[tgt_channel].detach().cpu().numpy()
    gt = gt[tgt_channel].detach().cpu().numpy()

    edges_pred, edges_gt = get_mask_edges(pred, gt)

    surface_distance_pred2gt = get_surface_distance(
        edges_pred, edges_gt, sampling=spacing, distance_metric=distance_metric)
    surface_distance_gt2pred = get_surface_distance(
        edges_gt, edges_pred, sampling=spacing,  distance_metric=distance_metric)
    surface_distance = np.concatenate(
        [surface_distance_pred2gt, surface_distance_gt2pred])

    if surface_distance.shape == (0,):
        return np.nan
    else:
        return surface_distance.mean()


def msd_torch(pred: torch.Tensor, gt: torch.Tensor, tgt_channel: int, spacing: Tuple[float], distance_metric: str = "euclidean") -> float:
    assert pred.dim() == 4 and gt.dim(
    ) == 4, "pred and gt shape should be [C,D,H,W]"
    assert torch.unique(pred).numel() <= 2 and torch.unique(
        gt).numel() <= 2, "pred and gt unique number should be less than 2"

    pred = pred[tgt_channel].detach().cpu().numpy()
    gt = gt[tgt_channel].detach().cpu().numpy()

    edges_pred, edges_gt = get_mask_edges(pred, gt)

    surface_distance_pred2gt = get_surface_distance(
        edges_pred, edges_gt, sampling=spacing, distance_metric=distance_metric)
    surface_distance_gt2pred = get_surface_distance(
        edges_gt, edges_pred, sampling=spacing, distance_metric=distance_metric)

    if surface_distance_pred2gt.shape == (0,) or surface_distance_gt2pred.shape == (0,):
        return np.nan
    else:
        return max(surface_distance_pred2gt.sum(), surface_distance_gt2pred.sum())


def rmsd_torch(pred: torch.Tensor, gt: torch.Tensor, tgt_channel: int, spacing: Tuple[float], distance_metric: str = "euclidean") -> float:
    assert pred.dim() == 4 and gt.dim(
    ) == 4, "pred and gt shape should be [C,D,H,W]"
    assert torch.unique(pred).numel() <= 2 and torch.unique(
        gt).numel() <= 2, "pred and gt unique number should be less than 2"

    pred = pred[tgt_channel].detach().cpu().numpy()
    gt = gt[tgt_channel].detach().cpu().numpy()

    edges_pred, edges_gt = get_mask_edges(pred, gt)

    surface_distance_pred2gt = get_surface_distance(
        edges_pred, edges_gt, sampling=spacing, distance_metric=distance_metric)
    surface_distance_gt2pred = get_surface_distance(
        edges_gt, edges_pred, sampling=spacing, distance_metric=distance_metric)
    surface_distance = np.concatenate(
        [surface_distance_pred2gt * surface_distance_pred2gt, surface_distance_gt2pred * surface_distance_gt2pred])

    if surface_distance_pred2gt.shape == (0,) or surface_distance_gt2pred.shape == (0,):
        return np.nan
    else:
        return np.sqrt(surface_distance.mean())


def hd95_torch(pred: torch.Tensor, gt: torch.Tensor, tgt_channel: int, spacing: Tuple[float], distance_metric: str = "euclidean") -> float:
    assert pred.dim() == 4 and gt.dim(
    ) == 4, "pred and gt shape should be [C,D,H,W]"
    assert torch.unique(pred).numel() <= 2 and torch.unique(
        gt).numel() <= 2, "pred and gt unique number should be less than 2"

    pred = pred[tgt_channel].detach().cpu().numpy()
    gt = gt[tgt_channel].detach().cpu().numpy()

    edges_pred, edges_gt = get_mask_edges(pred, gt)

    surface_distance_pred2gt = get_surface_distance(
        edges_pred, edges_gt, sampling=spacing, distance_metric=distance_metric)
    surface_distance_gt2pred = get_surface_distance(
        edges_gt, edges_pred, sampling=spacing, distance_metric=distance_metric)

    hd95 = np.percentile(
        np.hstack((surface_distance_pred2gt, surface_distance_gt2pred)), 95)
    return hd95


if __name__ == "__main__":
    A = torch.tensor(
        [[[
            [0, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
        ]]]
    )
    B = torch.tensor(
        [[[
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]]],
    )
    a = hd95_torch(A, B, 0, (1, 1, 1))
    print(a)
