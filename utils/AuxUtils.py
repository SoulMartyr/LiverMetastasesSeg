import os
import math
import yaml
import numpy as np
import pandas as pd
import SimpleITK as sitk
from typing import List, Tuple, Sequence, Union, Dict, Any
import torch
from torch import optim, nn
from torch.nn import functional as F

from .DataUtils import resize_dhw_numpy


# Mainly for training


def get_index(index_path: str, fold: List[int]) -> List[str]:
    index_df = pd.read_csv(index_path, index_col=0)
    index = []
    for f in fold:
        index.extend(index_df.loc[f, "index"].strip().split(" "))
    return index


def get_learning_rate(optimizer: optim.Optimizer) -> float:
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])

    assert len(lr) == 1, "Get Learning Rate More Than 1"
    lr = lr[0]

    return lr


def get_args(log_fold_dir: str) -> Dict[str, Any]:
    args_file_path = os.path.join(log_fold_dir, "exp_arg.yaml")
    with open(args_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
        args_dict = yaml.load(data, Loader=yaml.FullLoader)
    return args_dict


def get_ckpt_path(log_fold_dir: str) -> Dict[str, Any]:
    ckpt_path = os.path.join(log_fold_dir, "model.pth")
    if not os.path.exists(ckpt_path):
        raise RuntimeError("no checkpoint")
    return ckpt_path


def save_weight(ckpt_dir: str, epoch: int, model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler) -> None:
    if isinstance(model, nn.DataParallel):
        model = model.module

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "sched_state_dict": scheduler.state_dict()
    }
    torch.save(checkpoint, ckpt_dir + '/model.pth')


class AvgOutput(object):
    def __init__(self, length: int = 0) -> None:
        self.length = length
        if length == 0:
            self.sum = 0.
        else:
            self.sum = [0 for _ in range(length)]
        self.count = 0

    def add(self, x: Union[float, List[float]]) -> None:
        if self.length == 0:
            self.sum += x
        else:
            self.sum = [x + y for x, y in zip(self.sum, x)]
        self.count += 1

    def avg(self) -> float:
        if self.length == 0:
            return self.sum / self.count
        else:
            return [x / self.count for x in self.sum]

    def clear(self) -> None:
        if self.length == 0:
            self.sum = 0.
        else:
            self.sum = [0 for _ in range(self.length)]
        self.count = 0


# Mainly for predicting


def set_pred_dir(pred_dir: str, file_dir: str, fold: int) -> str:
    pred_dir = os.path.join(pred_dir, file_dir, "fold{}".format(fold))
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    return pred_dir


def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


def sliding_window_inference_2d(inputs: torch.Tensor, crop_size: Tuple[int],  model: nn.Module, outputs_size: torch.Size, is_softmax: bool) -> torch.Tensor:
    num_spatial_dims = len(inputs.shape) - 2

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape

    assert len(crop_size) == len(image_size_)
    image_size = tuple(max(image_size_[i], crop_size[i])
                       for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(crop_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode="constant", value=0.)

    scan_interval = _get_scan_interval(
        image_size, crop_size, num_spatial_dims, 0)

    scan_num = [math.ceil((image_size[i] - crop_size[i]) /
                          scan_interval[i]) + 1 for i in range(num_spatial_dims)]
    starts = []
    for dim in range(num_spatial_dims):
        dim_starts = []
        for idx in range(scan_num[dim]):
            start_idx = idx * scan_interval[dim]
            start_idx -= max(start_idx + crop_size[dim] - image_size[dim], 0)
            dim_starts.append(start_idx)
        starts.append(dim_starts)
    out = np.asarray([x.flatten()
                     for x in np.meshgrid(*starts, indexing="ij")]).T
    slices = [[slice(None), slice(None)]+[slice(s, s + crop_size[d])
                                          for d, s in enumerate(x)] for x in out]
    outputs = torch.zeros(outputs_size).to(inputs.device)
    count = torch.zeros(outputs_size).to(inputs.device)

    for slice_s in slices:
        input_s = inputs[slice_s].squeeze(0).permute(1, 0, 2, 3)
        part_out = model(input_s)
        part_out = part_out.permute(1, 0, 2, 3).unsqueeze(0)
        if is_softmax:
            part_out = F.softmax(part_out, dim=1)
        else:
            part_out = torch.sigmoid(part_out)
        outputs[slice_s] += part_out.clone()
        count[slice_s] += 1

    return outputs / count


def sliding_window_inference_3d(inputs: torch.Tensor, crop_size: Tuple[int],  model: nn.Module, outputs_size: torch.Size, is_softmax: bool, overlap: float = 0.5) -> torch.Tensor:
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape

    assert len(crop_size) == len(image_size_)
    image_size = tuple(max(image_size_[i], crop_size[i])
                       for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(crop_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode="constant", value=0.)

    scan_interval = _get_scan_interval(
        image_size, crop_size, num_spatial_dims, overlap)

    scan_num = [math.ceil((image_size[i] - crop_size[i]) /
                          scan_interval[i]) + 1 for i in range(num_spatial_dims)]
    starts = []
    for dim in range(num_spatial_dims):
        dim_starts = []
        for idx in range(scan_num[dim]):
            start_idx = idx * scan_interval[dim]
            start_idx -= max(start_idx + crop_size[dim] - image_size[dim], 0)
            dim_starts.append(start_idx)
        starts.append(dim_starts)
    out = np.asarray([x.flatten()
                     for x in np.meshgrid(*starts, indexing="ij")]).T
    slices = [[slice(None), slice(None)]+[slice(s, s + crop_size[d])
                                          for d, s in enumerate(x)] for x in out]
    outputs = torch.zeros(outputs_size).to(inputs.device)
    count = torch.zeros(outputs_size).to(inputs.device)

    for slice_s in slices:
        part_out = model(inputs[slice_s])
        if is_softmax:
            part_out = F.softmax(part_out, dim=1)
        else:
            part_out = torch.sigmoid(part_out)
        outputs[slice_s] += part_out.clone()
        count[slice_s] += 1

    return outputs / count


def predict_merge_channel_torch(tensor: torch.Tensor, is_softmax: bool) -> torch.Tensor:
    """
    tensor shape: [C, D, H, W]
    """
    if is_softmax:
        output = torch.argmax(tensor, dim=0)
    else:
        C, D, H, W = tensor.size()
        output = torch.zeros([D, H, W]).to(tensor.device)
        for c in range(C):
            output[tensor[c] == 1] = c+1
    return output


def save_predict_mask(mask_array: np.ndarray, img_path: str, mask_path: str) -> None:
    img_itk = sitk.ReadImage(img_path)
    img_size = sitk.GetArrayFromImage(img_itk).shape

    mask_array = resize_dhw_numpy(mask_array, order=0, dhw=img_size)

    mask_itk = sitk.GetImageFromArray(mask_array.astype(np.uint8))
    mask_itk.CopyInformation(img_itk)

    sitk.WriteImage(mask_itk, mask_path)


if __name__ == "__main__":
    inputs = torch.ones((1, 3, 54, 300, 320))
    outputs = torch.zeros(((1, 4, 54, 300, 320)))
    roi = (32, 224, 224)
    print(sliding_window_inference_3d(
        inputs, roi, 0.5, None, outputs.size()).shape)
