import os
import random
import torch.nn as nn
import SimpleITK as sitk
import numpy as np
import collections
import torch
from typing import Tuple, Dict, Any
from scipy import ndimage
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


def tuple_sub(tuple1: Tuple[Any], tuple2: Tuple[Any]) -> Tuple[Any]:
    result_tuple = tuple(x - y for x, y in zip(tuple1, tuple2))
    return result_tuple


def random_crop_numpy(img_array: np.ndarray, mask_array: np.ndarray, crop_size: Tuple[int], is_keyframe: bool) -> Tuple[np.ndarray, np.ndarray]:
    z, y, x = img_array.shape
    if is_keyframe:
        mask_flag = np.unique(mask_array)
        z_indexes, y_indexes, x_indexes = np.where(mask_array == mask_flag[-1])
        min_z, min_y, min_x, max_z, max_y, max_x = np.min(z_indexes), np.min(y_indexes), np.min(
            x_indexes), np.max(z_indexes) + 1, np.max(y_indexes) + 1, np.max(x_indexes) + 1
        diff_z, diff_y, diff_x = tuple_sub(crop_size, (min_z, min_y, min_x))
        min_z, min_y, min_x = min_z + \
            max(diff_z, 0), min_y + max(diff_y, 0), min_x + max(diff_x, 0)
        max_z, max_y, max_x = min(z, max_z + max(diff_z, 0)), min(y,
                                                                  max_y + max(diff_y, 0)), min(x, max_x + max(diff_x, 0))
    else:
        min_z, min_y, min_x, max_z, max_y, max_x = *crop_size, z, y, x

    crop_top_z, crop_top_y, crop_top_x = random.randint(
        min_z, max_z), random.randint(min_y, max_y), random.randint(min_x, max_x)
    crop_bot_z, crop_bot_y, crop_bot_x = tuple_sub(
        (crop_top_z, crop_top_y, crop_top_x), crop_size)
    return img_array[crop_bot_z:crop_top_z, crop_bot_y:crop_top_y, crop_bot_x:crop_top_x], mask_array[crop_bot_z:crop_top_z, crop_bot_y:crop_top_y, crop_bot_x:crop_top_x]


def change_virtual_3d_numpy(array: np.ndarray) -> np.ndarray:
    v3d_arr = np.zeros((3, *array.shape))

    v3d_arr[0, 1:] = array[:-1].copy()
    v3d_arr[1] = array.copy()
    v3d_arr[2, :-1] = array[1:].copy()

    v3d_arr[0, 0] = array[0].copy()
    v3d_arr[2, -1] = array[-1].copy()
    return v3d_arr


def resize_dhw_numpy(array: np.ndarray, order: int, dhw: tuple) -> np.ndarray:
    dhw = tuple([array.shape[i] if dhw[i] == -1 else dhw[i] for i in range(3)])

    if array.shape != dhw:
        z_down_scale = dhw[0] / array.shape[0]
        y_down_scale = dhw[1] / array.shape[1]
        x_down_scale = dhw[2] / array.shape[2]

        array = ndimage.zoom(
            array, (z_down_scale, y_down_scale, x_down_scale), order=order
        )

    return array


def min_max_norm_3d_numpy(array: np.ndarray) -> np.ndarray:
    arr_min = np.min(array)
    arr_max = np.max(array)

    return (array - arr_min) / (arr_max - arr_min)


def z_score_norm_3d_numpy(array: np.ndarray, nonzero: bool = True) -> np.ndarray:
    if nonzero:
        indexes = array != 0
        arr_mean = np.mean(array[indexes])
        arr_std = np.std(array[indexes])
        array[indexes] = (array[indexes] - arr_mean) / arr_std
    else:
        arr_mean = np.mean(array)
        arr_std = np.std(array)
        array = (array - arr_mean) / arr_std

    return array


def min_max_norm_2d_numpy(array: np.ndarray) -> np.ndarray:
    arr_min = array.min(axis=(1, 2), keepdims=True)
    arr_max = array.max(axis=(1, 2), keepdims=True)

    return np.divide((array - arr_min), (arr_max - arr_min))


def z_score_norm_2d_numpy(array: np.ndarray, nonzero: bool = True) -> np.ndarray:
    if nonzero:
        indexes = array == 0
        arr_mean = array.mean(axis=(1, 2), where=(array != 0), keepdims=True)
        arr_std = array.std(axis=(1, 2), where=(array != 0), keepdims=True)
        array = (array - arr_mean) / arr_std
        array[indexes] = 0
    else:
        arr_mean = array.mean(axis=(1, 2), keepdims=True)
        arr_std = array.std(axis=(1, 2), keepdims=True)
        array = (array - arr_mean) / arr_std

    return array


def ont_hot_mask_numpy(array: np.ndarray, num_classes: int, is_softmax: bool) -> np.ndarray:
    # assert is_softmax or ((not is_softmax) and num_classes == 1)

    if not is_softmax:
        one_hot = [(array >= i).astype(
            np.uint8) for i in range(1, num_classes + 1)]
    else:
        one_hot = [(array == i).astype(
            np.uint8) for i in range(num_classes + 1)]
    return np.stack(one_hot, axis=0)


def augmentation(img_array: np.ndarray, mask_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flip_num = np.random.randint(0, 8)
    if flip_num == 1:
        img_array = np.flipud(img_array)
        mask_array = np.flipud(mask_array)
    elif flip_num == 2:
        img_array = np.fliplr(img_array)
        mask_array = np.fliplr(mask_array)
    elif flip_num == 3:
        img_array = np.rot90(img_array, k=1, axes=(1, 2))
        mask_array = np.rot90(mask_array, k=1, axes=(1, 2))
    elif flip_num == 4:
        img_array = np.rot90(img_array, k=3, axes=(1, 2))
        mask_array = np.rot90(mask_array, k=3, axes=(1, 2))
    elif flip_num == 5:
        img_array = np.fliplr(img_array)
        mask_array = np.fliplr(mask_array)
        img_array = np.rot90(img_array, k=1, axes=(1, 2))
        mask_array = np.rot90(mask_array, k=1, axes=(1, 2))
    elif flip_num == 6:
        img_array = np.fliplr(img_array)
        mask_array = np.fliplr(mask_array)
        img_array = np.rot90(img_array, k=3, axes=(1, 2))
        mask_array = np.rot90(mask_array, k=3, axes=(1, 2))
    elif flip_num == 7:
        img_array = np.flipud(img_array)
        mask_array = np.flipud(mask_array)
        img_array = np.fliplr(img_array)
        mask_array = np.fliplr(mask_array)
    return img_array, mask_array


class Dataset2D(Dataset):
    def __init__(self, data_dir: str,  image_dir: str, mask_dir: str, index_list: list, is_train: bool = True, num_classes: int = 1, crop_size: Tuple[int] = (32, 224, 224),
                 norm: str = "zscore", dhw: Tuple[int] = (-1, 224, 224), is_keyframe: bool = True, is_softmax: bool = False, is_v3d: bool = False, is_flip: bool = False) -> None:
        super(Dataset2D, self).__init__()
        assert num_classes == 1 or num_classes == 2, "Num Classes should be 1 or 2"
        assert norm in ["zscore",
                        "minmax"], "norm should be \'zscore\' or \'minmax\'"

        self.data_dir = data_dir
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.index_list = index_list
        self.is_train = is_train
        self.num_classes = num_classes
        self.crop_size = crop_size
        self.norm = norm
        self.dhw = dhw
        self.is_keyframe = is_keyframe
        self.is_softmax = is_softmax
        self.is_v3d = is_v3d
        self.is_flip = is_flip

    def __len__(self) -> int:
        return len(self.index_list)

    def __getitem__(self, index) -> Dict:

        img = sitk.ReadImage(os.path.join(
            self.data_dir, self.image_dir, self.index_list[index]), sitk.sitkInt32)
        mask = sitk.ReadImage(os.path.join(
            self.data_dir, self.mask_dir, self.index_list[index]), sitk.sitkUInt8)

        img_array = sitk.GetArrayFromImage(img)
        mask_array = sitk.GetArrayFromImage(mask)

        assert len(np.unique(
            mask_array).tolist()) == self.num_classes + 1, "numbers in mask dont equal to num classes"

        nonzero_layers = np.nonzero(img_array.sum(axis=(1, 2)))[0]
        img_array = img_array[nonzero_layers]
        mask_array = mask_array[nonzero_layers]

        if self.norm == "zscore":
            img_array = z_score_norm_2d_numpy(img_array, nonzero=True)
        elif self.norm == "minmax":
            img_array = min_max_norm_2d_numpy(img_array)

        img_array = resize_dhw_numpy(img_array, order=3, dhw=self.dhw)
        mask_array = resize_dhw_numpy(mask_array, order=0, dhw=self.dhw)

        if self.is_train:
            img_array, mask_array = random_crop_numpy(
                img_array, mask_array, crop_size=self.crop_size, is_keyframe=self.is_keyframe)

            if self.is_flip:
                img_array, mask_array = augmentation(img_array, mask_array)

        if self.is_v3d:
            img_array = change_virtual_3d_numpy(img_array)
        else:
            img_array = np.expand_dims(img_array, axis=0)

        mask_array = ont_hot_mask_numpy(
            mask_array, num_classes=self.num_classes, is_softmax=self.is_softmax)

        img_tensor = torch.FloatTensor(img_array.copy())
        mask_tensor = torch.FloatTensor(mask_array.copy())

        if self.is_train:
            img_tensor = img_tensor.permute(1, 0, 2, 3)
            mask_tensor = mask_tensor.permute(1, 0, 2, 3)

        return {"index": self.index_list[index].split('.')[0], "img": img_tensor, "mask": mask_tensor}


class Dataset3D(Dataset):
    def __init__(self, data_dir: str,  image_dir: str, mask_dir: str, index_list: list, is_train: bool = True, num_classes: int = 1,
                 crop_size: Tuple[int] = (32, 224, 224), norm: str = "zscore", dhw: Tuple[int] = (-1, 224, 224), is_keyframe: bool = True, is_softmax: bool = False, is_flip: bool = False) -> None:
        super(Dataset3D, self).__init__()

        assert num_classes == 1 or num_classes == 2, "Num Classes should be 1 or 2"
        assert norm in ["zscore",
                        "minmax"], "norm should be \'zscore\' or \'minmax\'"

        self.data_dir = data_dir
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.index_list = index_list
        self.is_train = is_train
        self.num_classes = num_classes
        self.crop_size = crop_size
        self.norm = norm
        self.dhw = dhw
        self.is_keyframe = is_keyframe
        self.is_softmax = is_softmax
        self.is_flip = is_flip

    def __len__(self) -> int:
        return len(self.index_list)

    def __getitem__(self, index) -> Dict:

        img = sitk.ReadImage(os.path.join(
            self.data_dir, self.image_dir, self.index_list[index]), sitk.sitkInt32)
        mask = sitk.ReadImage(os.path.join(
            self.data_dir, self.mask_dir, self.index_list[index]), sitk.sitkUInt8)

        img_array = sitk.GetArrayFromImage(img)
        mask_array = sitk.GetArrayFromImage(mask)

        assert len(np.unique(
            mask_array).tolist()) == self.num_classes + 1, "numbers in mask dont equal to num classes"

        nonzero_layers = np.nonzero(img_array.sum(axis=(1, 2)))[0]
        img_array = img_array[nonzero_layers]
        mask_array = mask_array[nonzero_layers]

        if self.norm == "zscore":
            img_array = z_score_norm_3d_numpy(img_array, nonzero=True)
        elif self.norm == "minmax":
            img_array = min_max_norm_3d_numpy(img_array)

        img_array = resize_dhw_numpy(img_array, order=3, dhw=self.dhw)
        mask_array = resize_dhw_numpy(mask_array, order=0, dhw=self.dhw)

        if self.is_train:
            img_array, mask_array = random_crop_numpy(
                img_array, mask_array, crop_size=self.crop_size, is_keyframe=self.is_keyframe)
            if self.is_flip:
                img_array, mask_array = augmentation(img_array, mask_array)

        mask_array = ont_hot_mask_numpy(
            mask_array, num_classes=self.num_classes, is_softmax=self.is_softmax)

        img_tensor = torch.FloatTensor(img_array.copy()).unsqueeze(0)
        mask_tensor = torch.FloatTensor(mask_array.copy())

        return {"index": self.index_list[index].split('.')[0], "img": img_tensor, "mask": mask_tensor}


class Dataset2D_Test(Dataset):
    def __init__(self, data_dir: str,  image_dir: str, mask_dir: str, index_list: list, is_train: bool = True, num_classes: int = 1, crop_size: Tuple[int] = (32, 224, 224),
                 norm: str = "zscore", dhw: Tuple[int] = (-1, 224, 224), is_keyframe: bool = True, is_softmax: bool = False, is_v3d: bool = False, is_flip: bool = False) -> None:
        super(Dataset2D_Test, self).__init__()
        assert num_classes == 1 or num_classes == 2, "Num Classes should be 1 or 2"
        assert norm in ["zscore",
                        "minmax"], "norm should be \'zscore\' or \'minmax\'"

        self.data_dir = data_dir
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.index_list = index_list
        self.is_train = is_train
        self.num_classes = num_classes
        self.crop_size = crop_size
        self.norm = norm
        self.dhw = dhw
        self.is_keyframe = is_keyframe
        self.is_softmax = is_softmax
        self.is_v3d = is_v3d
        self.is_flip = is_flip

    def __len__(self) -> int:
        return len(self.index_list)

    def __getitem__(self, index) -> Dict:

        img = sitk.ReadImage(os.path.join(
            self.data_dir, self.image_dir, self.index_list[index]), sitk.sitkInt32)
        mask = sitk.ReadImage(os.path.join(
            self.data_dir, self.mask_dir, self.index_list[index]), sitk.sitkUInt8)

        img_array = sitk.GetArrayFromImage(img)
        mask_array = sitk.GetArrayFromImage(mask)

        assert len(np.unique(
            mask_array).tolist()) == self.num_classes + 1, "numbers in mask dont equal to num classes"

        nonzero_layers = np.nonzero(img_array.sum(axis=(1, 2)))[0]
        img_array = img_array[nonzero_layers]
        mask_array = mask_array[nonzero_layers]

        if self.norm == "zscore":
            img_array = z_score_norm_2d_numpy(img_array, nonzero=True)
        elif self.norm == "minmax":
            img_array = min_max_norm_2d_numpy(img_array)

        img_array = resize_dhw_numpy(img_array, order=3, dhw=self.dhw)
        mask_array = resize_dhw_numpy(mask_array, order=0, dhw=self.dhw)

        if self.is_train:
            img_array, mask_array = random_crop_numpy(
                img_array, mask_array, crop_size=self.crop_size, is_keyframe=self.is_keyframe)

            if self.is_flip:
                img_array, mask_array = augmentation(img_array, mask_array)

        if self.is_v3d:
            img_array = change_virtual_3d_numpy(img_array)
        else:
            img_array = np.expand_dims(img_array, axis=0)

        mask_array = ont_hot_mask_numpy(
            mask_array, num_classes=self.num_classes, is_softmax=self.is_softmax)

        img_tensor = torch.FloatTensor(img_array.copy())
        mask_tensor = torch.FloatTensor(mask_array.copy())

        if self.is_train:
            img_tensor = img_tensor.permute(1, 0, 2, 3)
            mask_tensor = mask_tensor.permute(1, 0, 2, 3)

        return {"index": self.index_list[index].split('.')[0], "img": img_tensor, "mask": mask_tensor, "spacing": img.GetSpacing()}


class Dataset3D_Test(Dataset):
    def __init__(self, data_dir: str,  image_dir: str, mask_dir: str, index_list: list, is_train: bool = True, num_classes: int = 1,
                 crop_size: Tuple[int] = (32, 224, 224), norm: str = "zscore", dhw: Tuple[int] = (-1, 224, 224), is_keyframe: bool = True, is_softmax: bool = False, is_flip: bool = False) -> None:
        super(Dataset3D_Test, self).__init__()

        assert num_classes == 1 or num_classes == 2, "Num Classes should be 1 or 2"
        assert norm in ["zscore",
                        "minmax"], "norm should be \'zscore\' or \'minmax\'"

        self.data_dir = data_dir
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.index_list = index_list
        self.is_train = is_train
        self.num_classes = num_classes
        self.crop_size = crop_size
        self.norm = norm
        self.dhw = dhw
        self.is_keyframe = is_keyframe
        self.is_softmax = is_softmax
        self.is_flip = is_flip

    def __len__(self) -> int:
        return len(self.index_list)

    def __getitem__(self, index) -> Dict:

        img = sitk.ReadImage(os.path.join(
            self.data_dir, self.image_dir, self.index_list[index]), sitk.sitkInt32)
        mask = sitk.ReadImage(os.path.join(
            self.data_dir, self.mask_dir, self.index_list[index]), sitk.sitkUInt8)

        img_array = sitk.GetArrayFromImage(img)
        mask_array = sitk.GetArrayFromImage(mask)

        assert len(np.unique(
            mask_array).tolist()) == self.num_classes + 1, "numbers in mask dont equal to num classes"

        nonzero_layers = np.nonzero(img_array.sum(axis=(1, 2)))[0]
        img_array = img_array[nonzero_layers]
        mask_array = mask_array[nonzero_layers]

        if self.norm == "zscore":
            img_array = z_score_norm_3d_numpy(img_array, nonzero=True)
        elif self.norm == "minmax":
            img_array = min_max_norm_3d_numpy(img_array)

        img_array = resize_dhw_numpy(img_array, order=3, dhw=self.dhw)
        mask_array = resize_dhw_numpy(mask_array, order=0, dhw=self.dhw)

        if self.is_train:
            img_array, mask_array = random_crop_numpy(
                img_array, mask_array, crop_size=self.crop_size, is_keyframe=self.is_keyframe)
            if self.is_flip:
                img_array, mask_array = augmentation(img_array, mask_array)

        mask_array = ont_hot_mask_numpy(
            mask_array, num_classes=self.num_classes, is_softmax=self.is_softmax)

        img_tensor = torch.FloatTensor(img_array.copy()).unsqueeze(0)
        mask_tensor = torch.FloatTensor(mask_array.copy())

        return {"index": self.index_list[index].split('.')[0], "img": img_tensor, "mask": mask_tensor, "spacing": img.GetSpacing()}


def keep_tuple_collate_fn(batch):
    elem = batch[0]
    if isinstance(elem, collections.abc.Mapping):
        return {key: keep_tuple_collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple):
        return batch
    else:
        return default_collate(batch)


class Dataset2D_Predict(Dataset):
    def __init__(self, data_dir: str,  image_dir: str, index_list: list, norm: str = "zscore", dhw: Tuple[int] = (-1, 224, 224), is_v3d: bool = False) -> None:
        super(Dataset2D_Predict, self).__init__()
        assert norm in ["zscore",
                        "minmax"], "norm should be \'zscore\' or \'minmax\'"

        self.data_dir = data_dir
        self.image_dir = image_dir
        self.index_list = index_list
        self.norm = norm
        self.dhw = dhw
        self.is_v3d = is_v3d

    def __len__(self) -> int:
        return len(self.index_list)

    def __getitem__(self, index) -> Dict:
        img_path = os.path.join(
            self.data_dir, self.image_dir, self.index_list[index])
        img = sitk.ReadImage(img_path, sitk.sitkInt32)

        img_array = sitk.GetArrayFromImage(img)

        if self.norm == "zscore":
            img_array = z_score_norm_2d_numpy(img_array, nonzero=True)
        elif self.norm == "minmax":
            img_array = min_max_norm_2d_numpy(img_array)

        img_array = resize_dhw_numpy(img_array, order=3, dhw=self.dhw)

        if self.is_v3d:
            img_array = change_virtual_3d_numpy(img_array)
        else:
            img_array = np.expand_dims(img_array, axis=0)

        img_tensor = torch.FloatTensor(img_array.copy())

        return {"file": self.index_list[index], "img_path": img_path, "img": img_tensor}


class Dataset3D_Predict(Dataset):
    def __init__(self, data_dir: str,  image_dir: str, index_list: list, norm: str = "zscore", dhw: Tuple[int] = (-1, 224, 224)) -> None:
        super(Dataset3D_Predict, self).__init__()

        assert norm in ["zscore",
                        "minmax"], "norm should be \'zscore\' or \'minmax\'"

        self.data_dir = data_dir
        self.image_dir = image_dir
        self.index_list = index_list
        self.norm = norm
        self.dhw = dhw

    def __len__(self) -> int:
        return len(self.index_list)

    def __getitem__(self, index) -> Dict:
        img_path = os.path.join(
            self.data_dir, self.image_dir, self.index_list[index])
        img = sitk.ReadImage(img_path, sitk.sitkInt32)

        img_array = sitk.GetArrayFromImage(img)

        if self.norm == "zscore":
            img_array = z_score_norm_3d_numpy(img_array, nonzero=True)
        elif self.norm == "minmax":
            img_array = min_max_norm_3d_numpy(img_array)

        img_array = resize_dhw_numpy(img_array, order=3, dhw=self.dhw)

        img_tensor = torch.FloatTensor(img_array.copy()).unsqueeze(0)

        return {"file": self.index_list[index], "img_path": img_path, "img": img_tensor}


if __name__ == "__main__":
    import pandas as pd
    from torch.utils.data import DataLoader
    fold = 0
    index_path = "./data/CRLM/index_V.csv"

    index_df = pd.read_csv(index_path, index_col=0)
    train_patients = []
    for i in range(5):
        if i != fold:
            train_patients.extend(index_df.loc[i, "index"].strip().split(" "))
    test_patients = index_df.loc[fold, "index"].strip().split(" ")

    data_dir = 'data/CRLM/resection_V'
    train_dataset = Dataset2D(data_dir=data_dir, image_dir="images", mask_dir="liver_tumor_masks",
                              index_list=train_patients, is_train=True, num_classes=2, crop_size=(32, 224, 224), norm="zscore", dhw=(-1, 224, 224), is_keyframe=True, is_softmax=True, is_v3d=True, is_flip=False)

    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    print("len", len(dataloader))

    info = next(iter(dataloader))
    print(info['img'].shape, info['mask'].shape)
