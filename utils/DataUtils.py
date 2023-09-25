import os
import random
import torch.nn as nn
import SimpleITK as sitk
import numpy as np
import torch
from typing import Tuple, Dict, Any
from scipy import ndimage


def tuple_sub(tuple1: Tuple[Any], tuple2: Tuple[Any]) -> Tuple[Any]:
    result_tuple = tuple(x - y for x, y in zip(tuple1, tuple2))
    return result_tuple


def random_crop_numpy(img_array: np.ndarray, mask_array: np.ndarray, crop_size: Tuple[int], is_keyframe: bool) -> Tuple[np.ndarray, np.ndarray]:
    z, y, x = img_array.shape
    if is_keyframe:
        mask_flag = np.unique(mask_array)
        z_indexes, y_indexes, x_indexes = np.where(mask_array == mask_flag[-1])
        min_z, min_y, min_x, max_z, max_y, max_x = np.min(z_indexes), np.min(y_indexes), np.min(
            x_indexes), np.max(z_indexes), np.max(y_indexes), np.max(x_indexes)

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
        indexes = np.nonzero(array)
        arr_mean = np.mean(array[indexes])
        arr_std = np.std(array[indexes])
    else:
        arr_mean = np.mean(array)
        arr_std = np.std(array)

    return (array - arr_mean) / arr_std


def min_max_norm_2d_numpy(array: np.ndarray) -> np.ndarray:
    arr_min = array.min(axis=(1, 2), keepdims=True)
    arr_max = array.max(axis=(1, 2), keepdims=True)

    return (array - arr_min) / (arr_max - arr_min)


def z_score_norm_2d_numpy(array: np.ndarray, nonzero: bool = True) -> np.ndarray:
    if nonzero:
        arr_mean = array.mean(axis=(1, 2), where=(array != 0), keepdims=True)
        arr_std = array.std(axis=(1, 2), where=(array != 0), keepdims=True)
    else:
        arr_mean = array.mean(axis=(1, 2), keepdims=True)
        arr_std = array.std(axis=(1, 2), keepdims=True)

    return (array - arr_mean) / arr_std


def ont_hot_mask_numpy(array: np.ndarray, num_classes: int, is_softmax: bool) -> np.ndarray:
    assert is_softmax or ((not is_softmax) and num_classes == 1)

    if not is_softmax:
        one_hot = [(array == i).astype(
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


class Dataset2D(nn.Module):
    def __init__(self, data_path: str,  image_dir: str, mask_dir: str, index_list: list, is_train: bool = True, num_classes: int = 1,
                 crop_size: Tuple[int] = (32, 224, 224), norm="zscore", dhw=(-1, 224, 224), is_keyframe: bool = True, is_softmax: bool = False, is_flip: bool = False) -> None:
        super(Dataset2D, self).__init__()
        assert num_classes == 1 or num_classes == 2, "Num Classes should be 1 or 2"
        assert norm in ["zscore",
                        "minmax"], "norm should be \'zscore\' or \'minmax\'"

        self.data_path = data_path
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
            self.data_path, self.image_dir, self.index_list[index]), sitk.sitkInt32)
        mask = sitk.ReadImage(os.path.join(
            self.data_path, self.mask_dir, self.index_list[index]), sitk.sitkUInt8)

        img_array = sitk.GetArrayFromImage(img)
        mask_array = sitk.GetArrayFromImage(mask)

        assert len(np.unique(
            mask_array).tolist()) == self.num_classes + 1, "numbers in mask dont equal to num classes"

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

        mask_array = ont_hot_mask_numpy(
            mask_array, num_classes=self.num_classes, is_softmax=self.is_softmax)

        img_tensor = torch.FloatTensor(img_array.copy())
        mask_tensor = torch.FloatTensor(mask_array.copy())
        if self.is_train:
            img_tensor = img_tensor.unsqueeze(1)
            mask_tensor = mask_tensor.permute(1, 0, 2, 3)
        else:
            img_tensor = img_tensor.unsqueeze(0)
        return {"index": self.index_list[index].split('.')[0], "img": img_tensor, "mask": mask_tensor}


class Dataset3D(nn.Module):
    def __init__(self, data_path: str,  image_dir: str, mask_dir: str, index_list: list, is_train: bool = True, num_classes: int = 1,
                 crop_size: Tuple[int] = (32, 224, 224), norm="zscore", dhw=(-1, 224, 224), is_keyframe: bool = True, is_softmax: bool = False, is_flip: bool = False) -> None:
        super(Dataset3D, self).__init__()

        assert num_classes == 1 or num_classes == 2, "Num Classes should be 1 or 2"
        assert norm in ["zscore",
                        "minmax"], "norm should be \'zscore\' or \'minmax\'"

        self.data_path = data_path
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
            self.data_path, self.image_dir, self.index_list[index]), sitk.sitkInt32)
        mask = sitk.ReadImage(os.path.join(
            self.data_path, self.mask_dir, self.index_list[index]), sitk.sitkUInt8)

        img_array = sitk.GetArrayFromImage(img)
        mask_array = sitk.GetArrayFromImage(mask)

        assert len(np.unique(
            mask_array).tolist()) == self.num_classes + 1, "numbers in mask dont equal to num classes"

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


# class Dataset2D_onlyTumor_MultiSeq(nn.Module):
#     def __init__(self, data_path_t2: str, data_path_v: str, index_list: list, num_slices: int = 16, num_classes: int = 2,
#                  is_flip: bool = False, is_train: bool = True, img_size=(512, 512)):
#         super(Dataset2D_onlyTumor_MultiSeq, self).__init__()
#         self.data_path_t2 = data_path_t2
#         self.data_path_v = data_path_v
#         self.index_list = index_list
#         self.num_slices = num_slices
#         self.num_classes = num_classes
#         self.flip = is_flip
#         self.is_train = is_train
#         self.img_size = img_size
#         random.seed(5)

#     def __len__(self):
#         return len(self.index_list)

#     def __getitem__(self, index):
#         assert self.num_classes == 2, "Num Classes should be 2"

#         img_v = sitk.ReadImage(os.path.join(
#             self.data_path_v, "images", self.index_list[index]), sitk.sitkInt32)
#         mask_v = sitk.ReadImage(os.path.join(
#             self.data_path_v, "masks", self.index_list[index]), sitk.sitkUInt8)

#         img_arr_v = sitk.GetArrayFromImage(img_v)
#         mask_arr_v = sitk.GetArrayFromImage(mask_v)

#         img_t2 = sitk.ReadImage(os.path.join(self.data_path_t2, "images",
#                                 self.index_list[index].split('_')[0]+"_T2.nii.gz"), sitk.sitkInt32)
#         mask_t2 = sitk.ReadImage(os.path.join(
#             self.data_path_t2, "masks", self.index_list[index].split('_')[0]+"_T2.nii.gz"), sitk.sitkUInt8)

#         img_arr_t2 = sitk.GetArrayFromImage(img_t2)
#         mask_arr_t2 = sitk.GetArrayFromImage(mask_t2)

#         img_arr_v = z_score_normalization_2d(img_arr_v)
#         img_arr_t2 = z_score_normalization_2d(img_arr_t2)

#         depth_v = img_arr_v.shape[0]
#         depth_t2 = img_arr_t2.shape[0]

#         if self.is_train:
#             nonzero_layers = np.nonzero(np.sum(mask_arr_v, axis=(1, 2)))[0]
#             random_idx = random.choice(nonzero_layers)

#             if random_idx - self.num_slices // 2 < 0:
#                 start_idx = 0
#                 end_idx = self.num_slices
#             elif random_idx + self.num_slices // 2 >= depth_v:
#                 start_idx = depth_v - self.num_slices
#                 end_idx = depth_v
#             else:
#                 start_idx = random_idx - self.num_slices // 2
#                 end_idx = random_idx + self.num_slices // 2
#             img_arr_v = img_arr_v[start_idx:end_idx, :, :]
#             mask_arr_v = mask_arr_v[start_idx:end_idx, :, :]

#             v_mid_silce = (end_idx - start_idx) // 2
#             t2_mid_slice = int((v_mid_silce / depth_v) * depth_t2)

#             if t2_mid_slice - 2 < 0:
#                 start_idx = 0
#                 end_idx = 4
#             elif t2_mid_slice + 1 >= depth_t2:
#                 start_idx = depth_t2 - 4
#                 end_idx = depth_t2
#             else:
#                 start_idx = t2_mid_slice - 2
#                 end_idx = t2_mid_slice + 2
#             img_arr_t2 = img_arr_t2[start_idx:end_idx, :, :]
#             mask_arr_t2 = mask_arr_t2[start_idx:end_idx, :, :]

#         img_arr_v = resize(img_arr_v, order=3, image_size=self.img_size)
#         mask_arr_v = resize(mask_arr_v, order=0, image_size=self.img_size)

#         # img_arr_t2 = resize_depth(img_arr_t2, order=3, depth=16)
#         # mask_arr_t2 = resize_depth(mask_arr_t2, order=0, depth=16)

#         img_arr_t2 = resize(img_arr_t2, order=3, image_size=self.img_size)
#         mask_arr_t2 = resize(mask_arr_t2, order=0, image_size=self.img_size)

#         mask_arr_t2 = np.sum(mask_arr_v, axis=0,
#                              keepdims=True).astype(np.uint8)
#         mask_arr_t2[mask_arr_t2 > 0] = 1

#         img_tensor_v = torch.FloatTensor(img_arr_v.copy())
#         mask_tensor_v = torch.FloatTensor(mask_arr_v.copy())

#         img_tensor_t2 = torch.FloatTensor(img_arr_t2.copy())
#         mask_tensor_t2 = torch.FloatTensor(mask_arr_t2.copy())

#         return {"index": self.index_list[index].split('.')[0], "img_v": img_tensor_v, "mask_v": mask_tensor_v, "img_t2": img_tensor_t2, "mask_t2": mask_tensor_t2}


if __name__ == "__main__":
    import pandas as pd
    from torch.utils.data import DataLoader
    fold = 0
    index_path = "./data/index_V.csv"

    index_df = pd.read_csv(index_path, index_col=0)
    train_patients = []
    for i in range(5):
        if i != fold:
            train_patients.extend(index_df.loc[i, "index"].strip().split(" "))
    test_patients = index_df.loc[fold, "index"].strip().split(" ")

    data_path = 'data\\resection_V'
    train_dataset = Dataset2D(data_path=data_path, image_dir="images", mask_dir="liver_tumor_masks",
                              index_list=train_patients, is_train=True, num_classes=3, crop_size=(32, 224, 224), norm="zscore", dhw=(-1, 224, 224), is_keyframe=True, is_softmax=True, is_flip=False)

    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    print("len", len(dataloader))

    info = next(iter(dataloader))
    print(info['img'].shape, info['mask'].shape)
