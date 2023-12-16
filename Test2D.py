from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

import Config
from models import models_2d
from utils.LogUtils import get_log_fold_dir
from utils.MetricsUtils import Metrics, binary_torch
from utils.DataUtils import Dataset2D_Test, keep_tuple_collate_fn
from utils.AuxUtils import get_args, get_index, get_ckpt_path, sliding_window_inference_2d


def test(test_loader: DataLoader, model: nn.Module, epoch: int, num_classes: int, crop_size: Tuple[int], device: str,
         is_softmax: bool, thres: List[float], overlap: float, metrics_types: List[str]) -> float:
    print("Epoch: {}".format(epoch))
    metrics = Metrics(num_samples=len(test_loader), num_classes=num_classes,
                      is_softmax=is_softmax, thres=thres, metrics_types=metrics_types)

    model.eval()
    for _, batch in enumerate(test_loader):
        index = batch["index"][0]
        img = batch['img'].to(device)
        mask = batch['mask'].to(device)
        spacing = batch["spacing"][0]

        with torch.no_grad():
            pred = sliding_window_inference_2d(
                img, crop_size, model, mask.size(), is_softmax, overlap)

        metrics.info_per_case_metrics(index, pred, mask, spacing, print)

    metrics.info_mean_per_case_metrics(print)
    metrics.info_global_metrics(print)


if __name__ == "__main__":
    args = Config.args

    assert len(args.fold) == 1, "test only support 1 fold once"

    fold = args.fold

    log_fold_dir = get_log_fold_dir(args.log_dir, args.log_folder, fold[0])

    args_dict = get_args(log_fold_dir)

    test_index = get_index(args_dict["index_path"], fold=fold)

    test_dataset_args = {"data_dir": args_dict["data_dir"], "image_dir": args_dict["image_dir"], "mask_dir": args_dict["mask_dir"], "index_list": test_index, "is_train": False, "num_classes": args_dict["num_classes"],
                         "crop_size": (args_dict["roi_z"], args_dict["roi_y"], args_dict["roi_x"]), "norm": args_dict["norm"], "dhw": (args_dict["img_d"], args_dict["img_h"], args_dict["img_w"]), "is_keyframe": args_dict["keyframe"], "is_softmax": args_dict["softmax"], "is_v3d": args_dict["v3d"], "is_flip": False}

    test_dataset = Dataset2D_Test(**test_dataset_args)

    test_dataloader_args = {"dataset": test_dataset, "batch_size": 1, "num_workers": args.num_workers,
                            "drop_last": False, "pin_memory": True, "collate_fn": keep_tuple_collate_fn}

    test_loader = DataLoader(**test_dataloader_args)

    model_maker = getattr(models_2d, args_dict["model"])
    if args_dict["softmax"]:
        out_channels = args_dict["num_classes"] + 1
    else:
        out_channels = args_dict["num_classes"]
    model = model_maker(args_dict["in_channels"], out_channels)
    if args.gpu:
        gpus = args.devices
        assert len(gpus) == 1, "test should use single device"
        device = torch.device("cuda:{}".format(gpus[0]))
    else:
        device = torch.device("cpu")
    model = model.to(device)

    ckpt_path = get_ckpt_path(log_fold_dir)
    ckpt = torch.load(ckpt_path, map_location=device)
    start_epoch = ckpt['epoch']
    model.load_state_dict(ckpt['model_state_dict'], strict=True)

    thres = [0.5 for _ in range(args_dict["num_classes"])] if len(
        args_dict["thres"]) == 0 else args_dict["thres"]
    assert len(
        thres) == args_dict["num_classes"], "thres length should equal to num classes"

    test_args = {"model": model, "device": device, "thres": thres, "test_loader": test_loader, "num_classes": args_dict["num_classes"], "epoch": start_epoch,
                 "crop_size": (args_dict["roi_z"], args_dict["roi_y"], args_dict["roi_x"]),  "is_softmax": args_dict["softmax"], "overlap": args_dict["overlap"], "metrics_types": args.metrics}
    test(**test_args)
