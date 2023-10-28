from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

import Config
from models import models_3d
from utils.LogUtils import get_log_fold_dir
from utils.MetricsUtils import Metrics, binary_torch
from utils.DataUtils import Dataset3D_Test, keep_tuple_collate_fn
from utils.AuxUtils import AvgOutput, get_args, get_index, get_ckpt_path, sliding_window_inference_3d


def test(test_loader: DataLoader, model: nn.Module, epoch: int, num_classes: int, crop_size: Tuple[int], device: str,
         is_softmax: bool, overlap: float, thres: List[float], metrics_types: List[str]) -> float:
    print("Epoch: {}".format(epoch))
    total_metrics = [AvgOutput(length=len(metrics_types))
                     for _ in range(0, num_classes)]
    metrics = Metrics(metrics_types)

    model.eval()
    for _, batch in enumerate(test_loader):
        index = batch["index"][0]
        img = batch['img'].to(device)
        mask = batch['mask'].to(device)
        spacing = batch["spacing"][0]

        with torch.no_grad():
            pred = sliding_window_inference_3d(
                img, crop_size, model, mask.size(), is_softmax, overlap)
            pred = binary_torch(pred, is_softmax, thres)
            pred, mask = pred.squeeze(0), mask.squeeze(0)

        bacth_info = "Index:{} ".format(index)

        channel_range = [i if not is_softmax else i +
                         1 for i in range(0, num_classes)]

        for idx, channel in enumerate(channel_range):
            metrics_info, metrics_result = metrics.compute_metrics(
                pred, mask, channel, spacing)

            if not is_softmax:
                bacth_info += "Thres:{} ".format(thres[idx])
            bacth_info += metrics_info
            total_metrics[idx].add(metrics_result)

        print(bacth_info)

    total_info = "Mean: "
    for idx in range(0, num_classes):
        mean_metric = total_metrics[idx].avg()
        for i, metrics_type in enumerate(metrics_types):
            total_info += "{}{}: {:.4f} ".format(
                metrics_type.upper(), idx, mean_metric[i])
    print(total_info)


if __name__ == "__main__":
    args = Config.args

    assert len(args.fold) == 1, "test only support 1 fold once"

    fold = args.fold

    log_fold_dir = get_log_fold_dir(args.log_dir, args.log_folder, fold[0])

    args_dict = get_args(log_fold_dir)

    test_index = get_index(args_dict["index_path"], fold=fold)

    test_dataset_args = {"data_dir": args_dict["data_dir"], "image_dir": args_dict["image_dir"], "mask_dir": args_dict["mask_dir"], "index_list": test_index, "is_train": False, "num_classes": args_dict["num_classes"],
                         "crop_size": (args_dict["roi_z"], args_dict["roi_y"], args_dict["roi_x"]), "norm": args_dict["norm"], "dhw": (args_dict["img_d"], args_dict["img_h"], args_dict["img_w"]), "is_keyframe": args_dict["keyframe"], "is_softmax": args_dict["softmax"], "is_flip": False}

    test_dataset = Dataset3D_Test(**test_dataset_args)

    test_dataloader_args = {"dataset": test_dataset, "batch_size": 1, "num_workers": args.num_workers,
                            "drop_last": False, "pin_memory": True, "collate_fn": keep_tuple_collate_fn}

    test_loader = DataLoader(**test_dataloader_args)

    model_maker = getattr(models_3d, args_dict["model"])
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

    if args_dict["num_classes"] == 1:
        thres = [args_dict["thres1"]]
    elif args_dict["num_classes"] == 2:
        thres = [args_dict["thres1"], args_dict["thres2"]]

    test_args = {"model": model, "device": device, "thres": thres, "test_loader": test_loader, "num_classes": args_dict["num_classes"], "epoch": start_epoch,
                 "crop_size": (args_dict["roi_z"], args_dict["roi_y"], args_dict["roi_x"]),  "is_softmax": args_dict["softmax"], "overlap": args_dict["overlap"], "metrics_types": args.metrics}
    test(**test_args)
