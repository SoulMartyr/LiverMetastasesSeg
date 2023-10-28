import os
from tqdm import tqdm
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

import Config
from models import models_3d
from utils.DataUtils import Dataset3D_Predict
from utils.MetricsUtils import binary_torch
from utils.LogUtils import get_month_and_day, get_log_fold_dir
from utils.AuxUtils import get_args, get_ckpt_path, predict_merge_channel_torch, set_pred_dir, get_index, sliding_window_inference_3d, save_predict_mask


def predict(pred_loader: DataLoader, model: nn.Module, pred_dir: str, out_channels: int, crop_size: Tuple[int], device: str,
            is_softmax: bool, overlap: float, thres: List[float]) -> float:
    model.eval()
    for _, batch in enumerate(tqdm(pred_loader)):
        img = batch['img'].to(device)
        img_path = batch["img_path"][0]
        file = batch["file"][0]

        outputs_size = torch.Size((img.size(0), out_channels, *img.size()[2:]))

        with torch.no_grad():
            pred = sliding_window_inference_3d(
                img, crop_size, model, outputs_size, is_softmax, overlap)

        pred = binary_torch(pred, is_softmax, thres)
        mask_tensor = predict_merge_channel_torch(pred.squeeze(0), is_softmax)

        if mask_tensor.is_cuda:
            mask_tensor = mask_tensor.cpu()

        mask_path = os.path.join(pred_dir, file)
        save_predict_mask(mask_tensor.numpy(), img_path, mask_path)


if __name__ == "__main__":
    args = Config.args

    assert len(args.fold) == 1, "predict only support 1 fold once"
    fold = args.fold

    cur_month, cur_day = get_month_and_day()
    file_dir = str(cur_month) + str(cur_day) + "_" + args.log_folder

    pred_dir = set_pred_dir(args.pred_dir, file_dir, fold[0])

    log_fold_dir = get_log_fold_dir(args.log_dir, args.log_folder, fold[0])

    args_dict = get_args(log_fold_dir)

    pred_index = get_index(args_dict["index_path"], fold=fold)

    pred_dataset_args = {"data_dir": args_dict["data_dir"], "image_dir": args_dict["image_dir"],
                         "index_list": pred_index,  "norm": args_dict["norm"], "dhw": (args_dict["img_d"], args_dict["img_h"], args_dict["img_w"])}

    pred_dataset = Dataset3D_Predict(**pred_dataset_args)

    pred_dataloader_args = {"dataset": pred_dataset, "batch_size": 1,
                            "num_workers": args.num_workers, "drop_last": False, "pin_memory": True}

    pred_loader = DataLoader(**pred_dataloader_args)

    model_maker = getattr(models_3d, args_dict["model"])
    if args_dict["softmax"]:
        out_channels = args_dict["num_classes"] + 1
    else:
        out_channels = args_dict["num_classes"]
    model = model_maker(args_dict["in_channels"], out_channels)
    if args.gpu:
        gpus = args.devices
        assert len(gpus) == 1, "predict should use single device"
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

    pred_args = {"model": model, "pred_dir": pred_dir, "out_channels": out_channels, "device": device, "thres": thres, "pred_loader": pred_loader,
                 "crop_size": (args_dict["roi_z"], args_dict["roi_y"], args_dict["roi_x"]), "is_softmax": args_dict["softmax"], "overlap": args_dict["overlap"]}
    predict(**pred_args)
