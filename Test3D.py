import sys

from torch import nn
from torch.utils.data import DataLoader

import Config
from models import models_3d
from utils.AuxUtils import *
from utils.LossUtils import *
from utils.LogUtils import *
from utils.DataUtils import *


def test(test_loader: DataLoader, model: nn.Module, epoch: int, num_classes: int, crop_size: Tuple[int], device: str,
         is_softmax: bool, overlap: float, thres: List[float]) -> float:
    print("Epoch: {}".format(epoch))
    model.eval()
    total_dice = [AvgOutput() for _ in range(0, num_classes)]
    try:
        for _, batch in enumerate(test_loader):
            index = batch["index"][0]
            img = batch['img'].to(device)
            mask = batch['mask'].to(device)

            with torch.no_grad():
                pred = sliding_window_inference_3d(
                    img, crop_size, model, mask.size(), is_softmax, overlap)

            bacth_info = "index:{} ".format(index)
            if is_softmax:
                channel_range = [i for i in range(1, num_classes+1)]
            else:
                channel_range = [i for i in range(0, num_classes)]
            for idx, channel in enumerate(channel_range):
                tmp_dice = dice_with_binary(
                    pred, mask, channel, is_softmax, thres)
                total_dice[idx].add(tmp_dice)
                if not is_softmax:
                    bacth_info += "thres:{} ".format(thres[idx])
                bacth_info += "Dice{}: {:.4f} ".format(idx, tmp_dice)
            print(bacth_info)
    except Exception as e:
        print(e, exc_info=True)
        sys.exit()

    total_info = "Mean: "
    for idx in range(0, num_classes):
        total_info += "Dice{}: {:.4f} ".format(idx, total_dice[idx].avg())
    print(total_info)


if __name__ == "__main__":
    args = Config.args

    assert len(args.fold) == 1, "test only support 1 fold once"
    fold = args.fold[0]

    test_index = get_index(args.index_path, fold=[fold])

    test_dataset_args = {"data_path": args.data_dir, "image_dir": args.image_dir, "mask_dir": args.mask_dir, "index_list": test_index, "is_train": False, "num_classes": args.num_classes,
                         "crop_size": (args.roi_z, args.roi_y, args.roi_x), "norm": args.norm, "dhw": (args.img_d, args.img_h, args.img_w), "is_keyframe": args.keyframe, "is_softmax": args.softmax, "is_flip": args.flip}

    test_dataset = Dataset3D(**test_dataset_args)

    test_dataloader_args = {"dataset": test_dataset, "batch_size": 1,
                            "num_workers": args.num_workers, "drop_last": False, "pin_memory": True}

    test_loader = DataLoader(**test_dataloader_args)

    model_maker = getattr(models_3d, args.model)
    if args.softmax:
        out_channels = args.num_classes + 1
    else:
        out_channels = args.num_classes
    model = model_maker(args.in_channels, out_channels)
    if args.gpu:
        gpus = args.devices
        assert len(gpus) == 1, "test should use single device"
        device = torch.device("cuda:{}".format(gpus[0]))
    else:
        device = torch.device("cpu")
    model = model.to(device)

    if args.use_ckpt:
        ckpt = torch.load(args.pre_ckpt_path)
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
    else:
        raise RuntimeError("no use checkpoint")

    if args.num_classes == 1:
        thres = [args.thres1]
    elif args.num_classes == 2:
        thres = [args.thres1, args.thres2]

    test_args = {"model": model, "device": device, "thres": thres, "test_loader": test_loader, "num_classes": args.num_classes, "epoch": start_epoch,
                 "crop_size": (args.roi_z, args.roi_y, args.roi_x),  "is_softmax": args.softmax, "overlap": args.overlap}
    test(**test_args)
