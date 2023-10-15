from tqdm import tqdm

from torch import nn
from torch.utils.data import DataLoader

import Config
from models import models_3d
from utils.AuxUtils import *
from utils.LossUtils import *
from utils.LogUtils import *
from utils.DataUtils import *


def predict(pred_loader: DataLoader, model: nn.Module, pred_dir: str, out_channels: int, crop_size: Tuple[int], device: str,
            is_softmax: bool, overlap: float, thres: List[float]) -> float:
    model.eval()
    for _, batch in enumerate(tqdm(pred_loader)):
        img = batch['img'].to(device)
        img_path = batch["img_path"][0]
        file = batch["file"][0]

        outputs_size = list(img.size())
        outputs_size[1] = out_channels
        outputs_size = torch.Size(outputs_size)

        with torch.no_grad():
            pred = sliding_window_inference_3d(
                img, crop_size, model, outputs_size, is_softmax, overlap)

        if is_softmax:
            pred = softmax_binary_torch(pred)
        else:
            pred = sigmoid_binary_torch(pred, thres)
        mask_tensor = predict_merge_channel_torch(pred.squeeze(0), is_softmax)

        if mask_tensor.is_cuda:
            mask_tensor = mask_tensor.cpu()

        mask_path = os.path.join(pred_dir, file)
        save_predict_mask(mask_tensor.numpy(), img_path, mask_path)


if __name__ == "__main__":
    args = Config.args

    assert len(args.fold) == 1, "predict only support 1 fold once"
    fold = args.fold[0]

    cur_month, cur_day = get_month_and_day()
    file_dir = str(cur_month) + str(cur_day) + "_" + \
        args.log_folder + "_" + "fold{}".format(fold)

    pred_dir = set_pred_dir(args.pred_dir, file_dir)

    pred_index = get_index(args.index_path, fold=[fold])

    pred_dataset_args = {"data_path": args.data_dir, "image_dir": args.image_dir,
                         "index_list": pred_index,  "norm": args.norm, "dhw": (args.img_d, args.img_h, args.img_w)}

    pred_dataset = Dataset3D_Predict(**pred_dataset_args)

    pred_dataloader_args = {"dataset": pred_dataset, "batch_size": 1,
                            "num_workers": args.num_workers, "drop_last": False, "pin_memory": True}

    pred_loader = DataLoader(**pred_dataloader_args)

    model_maker = getattr(models_3d, args.model)
    if args.softmax:
        out_channels = args.num_classes + 1
    else:
        out_channels = args.num_classes
    model = model_maker(args.in_channels, out_channels)
    if args.gpu:
        gpus = args.devices
        assert len(gpus) == 1, "predict should use single device"
        device = torch.device("cuda:{}".format(gpus[0]))
    else:
        device = torch.device("cpu")
    model = model.to(device)

    if args.use_ckpt:
        ckpt = torch.load(args.pre_ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
    else:
        raise RuntimeError("no use checkpoint")

    if args.num_classes == 1:
        thres = [args.thres1]
    elif args.num_classes == 2:
        thres = [args.thres1, args.thres2]

    pred_args = {"model": model, "pred_dir": pred_dir, "out_channels": out_channels, "device": device, "thres": thres, "pred_loader": pred_loader,
                 "crop_size": (args.roi_z, args.roi_y, args.roi_x), "is_softmax": args.softmax, "overlap": args.overlap}
    predict(**pred_args)
