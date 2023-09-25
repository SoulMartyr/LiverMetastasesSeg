from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import Config
import models
from utils.AuxUtils import *
from utils.LossUtils import *
from utils.LogUtils import *
from utils.DataUtils import *


def valid(valid_loader: DataLoader, model: nn.Module, epoch: int, num_classes: int, in_channels: int, crop_size: Tuple[int], device: str,
          is_softmax: bool, thres: List[float], logger_valid: logging.Logger) -> float:
    logger_valid.info("Epoch: {}".format(epoch))
    model.eval()
    total_dice = [AvgOutput() for _ in range(0, num_classes)]
    for _, batch in enumerate(valid_loader):
        index = batch["index"][0]
        img = batch['img'].to(device)
        mask = batch['mask'].to(device)

        if img.size(1) != in_channels:
            repeat_num = [in_channels if i ==
                          1 else 1 for i in range(img.dim())]
            img = img.repeat(repeat_num)

        with torch.no_grad():
            pred = sliding_window_inference_2d(
                img, crop_size, model, mask.size(), is_softmax)

        bacth_info = "index:{} ".format(index)
        if is_softmax:
            channel_range = [i for i in range(1, num_classes+1)]
        else:
            channel_range = [i for i in range(0, num_classes)]
        for idx, channel in enumerate(channel_range):
            tmp_dice = dice_with_binary(
                pred, mask, channel, is_softmax, thres[idx])
            total_dice[idx].add(tmp_dice)
            if not is_softmax:
                bacth_info += "thres:{} ".format(thres[idx])
            bacth_info += "Dice{}: {:.4f} ".format(idx, tmp_dice)
        logger_valid.info(bacth_info)
    total_info = "Mean: "
    for idx in range(0, num_classes):
        total_info += "Dice{}: {:.4f} ".format(idx, total_dice[idx].avg())
    logger_valid.info(total_info)
    model.train()
    return total_dice[-1].avg()


def train(model: nn.Module, device: str,  thres: List[float], train_loader: DataLoader,  optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler,  in_channels: int, is_softmax: bool,
          epoch_num: int, start_epoch: int, log_iter: int, valid_epoch: int, ckpt_dir: str, logger_train: logging.Logger, writer: SummaryWriter, valid_args: dict):
    epoch = start_epoch
    best_result = 0.

    while epoch <= epoch_num:

        is_valid = False

        bce_loss_iter = AvgOutput()
        dice_loss_iter = AvgOutput()
        core_dice_iter = AvgOutput()

        for iteration, batch in enumerate(train_loader):
            if not is_valid and epoch % valid_epoch == 0 and epoch != start_epoch:
                valid_args["epoch"] = epoch
                valid_result = valid(**valid_args)
                if valid_result > best_result:
                    save_weight(ckpt_dir, epoch, model.state_dict(
                    ), optimizer.state_dict(), scheduler.state_dict())
                    best_result = valid_result
                is_valid = True

            img = batch['img'].to(device)
            mask = batch['mask'].to(device)

            img_keep_dims, mask_keep_dims = img.size()[2:], mask.size()[2:]
            img, mask = img.view(-1, *
                                 img_keep_dims), mask.view(-1, *mask_keep_dims)

            if img.size(1) != in_channels:
                repeat_num = [in_channels if i ==
                              1 else 1 for i in range(img.dim())]
                img = img.repeat(repeat_num)

            pred = model(img)

            loss_bce = bce_loss(pred, mask)

            out_channels = pred.size(1)
            if is_softmax:
                tgt_dice_channels = [i for i in range(1, out_channels)]
            else:
                tgt_dice_channels = [i for i in range(out_channels)]
            loss_dice = dice_loss(
                pred, mask, tgt_channels=tgt_dice_channels, is_softmax=is_softmax)

            train_loss = loss_bce + loss_dice

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            bce_loss_iter.add(loss_bce.item())
            dice_loss_iter.add(loss_dice.item())
            core_dice_iter.add(dice_with_norm_binary(
                pred, mask, -1, threshold=thres[-1], is_softmax=is_softmax))

            if iteration % log_iter == 0:
                logger_train.info(
                    "Train Epoch:{} Iteration:{} Learning rate:{:.4f} - BCE Loss:{:.4f}, Dice Loss:{:.4f}".format(
                        epoch, iteration, get_learning_rate(optimizer), bce_loss_iter.avg(), dice_loss_iter.avg()))
                logger_train.info(
                    "Threshold {} - Tumor Dice:{:.4f},".format(thres[-1], core_dice_iter.avg()))
                logger_train.info(
                    "--------------------------------------------------")

                writer.add_scalar(tag="loss/train", scalar_value=train_loss.item(),
                                  global_step=epoch * len(train_loader) + iteration)

                bce_loss_iter.clear()
                dice_loss_iter.clear()
                core_dice_iter.clear()
        epoch += 1
        scheduler.step()

    writer.close()
    return best_result


if __name__ == "__main__":
    args = Config.args
    for fold in args.fold:
        cur_month, cur_day = get_month_and_day()
        file_dir = str(cur_month) + str(cur_day) + "_" + \
            args.log_folder + "_" + "fold{}".format(fold)

        log_dir = set_logdir(args.log_dir, file_dir, is_lock=args.lock)
        save_args(args, log_dir)
        logger_train = log_init(log_dir, mode="train")
        logger_valid = log_init(log_dir, mode="valid")
        writer = SummaryWriter(log_dir=log_dir)
        ckpt_dir = set_ckpt_dir(args.ckpt_dir, file_dir)
        logger_train.info("Init Success")

        # Dataset
        train_index = get_index(args.index_path, fold=[
                                f for f in range(5) if f != fold])
        valid_index = get_index(args.index_path, fold=[fold])

        train_dataset_args = {"data_path": args.data_dir, "image_dir": args.image_dir, "mask_dir": args.mask_dir, "index_list": train_index, "is_train": True, "num_classes": args.num_classes,
                              "crop_size": (args.roi_z, args.roi_y, args.roi_x), "norm": args.norm, "dhw": (args.img_d, args.img_h, args.img_w), "is_keyframe": args.keyframe, "is_softmax": args.softmax, "is_flip": args.flip}
        valid_dataset_args = {"data_path": args.data_dir, "image_dir": args.image_dir, "mask_dir": args.mask_dir, "index_list": valid_index, "is_train": False, "num_classes": args.num_classes,
                              "crop_size": (args.roi_z, args.roi_y, args.roi_x), "norm": args.norm, "dhw": (args.img_d, args.img_h, args.img_w), "is_keyframe": args.keyframe, "is_softmax": args.softmax, "is_flip": args.flip}

        train_dataset = Dataset2D(**train_dataset_args)
        valid_dataset = Dataset2D(**valid_dataset_args)

        train_dataloader_args = {"dataset": train_dataset, "batch_size": args.batch_size,
                                 "num_workers": args.num_workers, "drop_last": False, "pin_memory": True}
        valid_dataloader_args = {"dataset": valid_dataset, "batch_size": 1,
                                 "num_workers": args.num_workers, "drop_last": False, "pin_memory": True}

        train_loader = DataLoader(**train_dataloader_args)
        valid_loader = DataLoader(**valid_dataloader_args)
        logger_train.info("Load DataSet Success")

        model_maker = getattr(models, args.model)
        if args.softmax:
            out_channels = args.num_classes + 1
        else:
            out_channels = args.num_classes
        model = model_maker(args.in_channels, out_channels)
        if args.gpu:
            gpus = args.devices
            device = torch.device("cuda:{}".format(gpus[0]))
            model = nn.DataParallel(model, device_ids=gpus)
        else:
            device = torch.device("cpu")
        model = model.to(device)

        if args.optim == "adam":
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay, eps=1e-4)
        elif args.optim == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9,
                                        nesterov=True)
        elif args.optim == "adamw":
            logger_train.info(
                f"weight decay argument will not be used. Default is 11e-2")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        if args.warm_restart:
            logger_train.info(
                'Total number of epochs should be divisible by 10, else it will do odd things')
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 10, eta_min=1e-7)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, args.epochs + 10)

        if args.use_ckpt:
            weight = torch.load(args.pre_ckpt_path)
            start_epoch = weight['epoch']
            model.load_state_dict(weight['model_state_dict'], strict=True)
            optimizer.load_state_dict(weight['optim_state_dict'])
            scheduler.load_state_dict(weight['sched_state_dict'])
        else:
            start_epoch = 0

        logger_train.info("Load Model\Optimizer\Scheduler Success")

        if args.num_classes == 1:
            thres = [args.thres1]
        elif args.num_classes == 2:
            thres = [args.thres1, args.thres2]

        valid_args = {"model": model, "device": device, "thres": thres, "valid_loader": valid_loader, "num_classes": args.num_classes, "in_channels": args.in_channels, "epoch": start_epoch,
                      "crop_size": (args.roi_z, args.roi_y, args.roi_x), "logger_valid": logger_valid, "is_softmax": args.softmax}
        train_args = {"model": model, "device": device, "thres": thres, "train_loader": train_loader, "optimizer": optimizer, "scheduler": scheduler, "in_channels": args.in_channels, "is_softmax": args.softmax, "epoch_num": args.epoch_num,
                      "start_epoch": start_epoch, "log_iter": args.log_iter, "valid_epoch": args.valid_epoch, "ckpt_dir": ckpt_dir, "logger_train": logger_train, "writer": writer, "valid_args": valid_args}
        best_result = train(**train_args)
        logger_valid.info("Best Dice: {}".format(str(best_result)))

        total_vars = set(train_args.keys()) | set(valid_args.keys())
        for var in total_vars:
            del var
