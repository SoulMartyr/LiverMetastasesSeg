import sys
import logging
from typing import List, Tuple

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import Config
from models import models_2d
from utils.DataUtils import Dataset2D
from utils.MetricsUtils import Metrics
from utils.LossUtils import Loss, dice_with_norm_binary
from utils.LogUtils import get_month_and_day, set_log_fold_dir, save_args, log_init
from utils.AuxUtils import AvgOutput, get_index, get_learning_rate, get_ckpt_path, save_weight, sliding_window_inference_2d


def valid(valid_loader: DataLoader, model: nn.Module,  device: str, epoch: int, num_classes: int,
          is_softmax: bool, thres: List[float], crop_size: Tuple[int], logger_valid: logging.Logger) -> float:
    logger_valid.info("Epoch: {}".format(epoch))

    metrics_types = ["dice_per_case", "dice_global"]
    metrics = Metrics(num_samples=len(valid_loader), num_classes=num_classes,
                      is_softmax=is_softmax, thres=thres, metrics_types=metrics_types)

    cmp_channel = -1
    cmp_metric_type_idx = 0

    try:
        model.eval()

        for _, batch in enumerate(valid_loader):
            index = batch["index"][0]
            img = batch['img'].to(device)
            mask = batch['mask'].to(device)

            with torch.no_grad():
                pred = sliding_window_inference_2d(
                    img, crop_size, model, mask.size(), is_softmax)

            metrics.info_per_case_metrics(
                index, pred, mask, info_func=logger_valid.info)

        model.train()
    except Exception as e:
        logger_valid.error(e, exc_info=True)
        sys.exit()

    mean_metrics_result = metrics.compute_mean_per_case_metrics()
    metrics.info_mean_per_case_metrics(logger_valid.info)
    metrics.info_global_metrics(logger_valid.info)

    return mean_metrics_result[cmp_channel][cmp_metric_type_idx]


def train(model: nn.Module, device: str, train_loader: DataLoader,  optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, loss: nn.Module, thres: List[float],
          is_softmax: bool, epoch_num: int, start_epoch: int, log_iter: int, valid_epoch: int, ckpt_dir: str, logger_train: logging.Logger, writer: SummaryWriter, valid_args: dict) -> float:
    epoch = start_epoch
    best_result = 0.
    loss_iter = AvgOutput(length=len(loss))
    core_dice_iter = AvgOutput()

    try:
        while epoch <= epoch_num:

            if epoch % valid_epoch == 0 and epoch != start_epoch:
                valid_args["epoch"] = epoch
                valid_result = valid(**valid_args)
                if valid_result > best_result:
                    save_weight(ckpt_dir, epoch, model, optimizer, scheduler)
                    best_result = valid_result

            logger_train.info("Train Epoch:{}".format(epoch))

            for iteration, batch in enumerate(train_loader):
                img = batch['img'].to(device)
                mask = batch['mask'].to(device)

                img_keep_dims, mask_keep_dims = img.size()[2:], mask.size()[2:]
                img, mask = img.view(-1, *
                                     img_keep_dims), mask.view(-1, *mask_keep_dims)

                pred = model(img)

                train_loss, train_loss_dict = loss(pred, mask)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                writer.add_scalar(tag="loss/train", scalar_value=train_loss.item(),
                                  global_step=epoch * len(train_loader) + iteration)

                loss_iter.add(train_loss_dict.values())
                core_dice_iter.add(dice_with_norm_binary(
                    pred, mask, -1, threshold=thres, is_softmax=is_softmax))

                if iteration % log_iter == 0 and iteration != 0:
                    iter_info = "Iteration:{} Learning rate:{:.5f} - ".format(
                        iteration, get_learning_rate(optimizer))

                    avg_loss_iter = loss_iter.avg()
                    for i, loss_type in enumerate(train_loss_dict.keys()):
                        iter_info += "{}: {:.4f} ".format(
                            loss_type.upper(), avg_loss_iter[i])

                    iter_info += "- Core Dice: {:.4f} ".format(
                        core_dice_iter.avg())
                    logger_train.info(iter_info)

                    loss_iter.clear()
                    core_dice_iter.clear()

            epoch += 1
            scheduler.step()
            logger_train.info("—"*len(iter_info))

    except Exception as e:
        logger_train.error(e, exc_info=True)
        sys.exit()

    writer.close()
    return best_result


if __name__ == "__main__":
    args = Config.args
    for fold in args.fold:
        cur_month, cur_day = get_month_and_day()
        log_folder = str(cur_month) + str(cur_day) + "_" + \
            args.log_folder

        log_fold_dir = set_log_fold_dir(
            args.log_dir, log_folder, fold, is_lock=args.lock)
        save_args(args, log_fold_dir)
        logger_train = log_init(log_fold_dir, fold, mode="train")
        logger_valid = log_init(log_fold_dir, fold, mode="valid")
        writer = SummaryWriter(log_dir=log_fold_dir)
        logger_train.info("Init Success")

        # Dataset
        train_index = get_index(args.index_path, fold=[
                                f for f in range(5) if f != fold])
        valid_index = get_index(args.index_path, fold=[fold])

        train_dataset_args = {"data_dir": args.data_dir, "image_dir": args.image_dir, "mask_dir": args.mask_dir, "index_list": train_index, "is_train": True, "num_classes": args.num_classes,
                              "crop_size": (args.roi_z, args.roi_y, args.roi_x), "norm": args.norm, "dhw": (args.img_d, args.img_h, args.img_w), "is_keyframe": args.keyframe, "is_softmax": args.softmax, "is_flip": args.flip}
        valid_dataset_args = {"data_dir": args.data_dir, "image_dir": args.image_dir, "mask_dir": args.mask_dir, "index_list": valid_index, "is_train": False, "num_classes": args.num_classes,
                              "crop_size": (args.roi_z, args.roi_y, args.roi_x), "norm": args.norm, "dhw": (args.img_d, args.img_h, args.img_w), "is_keyframe": args.keyframe, "is_softmax": args.softmax, "is_flip": False}

        train_dataset = Dataset2D(**train_dataset_args)
        valid_dataset = Dataset2D(**valid_dataset_args)

        train_dataloader_args = {"dataset": train_dataset, "batch_size": args.batch_size, "shuffle": True,
                                 "num_workers": args.num_workers, "drop_last": False, "pin_memory": True}
        valid_dataloader_args = {"dataset": valid_dataset, "batch_size": 1,
                                 "num_workers": args.num_workers, "drop_last": False, "pin_memory": True}

        train_loader = DataLoader(**train_dataloader_args)
        valid_loader = DataLoader(**valid_dataloader_args)
        logger_train.info("Load DataSet Success")

        model_maker = getattr(models_2d, args.model)
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
                optimizer, args.epoch_num + 10)

        if args.use_ckpt:
            ckpt_path = get_ckpt_path(log_fold_dir)
            ckpt = torch.load(ckpt_path, map_location=device)
            start_epoch = ckpt['epoch']
            model.module.load_state_dict(ckpt['model_state_dict'], strict=True)
            optimizer.load_state_dict(ckpt['optim_state_dict'])
            scheduler.load_state_dict(ckpt['sched_state_dict'])
        else:
            start_epoch = 1

        logger_train.info("Load Model\Optimizer\Scheduler Success")

        class_weight = None if len(
            args.class_weight) == 0 else torch.tensor(args.class_weight).to(device)
        loss_weight = [1. for _ in range(len(args.loss))] if len(
            args.loss_weight) == 0 else args.loss_weight
        loss_weight = torch.tensor(loss_weight).to(device)

        loss = Loss(args.num_classes, args.background, args.softmax,
                    args.loss, class_weight, loss_weight)

        thres = [0.5 for _ in range(args.num_classes)] if len(
            args.loss_weight) == 0 else args.thres
        assert len(
            thres) == args.num_classes, "thres length should equal to num classes"

        valid_args = {"model": model, "device": device, "valid_loader": valid_loader, "epoch": start_epoch, "num_classes": args.num_classes,
                      "thres": thres, "is_softmax": args.softmax, "crop_size": (args.roi_z, args.roi_y, args.roi_x), "logger_valid": logger_valid}
        train_args = {"model": model, "device": device, "train_loader": train_loader, "optimizer": optimizer, "scheduler": scheduler, "loss": loss, "thres": thres, "is_softmax": args.softmax, "epoch_num": args.epoch_num,
                      "start_epoch": start_epoch, "log_iter": args.log_iter, "valid_epoch": args.valid_epoch, "ckpt_dir": log_fold_dir, "logger_train": logger_train, "writer": writer, "valid_args": valid_args}
        best_result = train(**train_args)
        logger_valid.info("Best Dice: {}".format(str(best_result)))

        total_vars = set(train_args.keys()) | set(valid_args.keys())
        for var in total_vars:
            del var
