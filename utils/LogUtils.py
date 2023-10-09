import os
import datetime
import logging
import shutil
import yaml
from argparse import ArgumentParser


def get_month_and_day():
    cur_datetime = datetime.datetime.now()
    cur_month = cur_datetime.month
    cur_day = cur_datetime.day
    return cur_month, cur_day


def set_logdir(log_path: str, file_dir: str, is_lock: bool) -> str:
    log_dir = os.path.join(log_path, file_dir)
    if os.path.exists(log_dir):
        if not os.path.exists(os.path.join(log_dir, "lock.txt")):
            shutil.rmtree(log_dir)
        else:
            raise RuntimeError("Create Log Dir failed")
    os.makedirs(log_dir)
    if is_lock:
        open(os.path.join(log_dir, "lock.txt"), "w").close()
    return log_dir


def log_init(log_dir: str, fold: int, mode: str) -> logging.Logger:
    logger = logging.getLogger('logger_{}_fold{}'.format(mode, fold))
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    filehandler = logging.FileHandler(
        os.path.join(log_dir, '{}.log').format(mode))
    filehandler.setLevel(logging.INFO)
    filehandler.setFormatter(formatter)

    logger.addHandler(filehandler)

    return logger


def save_args(args: ArgumentParser, log_dir: str) -> None:
    """Save parsed arguments to config file.
    """
    config = vars(args).copy()

    config_file = os.path.join(log_dir, "exp_arg.yaml")
    with open(config_file, "w") as file:
        yaml.dump(config, file)
