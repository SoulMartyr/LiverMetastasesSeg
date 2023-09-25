'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-06-27 22:28:52
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-06-27 23:55:15
FilePath: /ljytest/DMFMultiSequence/utils/Logger.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
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


def log_init(log_dir: str, mode: str) -> logging.Logger:
    logger = logging.getLogger('logger_{}'.format(mode))
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    filehandler = logging.FileHandler(
        os.path.join(log_dir, '{}.txt').format(mode))
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
