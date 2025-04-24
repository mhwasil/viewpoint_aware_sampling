"""
Author: Mohammad Wasil
Adapted from https://github.com/clovaai/rainbow-memory/blob/master/main.py

Modifications copyright 2024 Mohammad Wasil

Licensed under the GPLv3. See LICENSE file for full text.
"""

import logging.config
import json
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from randaugment import RandAugment
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from configuration import config
from utils.augment import Cutout, select_autoaugment
from utils.data_loader import get_test_datalist, get_statistics, get_task_test_list
from utils.data_loader import get_train_datalist
from utils.method_manager import select_method
import time
from datetime import datetime
import pandas as pd
from utils.train_utils import select_model, select_optimizer
from utils.data_loader import ImageDataset
from torch.utils.data import DataLoader
import argparse


def train(args):
    tr_names = ""
    for trans in args.transforms:
        tr_names += "_" + trans

    if args.dataset == "core50_ni_inc":
        args.n_init_cls = 50
        args.n_cls_a_task = 50
        args.n_tasks = 8
    elif args.dataset == "openloris_sequential":
        args.n_init_cls = 121
        args.n_cls_a_task = 121
        args.n_tasks = 12

    start_time = time.time()
    start_date_time = datetime.fromtimestamp(start_time)
    start_time_formatted = start_date_time.strftime("%d-%m-%Y_%H:%M:%S")

    if args.mode == "naive":
        save_path = f"{args.mode}_{args.mem_manage}_{args.stream_env}"

        if "core50" in args.dataset:
            # use run id instead of seed for core50
            save_path = save_path + f"_{args.core50_run}"
        else:
            save_path = save_path + f"_rnd{args.rnd_seed}"
    else:
        pretrain = "_pretrain" if args.pretrain else ""
        save_path = f"{args.mode}"
        if args.mode in ["rm", "viewpoint"]:
            save_path = save_path + f"_{args.mem_manage}"

        save_path = save_path + f"_{args.stream_env}"
        if args.stream_env == "online":
            buffer_finetune = "buffer_finetune" if args.finetune_with_buffer else ""
            save_path = f"{save_path}_{buffer_finetune}"

        save_path = save_path + f"_msz{args.memory_size}"

        save_path = save_path + f"_{pretrain}"

        if "core50" in args.dataset:
            # use run id instead of seed for core50
            save_path = save_path + f"_{args.core50_run}{tr_names}"
        else:
            save_path = save_path + f"_rnd{args.rnd_seed}{tr_names}"

    save_path = save_path if not args.debug else "debug_" + save_path
    save_path = f"{args.dataset}/{save_path}_{start_time_formatted}"
    # create dir for results for saving acc results
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir, exist_ok=True)

    result_dir = f"{args.result_dir}/{args.dataset}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    if args.stream_env == "online" and args.mode != "gdumb":
        result_prefix = f"{args.result_dir}/{save_path}_{args.model_name}_epoch_1"
    else:
        result_prefix = (
            f"{args.result_dir}/{save_path}_{args.model_name}_epoch_{args.n_epoch}"
        )

    # saved examples and model path and misc
    save_misc_dir = f"{result_prefix}"
    if not os.path.exists(save_misc_dir):
        os.makedirs(save_misc_dir, exist_ok=True)

    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()

    os.makedirs(f"logs/{args.dataset}", exist_ok=True)
    fileHandler = logging.FileHandler("logs/{}.log".format(save_path), mode="w")
    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    logger.info(f"Start time: {start_time_formatted}")
    logger.info(args)
    logger.info(f"Save path: {save_path}")

    writer = SummaryWriter("tensorboard")

    if torch.cuda.is_available():
        if args.cuda_idx:
            device = torch.device(f"cuda:{str(args.cuda_idx)}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Set the device ({device})")
    logger.info(f"Device name: {torch.cuda.get_device_name()}")

    # Fix the random seeds
    # https://hoya012.github.io/blog/reproducible_pytorch/
    torch.manual_seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)

    # Transform Definition
    mean, std, n_classes, inp_size, _ = get_statistics(dataset=args.dataset)
    train_transform = []
    if "cutout" in args.transforms:
        train_transform.append(Cutout(size=16))
    if "randaug" in args.transforms:
        train_transform.append(RandAugment())
    if "autoaug" in args.transforms:
        train_transform.append(select_autoaugment(args.dataset))

    train_transform = transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.RandomCrop(inp_size, padding=4),
            transforms.RandomHorizontalFlip(),
            *train_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    logger.info(f"Using train-transforms {train_transform}")

    test_transform = transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    logger.info(f"[1] Select a CIL method ({args.mode})")
    criterion = nn.CrossEntropyLoss(reduction="mean")
    method = select_method(
        args, criterion, device, train_transform, test_transform, n_classes
    )

    if args.debug:
        logger.info("Model's state_dict:")
        for param_tensor in method.model.state_dict():
            logger.info(
                f"{param_tensor} \t {method.model.state_dict()[param_tensor].size()}"
            )

    logger.info(f"[2] Incrementally training {args.n_tasks} tasks")
    task_records = defaultdict(list)
    for cur_iter in range(args.n_tasks):
        if args.mode == "joint" and cur_iter > 0:
            # raise ValueError(f"mode = {args.mode}")
            print(f"mode: {args.mode}, stopping the training...")
            break

        print("\n" + "#" * 50)
        print(f"# Task {cur_iter} iteration")
        print("#" * 50 + "\n")
        logger.info("[2-1] Prepare a datalist for the current task")

        # best acc given multiple epochs
        best_task_acc = 0.0
        eval_dict = dict()

        # get datalist
        cur_train_datalist = get_train_datalist(args, cur_iter)
        cur_test_datalist = get_test_datalist(args, args.exp_name, cur_iter)

        # Reduce datalist in Debug mode
        if args.debug:
            random.shuffle(cur_train_datalist)
            random.shuffle(cur_test_datalist)
            cur_train_datalist = cur_train_datalist[: int(len(cur_train_datalist) / 2)]
            cur_test_datalist = cur_test_datalist[:1280]
            args.n_epoch = 2

        logger.info("[2-2] Set environment for the current task")
        method.set_current_dataset(cur_train_datalist, cur_test_datalist)
        # Increment known class for current task iteration.
        method.before_task(cur_train_datalist, cur_iter, args.init_model, args.init_opt)

        # The way to handle streamed samles
        logger.info(f"[2-3] Start to train under {args.stream_env}")

        if args.stream_env == "offline" or args.mode == "joint" or args.mode == "gdumb":
            # Offline Train
            best_task_acc, eval_dict = method.train(
                cur_iter=cur_iter,
                n_epoch=args.n_epoch,
                batch_size=args.batchsize,
                n_worker=args.n_worker,
            )
            if args.mode == "joint":
                logger.info(f"joint accuracy: {best_task_acc}")

        elif args.stream_env == "online":
            # Online Train
            logger.info("Train over streamed data once")
            method.train(
                cur_iter=cur_iter,
                n_epoch=1,
                batch_size=args.batchsize,
                n_worker=args.n_worker,
            )

            if args.mode != "naive":
                method.update_memory(cur_iter)

            logger.info(f"Finetune with buffer: {args.finetune_with_buffer}")

            if args.finetune_with_buffer:
                # No stremed training data, train with only memory_list
                method.set_current_dataset([], cur_test_datalist)

                logger.info("Train over memory")
                best_task_acc, eval_dict = method.train(
                    cur_iter=cur_iter,
                    n_epoch=args.n_epoch,
                    batch_size=args.batchsize,
                    n_worker=args.n_worker,
                )

            # method.after_task(cur_iter)

        logger.info("[2-4] Update the information for the current task")
        method.after_task(cur_iter)

        # record task
        if cur_iter == 0:
            task_records["best_task_acc"] = {}
            task_records["cls_acc"] = {}
            task_records["acc"] = {}
            task_records["loss"] = {}

        # evaluate
        logger.info("Evalution on all tasks")
        task_records["cls_acc"][cur_iter] = []
        task_records["acc"][cur_iter] = []
        task_records["loss"][cur_iter] = []

        # set num test task to 1 (combined) for core 50, and n_tasks for openloris (split per task)
        num_test_tasts = 1 if "core50_ni" in args.exp_name else args.n_tasks

        for test_task_id in range(num_test_tasts):
            curr_task_test_datalist = get_task_test_list(args, test_task_id)
            # _, curr_task_test_loader = method.get_dataloader(args.batchsize, 2, None, test_list)
            eval_dict = method.evaluation_ext(curr_task_test_datalist)
            task_records["cls_acc"][cur_iter].append(eval_dict["cls_acc"])
            task_records["acc"][cur_iter].append(eval_dict["acc"])
            task_records["loss"][cur_iter].append(eval_dict["loss"])

        logger.info("[2-5] Report task result")

        # save selected examples for each task
        save_example = True
        if save_example and len(method.memory_list) > 0:
            save_example_path = f"{save_misc_dir}/task_{cur_iter}.csv"

            logger.info(f"Saving model for task {cur_iter} to {save_example_path}")
            pd.DataFrame(method.memory_list).to_csv(save_example_path)

        if args.save_model and not args.debug:
            model_path = f"{save_misc_dir}/model_task_{cur_iter}.pt"
            logger.info(f"Saving model for task {cur_iter} to {model_path}")
            torch.save(method.model.state_dict(), model_path)

    # record time
    finish_time = time.time()
    finish_date_time = datetime.fromtimestamp(finish_time)
    finish_time_formatted = finish_date_time.strftime("%d-%m-%Y_%H:%M:%S")
    total_time = finish_date_time - start_date_time

    logger.info(f"Finish time: {finish_time_formatted}")
    logger.info(f"Total training time: {str(total_time)}")

    result_file_path = f"{result_prefix}.pkl"
    with open(result_file_path, "wb") as fp:
        pickle.dump(task_records, fp)

    logger.info(f"Results saved to {result_file_path}")

    return task_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online Domain Incremental Learning")
    args = config.base_parser()
    task_record = train(args)
