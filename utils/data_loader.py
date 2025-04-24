"""
rainbow-memory
Copyright 2021-present NAVER Corp.

Modifications copyright 2024 Mohammad Wasil

Licensed under the GPLv3. See LICENSE file for full text.
"""
import logging.config
import os
from typing import List

import PIL
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random

logger = logging.getLogger()


class ImageDataset(Dataset):
    def __init__(
        self, data_frame: pd.DataFrame, dataset: str, dataset_root=None, transform=None
    ):
        self.data_frame = data_frame
        self.dataset = dataset
        self.transform = transform
        self.dataset_root = dataset_root

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_frame.iloc[idx]["file_name"]
        label = self.data_frame.iloc[idx].get("label", -1)

        # if dataset_root is given, img_name only contains train/factor/instance/filename.jpg
        # otherwise, img_name should contain full path data_dir/train/factor/instance/filename.jpg

        if self.dataset_root:
            img_path = os.path.join(self.dataset_root, img_name)
        else:
            img_path = os.path.join(img_name)
        # print(img_path)
        # img_path = os.path.join("dataset", self.dataset, img_name)
        image = PIL.Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        sample["image"] = image
        sample["label"] = label
        sample["image_name"] = img_path
        return sample

    def get_image_class(self, y):
        return self.data_frame[self.data_frame["label"] == y]


def get_train_datalist(args, cur_iter: int) -> List:
    # set random seed
    random.seed(args.rnd_seed)

    datalist = []

    if args.mode == "joint":
        for cur_iter_ in range(args.n_tasks):
            collection_name = get_train_collection_name(args, iteration=cur_iter_)
            datalist += pd.read_json(
                f"dataset/viewpoint_annotations/{args.exp_name}/{collection_name}.json"
            ).to_dict(orient="records")
            logger.info(f"[Train] Get datalist from {collection_name}.json")
    else:
        collection_name = get_train_collection_name(args, iteration=cur_iter)

        if args.exp_name == "core50_ni_inc":
            if args.core50_run not in [
                "run0",
                "run1",
                "run2",
                "run3",
                "run4",
                "run5",
                "run6",
                "run7",
                "run8",
                "run9",
            ]:
                raise ValueError(f"{args.core50_run} not in run0 to run9")

            collection_path = (
                f"dataset/viewpoint_annotations/{args.exp_name}_{args.core50_run}"
            )
        else:
            collection_path = f"dataset/viewpoint_annotations/{args.exp_name}"

        json_file = os.path.join(collection_path, f"{collection_name}.json")

        print(json_file)
        if os.path.isfile(json_file):
            datalist = pd.read_json(json_file).to_dict(orient="records")
            logger.info(f"[Train] Get datalist from {collection_name}.json")
        else:
            print(f"{json_file} not found")

    if not datalist:
        logger.warning("Datalist empty")
    incoming_classes = pd.DataFrame(datalist)["klass"].unique().tolist()
    logger.info(f"Number of classes in dataframe: {len(incoming_classes)}")

    # shuffle datalist
    random.shuffle(datalist)

    return datalist


def get_train_collection_name(args, iteration):
    ## openloris random seed generated with 3
    ol_seed = 3

    if "openloris_sequential" in args.dataset:
        collection_name = (
            f"{args.dataset}_train_rand{ol_seed}_cls{args.n_cls_a_task}_task{iteration}"
        )
    elif "core50_ni_inc" in args.dataset:
        collection_name = f"train_batch_0{iteration}_filelist"
    else:
        collection_name = f"{args.dataset}_train_{args.exp_name}_rand{args.rnd_seed}_cls{args.n_cls_a_task}_task{iteration}"

    return collection_name


def get_test_datalist(args, exp_name: str, cur_iter: int) -> List:
    """get all test list"""
    if exp_name is None:
        exp_name = args.exp_name

    if exp_name in ["joint", "openloris_sequential"]:
        tasks = list(range(args.n_tasks))
    elif exp_name == "core50_ni_inc":
        # core50 combines test set into one task
        tasks = [1]
    else:
        raise NotImplementedError

    ol_seed = 3

    datalist = []
    for iter_ in tasks:
        if "openloris" in args.dataset:
            collection_name = (
                f"{args.dataset}_test_rand{ol_seed}_cls{args.n_cls_a_task}_task{iter_}"
            )
            json_file = f"dataset/viewpoint_annotations/test/{collection_name}.json"
        elif "core50_ni_inc" in args.dataset:
            collection_name = "test_filelist"
            json_file = f"dataset/viewpoint_annotations/{args.exp_name}_{args.core50_run}/{collection_name}.json"
        else:
            collection_name = f"{args.dataset}_test_rand{args.rnd_seed}_cls{args.n_cls_a_task}_task{iter_}"
            json_file = f"dataset/viewpoint_annotations/test/{collection_name}.json"

        datalist += pd.read_json(json_file).to_dict(orient="records")

        logger.info(f"[Test ] Get datalist from {collection_name}.json")

    incoming_classes = pd.DataFrame(datalist)["klass"].unique().tolist()
    logger.info(f"Number of classes in dataframe: {len(incoming_classes)}")
    return datalist


def get_task_test_list(args, task_id: id) -> List:
    """
    get test data given task id
    """
    ol_seed = 3

    # datalist = []
    # for iter_ in tasks:
    if "openloris" in args.dataset:
        collection_name = (
            f"{args.dataset}_test_rand{ol_seed}_cls{args.n_cls_a_task}_task{task_id}"
        )
        datalist = pd.read_json(
            f"dataset/viewpoint_annotations/test/{collection_name}.json"
        ).to_dict(orient="records")
    elif "core50_ni_inc" in args.dataset:
        collection_name = "test_filelist"
        datalist = pd.read_json(
            f"dataset/viewpoint_annotations/{args.exp_name}_{args.core50_run}/{collection_name}.json"
        ).to_dict(orient="records")
    else:
        collection_name = f"{args.dataset}_test_rand{args.rnd_seed}_cls{args.n_cls_a_task}_task{task_id}"
        datalist = pd.read_json(
            f"dataset/viewpoint_annotations/test/{collection_name}.json"
        ).to_dict(orient="records")

    # datalist = pd.read_json(f"dataset/viewpoint_annotations/test/{collection_name}.json").to_dict(orient="records")

    print(f"[Test ] Get datalist from {collection_name}.json")

    incoming_classes = pd.DataFrame(datalist)["klass"].unique().tolist()
    print(f"Number of classes in dataframe: {len(incoming_classes)}")
    return datalist


def get_statistics(dataset: str):
    """
    Returns statistics of the dataset given a string of dataset name.
    To add new dataset, please add required statistics here
    """
    assert dataset in [
        "openloris_sequential",
        "core50_ni_inc",
    ]
    mean = {
        "openloris_sequential": (0.3839, 0.3846, 0.3655),
        "core50_ni_inc": (0.485, 0.456, 0.406),
    }

    std = {
        "openloris_sequential": (0.1387, 0.1382, 0.1520),
        "core50_ni_inc": (0.229, 0.224, 0.225),
    }

    classes = {
        "openloris_sequential": 121,
        "core50_ni_inc": 50,
    }

    in_channels = {
        "openloris_sequential": 3,
        "core50_ni_inc": 3,
    }

    inp_size = {
        "openloris_sequential": 224,
        "core50_ni_inc": 128,
    }
    return (
        mean[dataset],
        std[dataset],
        classes[dataset],
        inp_size[dataset],
        in_channels[dataset],
    )


# from https://github.com/drimpossible/GDumb/blob/74a5e814afd89b19476cd0ea4287d09a7df3c7a8/src/utils.py#L102:5
def cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5):
    assert alpha > 0
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
