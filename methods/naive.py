"""
Maintainer: Mohammad Wasil

rainbow-memory
Copyright 2021-present NAVER Corp.
GPLv3
"""
import logging
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from methods.finetune import Finetune
from utils.data_loader import cutmix_data, ImageDataset
from utils.train_utils import select_model, select_optimizer

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class Naive(Finetune):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )

        self.batch_size = kwargs["batchsize"]
        self.n_worker = kwargs["n_worker"]
        self.exp_env = kwargs["stream_env"]
        if kwargs["mem_manage"] == "default":
            self.mem_manage = "uncertainty"

    def set_current_dataset(self, train_datalist, test_datalist):
        random.shuffle(train_datalist)
        # self.prev_streamed_list = self.streamed_list
        self.streamed_list = train_datalist
        self.test_list = test_datalist

    def before_task(self, datalist, cur_iter, init_model=False, init_opt=True):
        logger.info("Apply before_task")

        # incoming classes for openloris is the same (default: blurry setup)
        # see make_openloris_dataset for more details
        if self.dataset == "openloris":
            incoming_classes = OPENLORIS_INSTANCES
        else:
            incoming_classes = pd.DataFrame(datalist)["klass"].unique().tolist()

        print(f"Number of classes in dataframe: {len(incoming_classes)}")
        self.exposed_classes = list(set(self.learned_classes + incoming_classes))
        self.num_learning_class = max(
            len(self.exposed_classes), self.num_learning_class
        )

        if self.mem_manage == "prototype":
            self.model.fc = nn.Linear(self.model.fc.in_features, self.feature_size)
            self.feature_extractor = self.model.to(self.device)
            self.model = ICaRLNet(
                self.feature_extractor, self.feature_size, self.num_learning_class
            )

        in_features = self.model.fc.in_features
        out_features = self.model.fc.out_features
        # To care the case of decreasing head
        new_out_features = max(out_features, self.num_learning_class)
        if init_model:
            # do not reset model if naive is used (fine tuning model from prev task)
            if not self.mode == "naive":
                # init model parameters in every iteration
                logger.info("Reset model parameters")
                self.model = select_model(
                    self.model_name, self.dataset, new_out_features
                )
            else:
                logger.info(f"Using previous model in {self.mode} mode")
        else:
            self.model.fc = nn.Linear(in_features, new_out_features)

        self.params = {
            n: p for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad
        }  # For regularzation methods
        self.model = self.model.to(self.device)

        if init_opt:
            # reinitialize the optimizer and scheduler
            logger.info(f"Reset the optimizer {self.opt_name} and scheduler states")
            self.optimizer, self.scheduler = select_optimizer(
                self.opt_name, self.lr, self.model, self.sched_name
            )

        logger.info(f"Increasing the head of fc {out_features} -> {new_out_features}")

    def after_task(self, cur_iter):
        logger.info("Apply after_task")
        self.learned_classes = self.exposed_classes
        self.num_learned_class = self.num_learning_class
        logger.info(f"No memory update in {self.mode} mode")
