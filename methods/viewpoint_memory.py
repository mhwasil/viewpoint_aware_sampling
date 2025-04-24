"""
Author: Mohammad Wasil
Viewpoint-Aware Sampling (VAS) Policy
Adapted from https://github.com/clovaai/rainbow-memory/blob/master/methods/rainbow_memory.py

Licensed under the GPLv3. See LICENSE file for full text.
"""
import logging
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from methods.finetune import Finetune
from utils.data_loader import cutmix_data, ImageDataset

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class MemoryManager:
    """
    Memory manager for VAS
    """

    def __init__(self, max_size, num_classes):
        self.max_size = max_size
        self.num_classes = num_classes
        self.memory_list_df = pd.DataFrame()
        self.last_task = 0

    def update_memory(self, cur_task, incoming_data_df):
        incoming_data_df["task"] = np.full(len(incoming_data_df), cur_task)
        self.last_task = cur_task

        if cur_task == 1:
            task_df = self.sample_and_filter_data(incoming_data_df)
            self.memory_list_df = task_df
        else:
            mem_per_task = self.max_size // self.last_task
            old_memory_df = self.reduce_samples(self.memory_list_df, mem_per_task)

            # check if there are still left over
            new_memory_df = self.reduce_samples(incoming_data_df, mem_per_task)

            self.memory_list_df = pd.concat(
                [old_memory_df, new_memory_df], ignore_index=True
            )

    def sample_and_filter_data(self, incoming_data_df):
        task_df = pd.DataFrame()
        cur_labels = incoming_data_df.label.unique()
        max_img_per_cls_per_task = int(self.max_size / self.num_classes)

        for i, label in enumerate(cur_labels):
            cls_df = incoming_data_df[incoming_data_df["label"] == label]
            selected_sample_df = self.sample_view_data(cls_df, max_img_per_cls_per_task)

            task_df = pd.concat([task_df, selected_sample_df], ignore_index=True)

        mem_per_cls = self.max_size // self.num_classes
        task_df = (
            task_df.groupby("klass")
            .apply(lambda x: x.sample(mem_per_cls, replace=False))
            .reset_index(drop=True)
        )

        return task_df

    def sample_view_data(self, cls_df, max_img_per_cls_per_task):
        view_df = pd.DataFrame()
        for _, group in cls_df.groupby("view"):
            sample_size = min(
                max_img_per_cls_per_task // len(group.view.unique()), len(group)
            )
            view_df = pd.concat(
                [view_df, group.sample(sample_size, replace=False)], ignore_index=True
            )
        return view_df

    def reduce_samples(self, task_data, mem_per_task, unfilled_slots_in_past_mem=0):
        result_df = pd.DataFrame()

        mem_per_cls_per_task = mem_per_task // self.num_classes
        total_mem_in_prev_tasks = mem_per_task * (self.last_task - 1)

        for task_id in task_data.task.unique():
            task_in_mem_df = task_data[task_data["task"] == task_id]

            if mem_per_task < self.num_classes:
                # take 1 sample per class
                selected_samples = (
                    task_in_mem_df.groupby("klass")
                    .apply(lambda x: x.sample(1, replace=False))
                    .reset_index(drop=True)
                )

                # again resample to match mem_per_task size
                selected_samples = task_in_mem_df.sample(n=mem_per_task, replace=False)

            else:
                if mem_per_cls_per_task < 1:
                    selected_samples = task_in_mem_df.sample(
                        frac=mem_per_cls_per_task, replace=False
                    )
                elif mem_per_cls_per_task == 1:
                    selected_samples = (
                        task_in_mem_df.groupby("klass")
                        .apply(
                            lambda x: x.sample(
                                min(mem_per_cls_per_task, len(x)), replace=False
                            )
                        )
                        .reset_index(drop=True)
                    )
                else:
                    selected_samples = task_in_mem_df.sample(
                        n=min(mem_per_cls_per_task, len(task_in_mem_df)), replace=True
                    )

            # maximize memory capacity by filling unfilled slots
            unfilled_slots = mem_per_task - len(selected_samples)
            while unfilled_slots > 0:
                merged = pd.merge(
                    task_in_mem_df, selected_samples, how="outer", indicator=True
                )

                if unfilled_slots > self.num_classes:
                    merged_by_class = (
                        merged.groupby("klass")
                        .apply(
                            lambda x: x.sample(
                                min(mem_per_cls_per_task, len(x)), replace=False
                            )
                        )
                        .reset_index(drop=True)
                    )
                else:
                    merged_by_class = (
                        merged.groupby("klass")
                        .apply(lambda x: x.sample(1, replace=False))
                        .reset_index(drop=True)
                    )

                merged_by_class = merged_by_class.sample(
                    n=min(unfilled_slots, len(merged_by_class)), replace=False
                )
                merged_by_class = merged_by_class[
                    merged_by_class["_merge"] == "left_only"
                ].drop(columns=["_merge"])

                selected_samples = pd.concat(
                    [selected_samples, merged_by_class], ignore_index=True
                )

                unfilled_slots = mem_per_task - len(selected_samples)

            # combine results
            result_df = pd.concat([result_df, selected_samples], ignore_index=True)

        return result_df

    def reduce_memory_in_cls(self, task_in_mem_df, mem_per_cls_per_task):
        cls_df = pd.DataFrame()

        for cls_id in task_in_mem_df.label.unique():
            task_in_mem_in_cls_df = task_in_mem_df[task_in_mem_df["label"] == cls_id]

            view_df = (
                task_in_mem_in_cls_df.groupby("view")
                .apply(
                    lambda x: x.sample(min(mem_per_cls_per_task, len(x)), replace=False)
                )
                .reset_index(drop=True)
            )

            cls_df = pd.concat([cls_df, view_df], ignore_index=True)

        # resample in case there are left over
        task_in_mem_df = (
            cls_df.groupby("klass")
            .apply(lambda x: x.sample(min(mem_per_cls_per_task, len(x)), replace=False))
            .reset_index(drop=True)
        )

        return task_in_mem_df


class ViewPointMemory(Finetune):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )

        self.batch_size = kwargs["batchsize"]
        self.n_worker = kwargs["n_worker"]
        self.exp_env = kwargs["stream_env"]

        self.mode = kwargs.get("mode", "viewpoint")

        # initialize VAS memory manager
        self.vas_manager = MemoryManager(
            max_size=self.memory_size,
            num_classes=n_classes,
        )

    def train(
        self,
        cur_iter,
        n_epoch,
        batch_size,
        n_worker,
        n_passes=0,
        eval_after_epoch=False,
    ):
        if len(self.memory_list) > 0:
            mem_dataset = ImageDataset(
                pd.DataFrame(self.memory_list),
                dataset=self.dataset,
                transform=self.train_transform,
            )
            memory_loader = DataLoader(
                mem_dataset,
                shuffle=True,
                batch_size=(batch_size // 2),
                num_workers=n_worker,
            )
            stream_batch_size = batch_size - batch_size // 2
        else:
            memory_loader = None
            stream_batch_size = batch_size

        # train_list == streamed_list in RM
        train_list = self.streamed_list
        test_list = self.test_list
        random.shuffle(train_list)
        # Configuring a batch with streamed and memory data equally.
        train_loader, test_loader = self.get_dataloader(
            stream_batch_size, n_worker, train_list, test_list
        )

        logger.info(f"Streamed samples: {len(self.streamed_list)}")
        logger.info(f"In-memory samples: {len(self.memory_list)}")
        logger.info(f"Train samples: {len(train_list)+len(self.memory_list)}")
        logger.info(f"Test samples: {len(test_list)}")

        # Start training
        best_acc = 0.0
        eval_dict = dict()
        self.model = self.model.to(self.device)
        for epoch in range(n_epoch):
            # initialize for each task
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr * 0.1
            elif epoch == 1:  # Then set to maxlr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
            else:  # Aand go!
                self.scheduler.step()

            train_loss, train_acc = self._train(
                train_loader=train_loader,
                memory_loader=memory_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
            )

            writer.add_scalar(f"task{cur_iter}/train/loss", train_loss, epoch)
            writer.add_scalar(f"task{cur_iter}/train/acc", train_acc, epoch)
            writer.add_scalar(
                f"task{cur_iter}/train/lr", self.optimizer.param_groups[0]["lr"], epoch
            )

            if eval_after_epoch:
                eval_dict = self.evaluation(
                    test_loader=test_loader, criterion=self.criterion
                )

                writer.add_scalar(
                    f"task{cur_iter}/test/loss", eval_dict["avg_loss"], epoch
                )
                writer.add_scalar(
                    f"task{cur_iter}/test/acc", eval_dict["avg_acc"], epoch
                )

                logger.info(
                    f"Task {cur_iter} | Epoch {epoch+1}/{n_epoch} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                    f"test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} | "
                    f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
                )

                best_acc = max(best_acc, eval_dict["avg_acc"])
            else:
                logger.info(
                    f"Task {cur_iter} | Epoch {epoch+1}/{n_epoch} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                    f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
                )

        return best_acc, eval_dict

    def update_model(self, x, y, criterion, optimizer):
        optimizer.zero_grad()

        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            logit = self.model(x)
            loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(
                logit, labels_b
            )
        else:
            logit = self.model(x)
            loss = criterion(logit, y)

        _, preds = logit.topk(self.topk, 1, True, True)

        # torch.autograd.set_detect_anomaly(True)
        loss.backward()
        optimizer.step()
        return loss.item(), torch.sum(preds == y.unsqueeze(1)).item(), y.size(0)

    def _train(self, train_loader, memory_loader, optimizer, criterion):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        self.model.train()
        if memory_loader is not None and train_loader is not None:
            data_iterator = zip(train_loader, cycle(memory_loader))
        elif memory_loader is not None:
            data_iterator = memory_loader
        elif train_loader is not None:
            data_iterator = train_loader
        else:
            raise NotImplementedError("None of dataloder is valid")

        for data in data_iterator:
            if len(data) == 2:
                stream_data, mem_data = data
                x = torch.cat([stream_data["image"], mem_data["image"]])
                y = torch.cat([stream_data["label"], mem_data["label"]])
            else:
                x = data["image"]
                y = data["label"]

            x = x.to(self.device)
            y = y.to(self.device)
            l, c, d = self.update_model(x, y, criterion, optimizer)
            total_loss += l
            correct += c
            num_data += d

        if train_loader is not None:
            n_batches = len(train_loader)
        else:
            n_batches = len(memory_loader)

        return total_loss / n_batches, correct / num_data

    def allocate_batch_size(self, n_old_class, n_new_class):
        new_batch_size = int(
            self.batch_size * n_new_class / (n_old_class + n_new_class)
        )
        old_batch_size = self.batch_size - new_batch_size
        return new_batch_size, old_batch_size

    def update_memory(self, cur_iter, num_class=None):
        if num_class is None:
            num_class = self.num_learning_class

        if not self.already_mem_update:
            candidates = self.streamed_list + self.memory_list

            if len(candidates) <= self.memory_size:
                self.memory_list = candidates
                self.seen = len(candidates)
                logger.warning("Candidates < Memory size")
            else:
                self.vas_manager.update_memory(
                    cur_iter + 1, pd.DataFrame.from_records(self.streamed_list)
                )
                self.memory_list = self.vas_manager.memory_list_df.to_dict(
                    orient="records"
                )
                logger.info(f"Stream examples: {len(self.streamed_list)}")
                logger.info(f"In old-memory Examples: {len(self.memory_list)}")
                logger.info(f"Combined examples: {len(candidates)}")

            assert len(self.memory_list) <= self.memory_size

            logger.info(f"Memory statistic in iter {cur_iter}")
            memory_df = pd.DataFrame(self.memory_list)
            # memory update happens only once per task iteratin.
            self.already_mem_update = True
        else:
            logger.warning(f"Already updated the memory during this iter ({cur_iter})")
