import logging
import random
import copy
import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from utils.data_loader import ImageDataset
from methods.finetune import Finetune
from utils.data_loader import cutmix_data
import torch.nn.functional as F

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class MIR(Finetune):
    """
    Maximally-Inferred Retrieval
    Online Continual Learning with Maximally Interfered Retrieval (NeurIPS 2019)
    """

    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )

        self.task_count = 0

        # mem_manage takes care of memory update, default = reservoir
        self.mem_manage = kwargs.get("mem_manage", "reservoir")
        # mem_retrieve takes care of how to retrive memort
        self.mem_retrieve = kwargs.get("mem_retrieve", "random")

        # number of samples to select from buffer. Default 50 in MIR paper
        self.subsample = kwargs.get("subsample", 50)

        # buffer batch size. Default 10 in MIR paper
        self.buffer_batch_size = kwargs.get("buffer_batch_size", 10)

        # number of iteration when replaying with memories
        self.mem_iters = kwargs.get("mem_iters", 1)

        self.online_reg = "online" in kwargs["stream_env"]

    def mem_random_retrieve(self, batch_size):
        """
        Random retrieval of memory given batch size
        """
        return np.random.choice(self.memory_list, batch_size)

    def train(self, cur_iter, n_epoch, batch_size, n_worker, eval_after_epoch=False):
        # Current streams loader
        train_list = self.streamed_list
        random.shuffle(train_list)
        test_list = self.test_list
        train_loader, test_loader = self.get_dataloader(
            batch_size, n_worker, train_list, test_list
        )

        logger.info(f"Streamed samples: {len(self.streamed_list)}")
        logger.info(f"In-memory samples: {len(self.memory_list)}")
        logger.info(f"Train samples: {len(train_list)}")
        logger.info(f"Test samples: {len(test_list)}")

        # TRAIN
        best_acc = 0.0
        eval_dict = dict()
        for epoch in range(n_epoch):
            # learning rate scheduling from
            # https://github.com/drimpossible/GDumb/blob/master/src/main.py
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
                optimizer=self.optimizer,
                epoch=epoch,
                total_epochs=n_epoch,
                batch_size=batch_size,
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

        # Save the weight and importance of weights of current task
        self.task_count += 1

        return best_acc, eval_dict

    def _train(self, train_loader, optimizer, epoch, total_epochs, batch_size=None):
        """
        Train one epoch
        """

        total_loss, correct, num_data = 0.0, 0.0, 0.0
        mem_total_loss, mem_correct, mem_data = 0.0, 0.0, 0.0

        self.model.train()
        for i, data in enumerate(train_loader):
            x = data["image"]
            y = data["label"]
            x = x.to(self.device)
            y = y.to(self.device)
            img_path = data["image_name"]

            # virtual update
            logits = self.model.forward(x)
            loss = self.criterion(logits, y)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            total_loss += loss.item()

            # MIR replay
            if self.cur_iter > 0:
                mem_x, mem_y = self.retrive_sample_from_buffer()
                mem_logits = self.model.forward(mem_x)
                mem_loss = self.criterion(mem_logits, mem_y)
                mem_loss.backward(retain_graph=True)
                total_loss += mem_loss.item()

            optimizer.step()

            # preds and corret are based on current data streams only
            _, preds = logits.topk(self.topk, 1, True, True)
            correct += torch.sum(preds == y.unsqueeze(1)).item()

            num_data += y.size(0)

            # update memory with reservoir at every minibatch iterazion
            self.update_memory_mir(mini_batch_img_paths=img_path)

        n_batches = len(train_loader)
        return total_loss / n_batches, correct / num_data

    def after_task(self, cur_iter):
        logger.info("Apply after_task")
        self.learned_classes = self.exposed_classes
        self.num_learned_class = self.num_learning_class

        logger.warning("ER-MIR manages memory update at every mini batch iteration")

    def update_memory(self, cur_iter):
        logger.warning("Memory update in ER-MIR is performed after each iteration")

    def update_memory_mir(self, mini_batch_img_paths):
        if self.mem_manage == "reservoir":
            self.reservoir_sampling(mini_batch_img_paths)
        else:
            logger.error("Not implemented memory management")
            raise NotImplementedError

        assert len(self.memory_list) <= self.memory_size

    def reservoir_sampling(self, mini_batch_img_paths):
        """
        Given batch img paths, filter self.streamed_list based on them,
        and update reservoir.

        ToDo: there should be a simpler way to this like mammoth framework?
        """
        mini_batch_img_paths = list(mini_batch_img_paths)
        streamed_list_df = pd.DataFrame(self.streamed_list)
        batch_list_df = streamed_list_df[
            streamed_list_df["file_name"].isin(mini_batch_img_paths)
        ].to_dict(orient="records")

        for sample in batch_list_df:
            if len(self.memory_list) < self.memory_size:
                self.memory_list += [sample]
            else:
                j = np.random.randint(0, self.seen)
                if j < self.memory_size:
                    self.memory_list[j] = sample
            self.seen += 1

    def retrive_sample_from_buffer(self):
        """
        Retrieve samples from buffer
        """
        grad_vector = self.get_grad_vec()
        model_temp = self.get_future_step_parameters(grad_vector)

        # retrieve samples from buffer
        sampled_buffer = random.sample(self.memory_list, self.subsample)
        mem_x, mem_y = None, None

        if len(sampled_buffer) > 0:
            sampled_buffer_loader, _ = self.get_dataloader(
                batch_size=self.subsample,
                n_worker=4,
                train_list=sampled_buffer,
                test_list=None,
            )

            for i, buffer_data in enumerate(sampled_buffer_loader):
                bx = buffer_data["image"].to(self.device)
                by = buffer_data["label"].to(self.device)

                logits_track_pre = self.model.forward(bx)
                logits_track_post = model_temp.forward(bx)

                pre_loss = F.cross_entropy(logits_track_pre, by, reduction="none")
                post_loss = F.cross_entropy(logits_track_post, by, reduction="none")

                # compute scores
                scores = post_loss - pre_loss
                # sort based on scores and take examples of size buffer batch size
                indexes = scores.sort(descending=True)[1][: self.buffer_batch_size]

                mem_x = bx[indexes]
                mem_y = by[indexes]
                break

        return mem_x, mem_y

    def get_grad_vec(self):
        """
        Compute gradient vectors
        """
        grad_dims = []
        for param in self.model.parameters():
            grad_dims.append(param.data.numel())

        grads = torch.Tensor(sum(grad_dims))
        grads.fill_(0.0)
        cnt = 0
        for param in self.model.parameters():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en].copy_(param.grad.data.view(-1))

            cnt += 1

        return grads

    def get_future_step_parameters(self, grad_vector):
        """
        computes \theta-\delta\theta
        :param grad_vector:
        """
        grad_dims = []
        for param in self.model.parameters():
            grad_dims.append(param.data.numel())

        new_model = copy.deepcopy(self.model)
        self.overwrite_grad(new_model.parameters, grad_vector, grad_dims)
        with torch.no_grad():
            for param in new_model.parameters():
                if param.grad is not None:
                    param.data = param.data - self.lr * param.grad.data
        return new_model

    def overwrite_grad(self, pp, new_grad, grad_dims):
        """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
        """
        cnt = 0
        for param in pp():
            param.grad = torch.zeros_like(param.data)
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = new_grad[beg:en].contiguous().view(param.data.size())
            param.grad.data.copy_(this_grad)
            cnt += 1
