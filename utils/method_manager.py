"""
rainbow-memory
Copyright 2021-present NAVER Corp.

Modifications copyright 2024 Mohammad Wasil

Licensed under the GPLv3. See LICENSE file for full text.
"""
import logging

from methods.finetune import Finetune
from methods.gdumb import GDumb
from methods.rainbow_memory import RM
from methods.joint import Joint
from methods.regularization import RWalk
from methods.naive import Naive
from methods.viewpoint_memory import ViewPointMemory
from methods.mir import MIR

logger = logging.getLogger()


def select_method(args, criterion, device, train_transform, test_transform, n_classes):
    kwargs = vars(args)
    # finetune resets the prev task model
    if args.mode == "finetune":
        method = Finetune(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    # naive uses prev task model and fine tune it w/o memory
    elif args.mode == "naive":
        method = Naive(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "joint":
        method = Joint(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "gdumb":
        method = GDumb(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "rm":
        method = RM(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "viewpoint":
        method = ViewPointMemory(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "rwalk":
        method = RWalk(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "mir":
        method = MIR(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    else:
        raise NotImplementedError("Choose the args.mode in [finetune, gdumb, rm, joint, naive, rwalk, viewpoint, mir]")

    logger.info(f"Method: {args.mode}")
    return method
