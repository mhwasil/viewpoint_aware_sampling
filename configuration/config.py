"""
rainbow-memory
Copyright 2021-present NAVER Corp.

Modifications copyright 2024 Mohammad Wasil

Licensed under the GPLv3. See LICENSE file for full text.
"""
import argparse


def base_parser():
    parser = argparse.ArgumentParser(description="Class Incremental Learning Research")

    # Mode and Exp. Settings.
    parser.add_argument(
        "--mode",
        type=str,
        default="finetune",
        help="CIL methods [joint, rwalk, icarl, rm,  gdumb, ewc, bic]",
    )
    parser.add_argument(
        "--mem_manage",
        type=str,
        default=None,
        help="memory management [default, random, reservoir, uncertainty, prototype]. If None (default) no memory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="[mnist, cifar10, cifar100, imagenet]",
    )
    parser.add_argument("--n_tasks", type=int, default="5", help="The number of tasks")
    parser.add_argument(
        "--n_cls_a_task", type=int, default=2, help="The number of class of each task"
    )
    parser.add_argument(
        "--n_init_cls",
        type=int,
        default=1,
        help="The number of classes of initial task",
    )
    parser.add_argument("--rnd_seed", type=int, help="Random seed number.")
    parser.add_argument(
        "--memory_size", type=int, default=500, help="Episodic memory size"
    )
    parser.add_argument(
        "--stream_env",
        type=str,
        default="offline",
        choices=["offline", "online"],
        help="the restriction whether to keep streamed data or not",
    )

    # Dataset
    parser.add_argument(
        "--log_path",
        type=str,
        default="results",
        help="The path logs are saved. Only for local-machine",
    )

    # Model
    parser.add_argument(
        "--model_name", type=str, default="resnet18small", help="[resnet18small]"
    )
    parser.add_argument("--pretrain", action="store_true", help="pretrain model or not")

    # Train
    parser.add_argument("--opt_name", type=str, default="sgd", help="[adam, sgd]")
    parser.add_argument("--sched_name", type=str, default="cos", help="[cos, anneal]")
    parser.add_argument("--batchsize", type=int, default=32, help="batch size")
    parser.add_argument("--n_epoch", type=int, default=1, help="Epoch")

    parser.add_argument("--n_worker", type=int, default=2, help="The number of workers")

    parser.add_argument("--lr", type=float, default=0.011, help="learning rate")
    parser.add_argument(
        "--initial_annealing_period",
        type=int,
        default=20,
        help="Initial Period that does not anneal",
    )
    parser.add_argument(
        "--annealing_period",
        type=int,
        default=20,
        help="Period (Epochs) of annealing lr",
    )
    parser.add_argument(
        "--learning_anneal", type=float, default=10, help="Divisor for annealing"
    )
    parser.add_argument(
        "--init_model",
        action="store_false",
        help="Initilize model parameters for every iterations",
    )
    parser.add_argument(
        "--init_opt",
        action="store_false",
        help="Initilize optimizer states for every iterations",
    )
    parser.add_argument(
        "--topk", type=int, default=1, help="set k when we want to set topk accuracy"
    )
    parser.add_argument(
        "--joint_acc",
        type=float,
        default=0.0,
        help="Accuracy when training all the tasks at once",
    )
    # Transforms
    parser.add_argument(
        "--transforms",
        nargs="*",
        default=[],
        help="Additional train transforms [cotmix, cutout, randaug]",
    )

    # Benchmark
    parser.add_argument("--exp_name", type=str, default="", help="[disjoint, blurry]")

    # ICARL
    parser.add_argument(
        "--feature_size",
        type=int,
        default=2048,
        help="Feature size when embedding a sample",
    )

    # BiC
    parser.add_argument(
        "--distilling",
        action="store_true",
        help="use distilling loss with classification",
    )

    # Regularization
    parser.add_argument(
        "--reg_coef",
        type=int,
        default=100,
        help="weighting for the regularization loss term",
    )

    # Uncertain
    parser.add_argument(
        "--uncert_metric",
        type=str,
        default="vr",
        choices=["vr", "vr1", "vr_randaug", "loss"],
        help="A type of uncertainty metric",
    )

    parser.add_argument(
        "--stream_task_0",
        type=str,
        default="online",
        choices=["offline", "online"],
        help="whether to train the task 0 offline or online (default)",
    )

    parser.add_argument(
        "--cuda_idx",
        type=int,
        default=None,
        help="cuda to use",
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="dataset root for each image, used when file_name in json file does not include data root",
    )

    parser.add_argument(
        "--save_model", action="store_true", help="Save model for each task"
    )
    # Debug
    parser.add_argument("--debug", action="store_true", help="Turn on Debug mode")

    parser.add_argument(
        "--finetune_with_buffer",
        action="store_true",
        help="Whether to finetune with buffer data",
    )

    # trained model path for evaluation
    parser.add_argument(
        "--trained_model_path",
        type=str,
        default=None,
        help="Trained model path for eval",
    )

    parser.add_argument(
        "--core50_run",
        type=str,
        default="run0",
        choices=[
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
        ],
        help="Core50 run from 0 to 8 (10 runs or different experiments)",
    )

    parser.add_argument(
        "--result_dir",
        type=str,
        default="./results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    return args
