"""ImageNet Benchmark"""

import os
import json
import shutil
import time
import warnings
from enum import Enum

import tqdm
import timeit
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import sys

sys.path.append("../")
sys.path.append("../../datasetq")

import benchmark
from dataset import decorate_with_indices

# TODO: Add best model logging from https://github.com/pytorch/examples/blob/main/imagenet/main.py


def get_model(args):
    if args["pretrained"]:
        print("=> using pre-trained model '{}'".format(args["arch"]))
        model = models.__dict__[args["arch"]](pretrained=True)
    else:
        print("=> creating model '{}'".format(args["arch"]))
        model = models.__dict__[args["arch"]]()

    return model


def run_benchmark(args, train_dataset, test_dataset, save_dir):
    train_kwargs = args["train_kwargs"]
    test_kwargs = args["test_kwargs"]
    n_trials = args["n_trials"]
    n_epochs = args["epochs"]
    device = args["device"]
    num_workers = args["num_workers"]
    seed = args["seed"]
    lr = args["lr"]
    momentum = args["momentum"]
    weight_decay = args["weight_decay"]
    gamma = args["gamma"]

    train_results = []
    test_results = []

    for trial in tqdm.tqdm(range(n_trials), desc="trials", position=0):
        benchmark.set_seed(seed)
        train_kwargs.update({"seed": seed, "num_workers": num_workers})
        test_kwargs.update({"seed": seed, "num_workers": num_workers})

        # --- Setup...
        model = get_model(args).to(device)
        loss_fn = F.cross_entropy
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        scheduler = StepLR(optimizer, step_size=30, gamma=gamma)
        train_loader = benchmark.get_loader(train_dataset, **train_kwargs)
        test_loader = benchmark.get_loader(test_dataset, **test_kwargs)

        results = fit(
            args,
            model,
            train_loader,
            test_loader,
            loss_fn,
            optimizer,
            scheduler,
            n_epochs,
            trial,
            save_dir,
        )

        train_result, test_result = results
        train_results.append(train_result)
        test_results.append(test_result)

    return train_results, test_results


def fit(
    args,
    model,
    train_loader,
    test_loader,
    loss_fn,
    optimizer,
    scheduler,
    n_epochs,
    trial,
    save_dir,
):
    device = args["device"]

    train_results = []
    test_results = []

    for epoch in tqdm.tqdm(range(n_epochs), desc="epochs", position=1, leave=False):
        start = timeit.default_timer()

        train_result = benchmark.train(model, device, train_loader, loss_fn, optimizer)
        test_result = benchmark.test(
            model, device, test_loader, loss_fn, dataset="imagenet"
        )

        end = timeit.default_timer()

        benchmark.log(
            trial,
            epoch,
            train_result,
            save_dir / "train_result.csv",
            end - start,
        )
        benchmark.log(
            trial,
            epoch,
            test_result,
            save_dir / "test_result.csv",
            end - start,
        )

        train_results.append(train_result)
        test_results.append(test_result)

        scheduler.step()

    if hasattr(train_loader.sampler, "get_history"):
        benchmark.log(
            trial,
            epoch,
            train_loader.sampler.get_history().to_dict("records"),
            save_dir / "train_cache_map.csv",
        )
    if hasattr(test_loader.sampler, "get_history"):
        benchmark.log(
            trial,
            epoch,
            test_loader.sampler.get_history().to_dict("records"),
            save_dir / "test_cache_map.csv",
        )

    torch.save(model, save_dir / "model.pth")

    return train_results, test_results


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


if __name__ == "__main__":
    import argparse
    from pprint import pprint

    model_names = sorted(
        name
        for name in models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(models.__dict__[name])
    )

    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument(
        "--dataset",
        metavar="DIR",
        default="imagenet",
        help="path to dataset (default: imagenet)",
    )
    parser.add_argument(
        "--arch",
        metavar="ARCH",
        default="resnet18",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
    )
    parser.add_argument(
        "--train-kwargs",
        type=json.loads,
        default='{"batch_size": 256, "sampler": false, "shuffle": true}',
        metavar="K",
        help='train dataloader and sampler keyword arguments (default: {"batch_size": 256, "sampler": false, "shuffle": true)',
    )
    parser.add_argument(
        "--test-kwargs",
        type=json.loads,
        default='{"batch_size": 256, "sampler": false, "shuffle": false}',
        metavar="K",
        help='test dataloader and sampler keyword arguments (default: {"batch_size": 256, "sampler": false, "shuffle": false})',
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=1,
        metavar="N",
        help="number of benchmarking trials (default: 1)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run (default: 90)",
    )
    parser.add_argument(
        "--lr",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate (default: 0.1)",
        dest="lr",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        metavar="M",
        help="learning rate step gamma (default: 0.1)",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        metavar="M",
        help="momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="use pre-trained model",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training (default: False)",
    )
    parser.add_argument(
        "--num-workers",
        default=6,
        type=int,
        help="number of workers for dataloading. (default: 6)",
    )
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="seed for initializing training. (default: 1)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        metavar="S",
        help="directory of saved training results (default: results)",
    )

    ###############################################################################

    args = parser.parse_args()
    args = vars(args)

    use_cuda = not args["no_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    args["device"] = device

    cudnn.benchmark = True

    traindir = os.path.join(args["dataset"], "train")
    valdir = os.path.join(args["dataset"], "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    ImageFolder = decorate_with_indices(datasets.ImageFolder)

    train_dataset = ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    test_dataset = ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    save_dir = Path(args["dataset"] + "_" + args["save_dir"]).resolve() / time.strftime(
        "%Y-%m-%d_%H-%M-%S_%Z", time.localtime()
    )

    Path(save_dir).resolve().mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(save_dir / "args.json", "w") as f:
        json.dump(
            {k: v if benchmark.is_jsonable(v) else str(v) for k, v in args.items()}, f
        )

    train_results, test_results = run_benchmark(
        args, train_dataset, test_dataset, save_dir
    )

    print("train results:")
    pprint(train_results)
    print()
    print("test results:")
    pprint(test_results)
