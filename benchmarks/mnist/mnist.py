"""MNIST Benchmark"""

import time
import json
import tqdm
import timeit
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

import sys

sys.path.append("../")
sys.path.append("../../datasetq")

import benchmark
from dataset import decorate_with_indices


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def run_benchmark(args, train_dataset, test_dataset, save_dir):
    train_kwargs = args["train_kwargs"]
    test_kwargs = args["test_kwargs"]
    n_trials = args["n_trials"]
    n_epochs = args["epochs"]
    device = args["device"]
    num_workers = args["num_workers"]
    seed = args["seed"]
    lr = args["lr"]
    gamma = args["gamma"]

    train_results = []
    test_results = []

    for trial in tqdm.tqdm(range(n_trials), desc="trials", position=0):
        benchmark.set_seed(seed)
        train_kwargs.update({"seed": seed, "num_workers": num_workers})
        test_kwargs.update({"seed": seed, "num_workers": num_workers})

        # --- Setup...
        model = Net().to(device)
        loss_fn = F.nll_loss
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
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
            model, device, test_loader, loss_fn, dataset="mnist"
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


if __name__ == "__main__":
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser(description="PyTorch MNIST Benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "kmnist", "fashion_mnist"],
        help="dataset to use (default: mnist)",
    )
    parser.add_argument(
        "--train-kwargs",
        type=json.loads,
        default='{"batch_size": 64, "sampler": false, "shuffle": true}',
        metavar="K",
        help='train dataloader and sampler keyword arguments (default: {"batch_size": 64, "sampler": false, "shuffle": true)',
    )
    parser.add_argument(
        "--test-kwargs",
        type=json.loads,
        default='{"batch_size": 1000, "sampler": false, "shuffle": false}',
        metavar="K",
        help='test dataloader and sampler keyword arguments (default: {"batch_size": 1000, "sampler": false, "shuffle": false})',
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
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training (default: False)",
    )
    parser.add_argument(
        "--num-workers",
        default=1,
        type=int,
        help="number of workers for dataloading. (default: 1)",
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

    dataset = {
        "mnist": datasets.MNIST,
        "kmnist": datasets.KMNIST,
        "fashion_mnist": datasets.FashionMNIST,
    }[args["dataset"]]

    dataset = decorate_with_indices(dataset)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = dataset(
        args["dataset"],
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = dataset(
        args["dataset"],
        train=False,
        transform=transform,
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
