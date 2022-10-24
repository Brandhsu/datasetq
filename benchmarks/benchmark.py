# https://pytorch.org/docs/stable/notes/randomness.html

import os
import sys
import json
import torch
import numpy
import random
import pandas as pd

###########################################################################################

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "metric"))
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "datasetq"
    )
)

from metric import scoring_metric
from sampler import HeapqSampler


###########################################################################################


# --- Seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


# --- Data


def get_loader(dataset, batch_size, sampler, shuffle, seed, num_workers, **kwargs):
    generator = None

    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    if sampler:
        sampler = HeapqSampler(
            dataset, **{**kwargs, **{"shuffle": shuffle, "generator": generator}}
        )
        shuffle = False
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return dataloader


# --- Training
def to_gpu(data, target, device):
    if type(data) == tuple:
        data = torch.stack(data, axis=0)

    if type(target) == tuple:
        target = torch.LongTensor(target)

    return data.to(device), target.to(device)


def step(model, data, target, loss_fn):
    output = model(data)
    loss = loss_fn(output, target, reduction="none")

    return loss, output


def masked_batch(output, target):
    pred = torch.argmax(output, axis=-1)
    return pred != target


def train(model, device, train_loader, loss_fn, optimizer):
    model.train()
    train_loss = 0

    for i, batch in train_loader:
        data, target = batch
        data, target = to_gpu(data, target, device)
        loss, output = step(model, data, target, loss_fn)

        optimizer.zero_grad()
        torch.mean(loss).backward()
        optimizer.step()

        train_loss += torch.sum(loss).item()

        if hasattr(train_loader.sampler, "heapq"):
            train_loader.sampler.update(i.tolist(), {"loss": loss.tolist()})

    metrics = {}
    metrics["loss"] = train_loss / len(train_loader.dataset)

    return metrics


def test(model, device, test_loader, loss_fn, dataset):
    model.eval()
    y_true = []
    y_pred = []
    test_loss = 0

    with torch.no_grad():
        for i, batch in test_loader:
            data, target = batch
            data, target = to_gpu(data, target, device)
            loss, output = step(model, data, target, loss_fn)

            test_loss += torch.sum(loss).item()

            y_true += target.tolist()
            y_pred += output.tolist()

            if hasattr(test_loader.sampler, "heapq"):
                test_loader.sampler.update(i.tolist(), {"loss": loss.tolist()})

    metrics = {}
    metrics["loss"] = test_loss / len(test_loader.dataset)
    metrics.update(scoring_metric(dataset, y_true, y_pred))

    return metrics


# --- Logging


def log(trial, epoch, result, fname, time=None):
    rows = []

    if type(result) == dict:
        result = [result]

    for res in result:
        row = format(trial, epoch, res, time)
        rows.append(row)

    df = pd.DataFrame(rows)
    to_csv(df, fname)


def format(trial, epoch, result, time=None):
    row = {}

    row["trial"] = trial
    row["epoch"] = epoch
    row.update(**result)

    if time:
        row["time"] = time

    return row


def to_csv(df, fname):
    with open(fname, "a") as f:
        df.to_csv(f, mode="a", header=not f.tell(), index=False)


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False
