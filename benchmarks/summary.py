"""Summarizes benchmarking runs"""

import os
import ast
import math
import json
import numpy as np
import pandas as pd
from typing import List
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# --- Paths


def get_dataframes(
    dataset: str = "mnist",
    results: str = "results",
    csv_file: str = "test_result.csv",
    param: str = "args.json",
):
    """Returns param and result dataframe for all runs under a given dataset"""

    # --- Setup paths
    param_paths = get_paths(param, dataset, results)
    result_paths = get_paths(csv_file, dataset, results)

    # --- Create dataframes
    param_dict = pd.concat(
        [get_param_df(path, i) for i, path in enumerate(param_paths)]
    )
    result_df = pd.concat(
        [get_result_df(path, i) for i, path in enumerate(result_paths)]
    )

    return param_dict, result_df


def get_paths(
    fname: str,
    dataset: str = "mnist",
    results: str = "results",
):
    """Returns the path to all runs for a given dataset"""

    # --- Setup paths
    dataset = Path(dataset)
    results = Path(results)
    fname = Path(fname)
    path = dataset / results

    # --- Get paths
    paths = [os.path.join(path, run) for run in os.listdir(path)]
    paths = [os.path.join(path, fname) for path in paths]

    return sorted(paths)


def get_param_df(path: str, id: int):
    dct = json_load(path)
    dct["path"] = path
    dct["id"] = id
    return pd.DataFrame([dct])


def get_result_df(path: str, id: int):
    df = pd.read_csv(path)
    df["path"] = path
    df["id"] = id
    return df


def json_load(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


# --- Metrics


def yield_all_average_metrics_over_n_trials(
    params: List[pd.DataFrame],
    results: List[pd.DataFrame],
    metrics: List[str],
):
    """Yields average metrics over n_trials for all runs"""

    for param, result in zip(params, results):
        train_kwargs = param["train_kwargs"]
        train_kwargs = train_kwargs.astype(str).unique()

        # --- Loop over different dataset sampling hyperparameters
        for train_kwarg in train_kwargs:
            train_kwarg = ast.literal_eval(train_kwarg)
            df_filter = lambda df, dct: df[df["train_kwargs"] == dct]

            avg_metrics = average_metrics_over_n_trials(
                df_filter(param, train_kwarg), result, metrics
            )

            yield train_kwarg, avg_metrics


def average_metrics_over_n_trials(
    params: pd.DataFrame,
    results: pd.DataFrame,
    metrics: list,
):
    """Averages all metrics over n_trials"""

    return {
        metric: average_metric_over_n_trials(params, results, metric)[metric]
        for metric in metrics
    }


def average_metric_over_n_trials(
    params: pd.DataFrame,
    results: pd.DataFrame,
    metric: str,
):
    """Averages a single metric over n_trials"""

    ids = params["id"]
    df_filter = lambda df, i: df[df["id"] == i]
    average = np.mean([df_filter(results, i)[metric] for i in ids], axis=0)

    return {metric: average}


def yield_epoch_average_metrics_over_n_trials(
    params: List[pd.DataFrame],
    results: List[pd.DataFrame],
    metrics: List[str],
    epoch: int,
):
    """Yields average metrics over n_trials at a specific epoch"""

    for param, result in zip(params, results):
        train_kwargs = param["train_kwargs"]
        train_kwargs = train_kwargs.astype(str).unique()

        # --- Loop over different dataset sampling hyperparameters
        for train_kwarg in train_kwargs:
            train_kwarg = ast.literal_eval(train_kwarg)
            df_filter = lambda df, dct: df[df["train_kwargs"] == dct]

            avg_metrics = epoch_average_metrics_over_n_trials(
                df_filter(param, train_kwarg), result, metrics, epoch
            )

            yield train_kwarg, avg_metrics


def epoch_average_metrics_over_n_trials(
    params: pd.DataFrame,
    results: pd.DataFrame,
    metrics: list,
    epoch: int,
):
    """Averages all metrics over n_trials at a specifc epoch"""

    return {
        metric: epoch_average_metric_over_n_trials(params, results, metric, epoch)[
            metric
        ]
        for metric in metrics
    }


def epoch_average_metric_over_n_trials(
    params: pd.DataFrame,
    results: pd.DataFrame,
    metric: str,
    epoch: int,
):
    """Averages a single metric over n_trials at a specifc epoch"""

    average = average_metric_over_n_trials(params, results, metric)
    average[metric] = average[metric][epoch]

    return average


# --- Display


def show_experiments_at_epoch(
    params: List[pd.DataFrame],
    results: List[pd.DataFrame],
    metrics: List[str],
    epoch: int,
):
    """DataFrame of experiments showing all metrics averaged over n_trials at a specific epoch"""

    gen = yield_epoch_average_metrics_over_n_trials(
        params,
        results,
        metrics,
        epoch,
    )

    results = [{**avg_metrics, **train_kwarg} for train_kwarg, avg_metrics in gen]

    return pd.DataFrame(results)


def plot_experiments_as_grid(
    params: List[pd.DataFrame],
    results: List[pd.DataFrame],
    metrics: List[str],
    title: str,
    figsize=(16, 12),
    dpi=80,
):
    """Plot summary of experiments showing all metrics averaged over n_trials"""

    n = len(metrics)
    r, c = get_num_row_columns(n)

    fig, ax = plt.subplots(r, c, figsize=figsize, dpi=dpi)

    if not hasattr(ax, "__len__"):
        ax = np.array([ax])

    gen = yield_all_average_metrics_over_n_trials(
        params,
        results,
        metrics,
    )

    for train_kwarg, avg_metrics in gen:

        # --- Plot metrics
        for subplot, name in zip(ax.flatten(), avg_metrics):
            subplot.plot(avg_metrics[name], label=f"{train_kwarg}")
            subplot.set_xlabel("epoch")
            subplot.set_ylabel("value")
            subplot.set_title(name)

    handles, labels = subplot.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.5, -0.06),
        fancybox=True,
        shadow=True,
        ncol=2,
    )
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.show()


def plot_images_as_grid(
    datas,
    targets,
    preds,
    indices,
    cmap="gray",
    axes_pad=0.5,
    fontsize=14,
    figsize=(10, 10),
    labels=None,
    caption_bottom=-0.3,
):
    """Plot images annotated with model predictions and ground-truth"""

    n = len(indices)
    r, c = get_num_row_columns(n)

    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(r, c),
        axes_pad=axes_pad,
    )

    results = []

    for ax, data, target, pred, index in zip(
        grid,
        datas,
        targets,
        preds,
        indices,
    ):
        ax.imshow(data)
        target = int(target)
        pred = int(pred)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"({index}, {target}, {pred})")

        if labels:
            target = labels[target]
            pred = labels[pred]

        results.append(
            {"index": index, "correct": target == pred, "target": target, "pred": pred}
        )

    fig.suptitle(("(Index, Target, Pred)"), fontsize=fontsize)
    plt.show()

    return results


def get_num_row_columns(n):
    r = int(n ** 0.5)
    c = math.ceil(n / r)
    return r, c
