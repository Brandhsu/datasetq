"""Helpers to organize, present, and verify data"""


import pandas as pd
from ._heapq import HeapQ
from ._verify import is_heap
from ._metric import calibrate_metric


def _namedtuple_to_dict(tup: tuple):
    return {**tup._asdict()}


def _to_dataframe(dct: dict):
    return pd.DataFrame([v for k, v in dct.items()], index=dct.keys())


def get_history(history: dict, extrema: str = "min"):
    history = {k: calibrate_metric(v, extrema) for k, v in history.items()}
    return _to_dataframe(history)


def get_evicted(history: dict, extrema: str = "min"):
    history = {
        k: calibrate_metric(v, extrema) for k, v in history.items() if v["evict"]
    }
    return _to_dataframe(history)


def get_heap(heap: HeapQ, extrema: str = "min"):
    heap = {
        i: calibrate_metric(_namedtuple_to_dict(v), extrema) for i, v in enumerate(heap)
    }
    return _to_dataframe(heap)


def is_heap(heap: HeapQ, extrema: str = "min"):
    return is_heap(heap, extrema)
