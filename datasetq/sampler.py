import math
from typing import Dict, List, Tuple, Sized
from torch.utils.data import Sampler, SequentialSampler, SubsetRandomSampler

import util
from _data import Item, Priority


class HeapqSampler(Sampler[int]):
    """Official implementation of heap queue dataset sampling in PyTorch

    Description:

    ============================================================================
    Example:
        from torchvision.datasets import MNIST
        from torch.utils.data import DataLoader
        from datasetq.dataset import decorate_with_indices
        from datasetq.sampler import HeapqSampler

        MNIST = decorate_with_indices(MNIST)
        dataset = MNIST(root, train=True)
        sampler = HeapqSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

        # for i, batch in dataloader:
        #     data, target = batch
        #     output = model(data)
        #     loss = loss_fun(output, target, reduction="none")
        #     dataloader.sampler.update(i.tolist(), {"loss": loss.tolist()})
    ============================================================================

    Args:
        data_source (Sized): PyTorch dataset
        extrema (str, optional): Defines {min, max} heap. Defaults to "min".
        size (int, optional): Size of heap. Defaults to None (Same size as dataset).
        steps (int, optional): Samples every n iteration(s) without heap (1 acts as a normal sampler). Defaults to None (Sample only from heap).
        start_at (int, optional): Epoch to start heap sampling. Defaults to None (Sample after the first epoch).
        max_visits (int, optional): Maximum frequency a sample is visited per epoch before heap eviction (forever). Defaults to None (Max visits is infinite).
        reset_every (int, optional): Reset heap and evicted every n iteration(s). Defaults to None (Reset heap after infinite iterations).
        shuffle (bool, optional): Shuffles dataset. Defaults to False (No dataset shuffling).
        generator (Generator, optional): Generator used in sampling, valid only when shuffle=True. Defaults to None.
    """

    def __init__(
        self,
        data_source: Sized,
        extrema: str = "min",
        size: int = None,
        steps: int = None,
        start_at: int = None,
        max_visits: int = None,
        reset_every: int = None,
        shuffle: bool = False,
        generator=None,
    ) -> None:

        assert size is None or size >= 0
        assert steps is None or steps > 0
        assert start_at is None or start_at > 0
        assert max_visits is None or max_visits > 0
        assert reset_every is None or reset_every > 0

        self.data_source = data_source
        self.extrema = extrema
        self.size = size if size else math.inf
        self.steps = steps if steps else math.inf
        self.start_at = start_at if start_at else 1
        self.max_visits = max_visits if max_visits else math.inf
        self.reset_every = reset_every if reset_every else math.inf
        self.shuffle = shuffle
        self.generator = generator

        self.n_iters = 0
        self.n_epochs = 0
        self.priority = None
        self.indices = [i for i in range(len(self))]
        self.visits = {i: 0 for i in self.indices}

        self.heapq = util.HeapQ()
        self.history = {
            i: Item(
                iter=0,
                visits=0,
                evict=False,
                dirty=False,
                id=i,
            ).asdict()
            for i in self.indices
        }

        self.base_sampler = SubsetRandomSampler if self.shuffle else SequentialSampler

    def __iter__(self):
        self.visits = {i: 0 for i in self.indices}
        indices = self.base_sampler.__iter__(self)

        for index in indices:
            yield self.get_index(index)

        self.heapq.clean()  # remove dirty items and rebuild heap in linear time
        assert self.is_heap()
        self.n_epochs += 1

    def __len__(self):
        return len(self.data_source)

    def get_index(self, index: int):
        hist = self.history[index]

        if not self.n_iters % self.reset_every:
            self.reset()

        if self.n_epochs >= self.start_at:
            if self.n_iters % self.steps:
                # NOTE: Might be too strict
                if hist["evict"]:
                    if self.heapq.heap:
                        root = self.heapq.heappop()
                        index = root.id
                        assert not root.dirty
                else:
                    node = self.priority(**hist)
                    root = self.heapq.heappushpop(node, hist["id"])
                    index = root.id
                    assert not root.dirty

        self.n_iters += 1
        hist["iter"] = self.n_iters

        return index

    def reset(self):
        self.heapq.flush()

        for k in self.history:
            self.history[k]["evict"] = False

    def update(self, index: int, metrics: Dict[str, List[float]]):
        metrics = [dict(zip(metrics, value)) for value in zip(*metrics.values())]
        self.update_batch([index, metrics])

    def update_batch(self, batch: List[Tuple[List[int], Dict[str, float]]]):
        for sample in zip(*batch):
            self.update_sample(sample)

    def update_sample(self, sample: Tuple[int, Dict[str, float]]):
        index, metric = sample

        hist = self.history[index]
        hist["visits"] += 1
        hist = {k: v for k, v in hist.items() if k not in metric}
        self.history[index] = util.calibrate_metric({**metric, **hist}, self.extrema)

        if self.priority is None:
            self.priority = Priority(self.history[index].keys())

        self.overvisited(index)
        self.add(index)

    def overvisited(self, index: int):
        self.visits[index] += 1

        if self.visits[index] >= self.max_visits:
            self.history[index]["evict"] = True

    def add(self, index: int):
        if not self.history[index]["evict"]:
            hist = self.history[index]
            node = self.priority(**hist)
            self.heapq.heappush(node, hist["id"])

            if self.is_full():
                self.heapq.pop()

    def is_empty(self):
        return len(self.heapq) == 0

    def is_full(self):
        return (self.size - len(self.heapq)) < 0

    def get_history(self):
        return util.get_history(self.history, self.extrema)

    def get_evicted(self):
        return util.get_evicted(self.history, self.extrema)

    def get_heap(self):
        return util.get_heap(self.heapq.heap, self.extrema)

    def is_heap(self):
        return util.is_heap(self.heapq.heap, self.extrema)
