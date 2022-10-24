import heapq


class HeapQ:
    """
    Heap Queue implementing a subset of the original operations

    NOTE: By default, this class assumes that the unique identifier for each
    item inserted into the heap appears in the last index of the tuple
    """

    def __init__(self):
        self.heap = []
        self.dict = {}

    def __len__(self):
        return len(self.heap)

    def __getitem__(self, index: int):
        return self.heap[index]

    def heappush(self, item, id):
        """Push non-dirty item onto heap, maintaining the heap invariant."""

        if id in self.dict:
            self.remove(id)

        self.dict[id] = item
        heapq.heappush(self.heap, item)

    def heappop(self):
        """Pop smallest non-dirty item off heap, maintaining the heap invariant."""

        while self.heap:
            node = heapq.heappop(self.heap)
            if not node.dirty:
                del self.dict[node.id]
                return node

    def heappushpop(self, item, id):
        """Same as a heappush followed by a heappop, returns item if heap is empty."""

        if self.heap:
            if self.heap[0] < item or self.heap[0].dirty:
                self.heappush(item, id)
                node = self.heappop()
                if node:
                    item = node

        return item

    def clean(self):
        """Remove dirty items from heap"""

        self.heap = [item for item in self.heap if not item.dirty]
        heapq.heapify(self.heap)

    def remove(self, id):
        node = self.dict.pop(id)
        node.dirty = True

    def pop(self):
        node = self.heap.pop()
        return node

    def flush(self):
        self.heap = []
        self.dict = {}
