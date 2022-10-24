"""Verifies heapq structure"""


def is_heap(arr: list, extrema: str = "min"):
    return {"min": is_minheap, "max": is_maxheap}[extrema](arr)


def is_minheap(arr: list):
    n = len(arr)
    for i in range(int((n - 2) / 2) + 1):
        if arr[2 * i + 1] < arr[i]:
            return False
        if 2 * i + 2 < n and arr[2 * i + 2] < arr[i]:
            return False

    return True


def is_maxheap(arr: list):
    n = len(arr)
    for i in range(int((n - 2) / 2) + 1):
        if arr[2 * i + 1] > arr[i]:
            return False
        if 2 * i + 2 < n and arr[2 * i + 2] > arr[i]:
            return False

    return True
