def calibrate_metric(
    dct: dict,
    extrema: str = "min",
    dtype: type = float,
):
    """Calibrates metric such that heap is ordered properly based on objective {min, max}"""

    metric, other = partition(dct, dtype)
    metric = {"min": _min, "max": _max}[extrema](metric)
    return {**metric, **other}


def partition(dct: dict, dtype):
    metric = {k: v for k, v in dct.items() if is_metric(v, dtype)}
    other = {k: v for k, v in dct.items() if not is_metric(v, dtype)}

    return metric, other


def is_metric(value, dtype):
    return (False, True)[type(value) == dtype]


def _min(dct: dict):
    return {k: -v for k, v in dct.items()}


def _max(dct: dict):
    return {k: v for k, v in dct.items()}
