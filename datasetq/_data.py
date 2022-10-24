"""Defines data (tuple) schema"""

import dataclasses


def make_dataclass(cls_name, fields):
    return dataclasses.make_dataclass(
        cls_name,
        fields,
        namespace={"asdict": lambda self: dataclasses.asdict(self)},
        order=True,
    )


Item = make_dataclass("Item", ["iter", "visits", "evict", "dirty", "id"])
Priority = lambda fields: make_dataclass("Priority", fields)
