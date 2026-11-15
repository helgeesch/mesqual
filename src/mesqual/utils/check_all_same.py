"""Check whether all items are the same object."""

from typing import Iterable


def all_same_object(items: Iterable) -> bool:
    """`True` if all items are the same object. An empty input gives `True`. A generator input works."""
    items = list(items)
    return all(x is items[0] for x in items)
