from typing import Iterable


def all_same_object(items: Iterable) -> bool:
    tmp = list(items)
    return all(x is tmp[0] for x in items)
