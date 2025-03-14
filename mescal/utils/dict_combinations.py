from typing import Any, Iterable, Iterator
from itertools import product


def dict_combination_iterator(input_dict: dict[Any, Iterable[Any]]) -> Iterator[dict[Any, Any]]:
    keys = list(input_dict.keys())
    values = list(input_dict.values())

    for combination in product(*values):
        yield dict(zip(keys, combination))
