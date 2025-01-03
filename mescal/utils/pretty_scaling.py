from typing import Iterable
from collections import Counter
import math
import numpy as np


def _all_bool(values: Iterable) -> bool:
    return all(isinstance(v, bool) for v in values)


def get_pretty_min_max(
        values: Iterable[float | int | bool],
        lower_percentile: float = 1,
        upper_percentile: float = 99
) -> tuple[float, float]:

    if _all_bool(values):
        return 0, 1

    min_val = np.percentile(list(values), lower_percentile)
    max_val = np.percentile(list(values), upper_percentile)

    range_val = max_val - min_val

    if np.isnan(range_val):
        return -1, 1

    intervals = [
        (10, 100),
        (20, 200),
        (20, 500),
        (50, 1000),
        (100, 2000),
        (500, 5000),
        (1000, float('inf'))
    ]

    rounding_interval = next((interval for interval, upper_bound in intervals if range_val <= upper_bound), 1000)

    pretty_min = np.floor(min_val / rounding_interval) * rounding_interval
    pretty_max = np.ceil(max_val / rounding_interval) * rounding_interval

    if symmetric_scaling_around_0_seems_appropriate(values):
        abs_max = max([abs(pretty_min), pretty_max])
        pretty_max = abs_max
        pretty_min = -abs_max

    return pretty_min, pretty_max


def symmetric_scaling_around_0_seems_appropriate(values: Iterable[float | int]) -> bool:
    values = np.array(values)

    min_value = min(values)
    max_value = max(values)

    if not ((min_value < 0) and (max_value > 0)):
        return False

    if (sum(values) / sum(abs(values))) < 0.1:
        return True

    abs_max = max([abs(min_value), max_value])

    if ((max_value - abs(min_value)) / abs_max) < 0.1:
        return True

    return False


def get_pretty_order_of_mag(
        values: Iterable[float | int | bool],
) -> float:

    if _all_bool(values):
        return 1

    def pretty_order_of_mag(num):
        magnitude = math.floor(math.log10(abs(num)))
        magnitude = magnitude - magnitude % 3
        return 10 ** magnitude

    order_counts = Counter([pretty_order_of_mag(v) for v in values])
    most_common = order_counts.most_common(1)[0][0]
    return most_common


def get_pretty_num_of_decimals(values: Iterable[float | int | bool], order_of_mag: float = None) -> int:

    if _all_bool(values):
        return 0

    if order_of_mag is None:
        order_of_mag = get_pretty_order_of_mag(values)

    scaled_values = [v / order_of_mag for v in values]
    abs_max_value = max(scaled_values)
    abs_min_value = min(scaled_values)
    spread = (abs_max_value - abs_min_value)
    if spread < 0.1:
        return 3
    if spread < 1:
        return 2
    if spread < 10:
        return 1
    if abs_max_value > 100:
        return 0
    if abs_max_value > 10:
        return 1
    if abs_max_value > 1:
        return 2
    return 2
