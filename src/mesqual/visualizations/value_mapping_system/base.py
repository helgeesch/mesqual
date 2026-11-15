from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


def is_missing(value: Any) -> bool:
    """Return True if `value` is a missing value.

    Missing values are the scalars that pandas counts as missing: `None`,
    `NaN`, `pd.NA`, `pd.NaT` and `np.datetime64("NaT", "ns")`. A list, tuple or array
    is not a missing value, also when it holds missing values.

    Args:
        value: Value to check.

    Returns:
        True if `value` is a missing scalar.
    """
    return pd.api.types.is_scalar(value) and bool(pd.isna(value))


class BaseMapping(ABC):
    """Base class of the value mappings: a call maps one input value to one output."""

    @abstractmethod
    def __call__(self, value: Any) -> Any:
        """Return the output for `value`."""
