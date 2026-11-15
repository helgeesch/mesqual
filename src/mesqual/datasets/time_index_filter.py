"""Row filters on one level of the time index of a time series result."""

from __future__ import annotations

import datetime
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Range:
    """Inclusive range of index values.

    On a datetime level, `start` and `end` are dates (`datetime.date` or ISO
    date strings), which match every day from start to end, or `Timestamp`s,
    which match `start <= timestamp <= end`. On any other level, the range
    matches `start <= value <= end`.

    Attributes:
        start: First value of the range.
        end: Last value of the range.
    """

    start: Any
    end: Any


@dataclass(frozen=True)
class TimeIndexFilter:
    """Keeps the rows whose value on one level of the time index passes the filter.

    The filter acts on the time index of time series flags. `level` names a
    level of the time index, for example `hour` of `(day, hour)`.
    `None` means the datetime level. A result without that level is returned
    unchanged, so model flags pass through.

    A row passes when its value is in `include` (if set) and not in `exclude`
    (if set). `include` and `exclude` can mix plain values and `Range` objects.

    On a datetime level, a `datetime.date` or an ISO date string matches the
    local calendar date of the row, and a `Timestamp` matches that exact
    timestamp. On any other level, values match directly.

    On a `MultiIndex`, the other levels stay.

    Attributes:
        level: Level of the time index to filter. `None`: the datetime level.
        include: Values or ranges to keep. Put a tuple value in a list
            (`[(1, 2)]`), else its elements count as separate values.
        exclude: Values or ranges to drop. Same form as `include`.

    Examples:
        >>> march_without_29th = TimeIndexFilter.dates(include=[Range('2024-03-01', '2024-03-31')], exclude=['2024-03-29'])
        >>> prices = dataset.fetch('Node.Results.price', config={'time_index_filter': march_without_29th})
        >>> sorted(set(prices.index.day))
        [28, 30, 31]
        >>> first_day_periods = TimeIndexFilter(level='period', include=[Range(1, 24)])
    """

    level: str | None = None
    include: Collection | None = None
    exclude: Collection | None = None

    @classmethod
    def dates(cls, include: Collection | None = None, exclude: Collection | None = None) -> TimeIndexFilter:
        """Short form for a filter on calendar dates of the time level.

        Args:
            include: Dates, timestamps or ranges to keep.
            exclude: Dates, timestamps or ranges to drop.

        Returns:
            A filter on the time level of the result.
        """
        return cls(level=None, include=include, exclude=exclude)

    def apply(self, data: pd.Series | pd.DataFrame, time_level: str | int | None) -> pd.Series | pd.DataFrame:
        """Filter the rows of `data`.

        Args:
            data: Result to filter.
            time_level: Name or position of the time level of `data`, used
                when `level` is `None`. `None`: the result has no time level.

        Returns:
            The rows that pass the filter.
        """
        level = self.level if self.level is not None else time_level
        if level is None or not self._has_level(data.index, level):
            return data
        values = data.index.get_level_values(level)
        mask = np.ones(len(values), dtype=bool)
        if self.include is not None:
            mask &= self._matches(values, self.include)
        if self.exclude is not None:
            mask &= ~self._matches(values, self.exclude)
        return data.loc[mask]

    @staticmethod
    def _has_level(index: pd.Index, level: str | int) -> bool:
        if isinstance(level, int):
            return level < index.nlevels
        return level in index.names

    @classmethod
    def _matches(cls, values: pd.Index, items: Collection) -> np.ndarray:
        if isinstance(items, (str, Range, datetime.date)) or not isinstance(items, Collection):
            items = [items]
        if isinstance(values, pd.DatetimeIndex):
            return _DatetimeMatcher(values).matches(items)
        return _ValueMatcher(values).matches(items)


class _ValueMatcher:
    """Matches plain values and ranges against a non-datetime index level."""

    def __init__(self, values: pd.Index):
        self._values = values

    def matches(self, items: Collection) -> np.ndarray:
        mask = np.zeros(len(self._values), dtype=bool)
        plain = [item for item in items if not isinstance(item, Range)]
        if plain:
            mask |= self._values.isin(plain)
        for item in items:
            if isinstance(item, Range):
                mask |= self._in_range(item)
        return mask

    def _in_range(self, item: Range) -> np.ndarray:
        if self._values.dtype == object:
            return np.array([item.start <= value <= item.end for value in self._values], dtype=bool)
        return np.asarray((self._values >= item.start) & (self._values <= item.end), dtype=bool)


class _DatetimeMatcher:
    """Matches dates, timestamps and ranges against a datetime index level."""

    def __init__(self, values: pd.DatetimeIndex):
        self._values = values
        self._days = values.tz_localize(None).normalize() if values.tz is not None else values.normalize()

    def matches(self, items: Collection) -> np.ndarray:
        mask = np.zeros(len(self._values), dtype=bool)
        for item in items:
            mask |= self._match_one(item)
        return mask

    def _match_one(self, item: Any) -> np.ndarray:
        if isinstance(item, Range):
            return self._match_range(item)
        if self._is_timestamp(item):
            return np.asarray(self._values == self._timestamp(item), dtype=bool)
        return np.asarray(self._days == self._day(item), dtype=bool)

    def _match_range(self, item: Range) -> np.ndarray:
        start_is_timestamp, end_is_timestamp = self._is_timestamp(item.start), self._is_timestamp(item.end)
        if start_is_timestamp != end_is_timestamp:
            raise TypeError(f'A Range needs two dates or two Timestamps, got {item!r}.')
        if start_is_timestamp:
            start, end = self._timestamp(item.start), self._timestamp(item.end)
            return np.asarray((self._values >= start) & (self._values <= end), dtype=bool)
        start, end = self._day(item.start), self._day(item.end)
        return np.asarray((self._days >= start) & (self._days <= end), dtype=bool)

    @staticmethod
    def _is_timestamp(item: Any) -> bool:
        return isinstance(item, (datetime.datetime, np.datetime64))

    def _timestamp(self, item: datetime.datetime | np.datetime64) -> pd.Timestamp:
        timestamp = pd.Timestamp(item)
        if (timestamp.tz is None) != (self._values.tz is None):
            raise TypeError(
                f'Timestamp {timestamp} and the index (tz={self._values.tz}) must both be tz-aware or both tz-naive.'
            )
        return timestamp

    def _day(self, item: datetime.date | str) -> pd.Timestamp:
        if isinstance(item, str):
            item = datetime.date.fromisoformat(item)
        if not isinstance(item, datetime.date):
            raise TypeError(f'Expected a date, an ISO date string or a Timestamp, got {item!r}.')
        return pd.Timestamp(item)
