import warnings
from functools import wraps

import numpy as np
import pandas as pd


class GranularityError(Exception):
    """Raised in strict mode when an index has no single step length.

    `TimeSeriesGranularityAnalyzer` raises it for duplicate timestamps, mixed
    step lengths within a day or across the index, or a day with one row.
    `CongestionRentCalculator` raises it again with a hint to pass
    `granularity_hrs`.
    """


def _validate_dt_index(func):
    @wraps(func)
    def wrapper(self, dt_index, *args, **kwargs):
        if not isinstance(dt_index, pd.DatetimeIndex):
            try:
                dt_index = pd.DatetimeIndex(dt_index)
            except Exception as e:
                raise TypeError(f'Input must be convertible to DatetimeIndex, got {type(dt_index)}') from e
        return func(self, dt_index, *args, **kwargs)

    return wrapper


class TimeSeriesGranularityAnalyzer:
    """Analyzes and validates time granularity in DatetimeIndex sequences.

    It detects the granularity of a time series, also per day. A series can
    have mixed granularities; strict mode (the default) raises an error for
    them, the other mode warns.

    Args:
        strict_mode: If True, raises GranularityError for multiple granularities,
            duplicate timestamps and days with one row. If False, issues
            warnings, and a day with one row gives NaT. Default is True.

    Examples:
        >>> analyzer = TimeSeriesGranularityAnalyzer()
        >>> index = pd.date_range('2024-01-01', periods=24, freq='h')
        >>> analyzer.get_granularity_as_timedelta(index)
        Timedelta('0 days 01:00:00')
    """

    def __init__(self, strict_mode: bool = True):
        self._strict_mode = strict_mode

    @property
    def strict_mode(self) -> bool:
        """If True, the analyzer raises `GranularityError` where it else issues a warning. Can be set."""
        return self._strict_mode

    @strict_mode.setter
    def strict_mode(self, value: bool):
        self._strict_mode = value

    @_validate_dt_index
    def get_granularity_as_series_of_timedeltas(self, dt_index: pd.DatetimeIndex) -> pd.Series:
        """Return the step length of each row, detected per day.

        The step of a row is the gap to the next row of the same day; the
        last row of a day takes the step before it. A day is the calendar
        date in the timezone of the index. The index may be unsorted: the
        steps are computed on the sorted unique timestamps and returned in
        the order of `dt_index`.

        Without strict mode, duplicate timestamps and mixed step lengths
        issue a warning. The first occurrence of a timestamp in `dt_index`
        gets its step, and each later occurrence gets 0, so a sum of the
        steps counts each timestamp once. A day with one row gives `NaT`.

        Args:
            dt_index: Timestamps, or values that convert to a `DatetimeIndex`.

        Returns:
            Series of `Timedelta` with `dt_index` as index. Empty for an empty index.

        Raises:
            ValueError: The index contains `NaT`.
            GranularityError: Strict mode, and the index has duplicate
                timestamps, a day has mixed step lengths, or a day has one row.
        """
        if dt_index.hasnans:
            raise ValueError('Granularity detection needs an index without NaT.')
        if len(dt_index) == 0:
            return pd.Series(index=dt_index, dtype='timedelta64[ns]')
        if not dt_index.is_unique:
            self._raise_or_warn('The index has duplicate timestamps.')
        timestamps = dt_index.unique().sort_values()
        steps = pd.Series(index=timestamps, dtype=float)
        steps = steps.groupby(timestamps.date).apply(lambda g: self._get_granularity_for_day(g.index))
        step_per_timestamp = pd.Series(steps.to_numpy(dtype='timedelta64[ns]'), index=timestamps)
        in_input_order = step_per_timestamp.reindex(dt_index).to_numpy(copy=True)
        in_input_order[dt_index.duplicated(keep='first')] = np.timedelta64(0, 'ns')
        return pd.Series(in_input_order, index=dt_index)

    @_validate_dt_index
    def get_granularity_as_series_of_minutes(self, dt_index: pd.DatetimeIndex) -> pd.Series:
        """Return the step length of each row in minutes, NaN for a step of `NaT`.

        The rules of `get_granularity_as_series_of_timedeltas` apply.
        """
        return self.get_granularity_as_series_of_timedeltas(dt_index).dt.total_seconds() / 60

    @_validate_dt_index
    def get_granularity_as_series_of_hours(self, dt_index: pd.DatetimeIndex) -> pd.Series:
        """Return the step length of each row in hours, NaN for a step of `NaT`.

        The rules of `get_granularity_as_series_of_timedeltas` apply.
        """
        return self.get_granularity_as_series_of_timedeltas(dt_index).dt.total_seconds() / 3600

    def _get_granularity_for_day(self, dt_index: pd.DatetimeIndex) -> pd.Series:
        gran = dt_index.to_series().diff().shift(-1)
        if len(gran) > 1:
            gran.iloc[-1] = gran.iloc[-2]
        elif self._strict_mode:
            raise GranularityError(f'The day {dt_index[0].date()} has one row, so it has no step length.')
        if gran.nunique() > 1:
            self._raise_or_warn(f'Multiple granularities identified within a day: {gran.unique()}.')
        return gran

    def _raise_or_warn(self, msg: str) -> None:
        if self._strict_mode:
            raise GranularityError(msg)
        warnings.warn(msg)

    @_validate_dt_index
    def get_granularity_as_timedelta(self, dt_index: pd.DatetimeIndex) -> pd.Timedelta:
        """Return the one step length of the index.

        The reference step is the step of the earliest timestamp, so the
        result does not depend on the row order. Repeated timestamps do not
        count as another step length. Without strict mode, a reference
        step of `NaT` (the earliest day has one row) issues a warning and
        the result is `NaT`. The rules of
        `get_granularity_as_series_of_timedeltas` apply.

        Args:
            dt_index: Timestamps, or values that convert to a `DatetimeIndex`.

        Returns:
            The step length. `Timedelta(0)` for an empty index.

        Raises:
            ValueError: The index contains `NaT`.
            GranularityError: Strict mode, and the index has more than one
                step length, or a rule of `get_granularity_as_series_of_timedeltas` fails.
        """
        if len(dt_index) == 0:
            return pd.Timedelta(0)
        gran = self.get_granularity_as_series_of_timedeltas(dt_index)
        reference = gran.iloc[dt_index.argmin()]
        if pd.isna(reference):
            self._raise_or_warn('The earliest timestamp has no step length, so the granularity is NaT.')
        if gran[~dt_index.duplicated()].nunique() > 1:
            self._raise_or_warn(
                f'Multiple granularities found: {gran.unique()}. Using {reference} as the reference granularity.'
            )
        return reference

    @_validate_dt_index
    def get_granularity_as_hours(self, dt_index: pd.DatetimeIndex) -> float:
        """Return the one step length of the index in hours, NaN for `NaT`.

        The rules of `get_granularity_as_timedelta` apply.
        """
        return self.get_granularity_as_timedelta(dt_index).total_seconds() / 3600

    @_validate_dt_index
    def get_granularity_as_minutes(self, dt_index: pd.DatetimeIndex) -> float:
        """Return the one step length of the index in minutes, NaN for `NaT`.

        The rules of `get_granularity_as_timedelta` apply.
        """
        return self.get_granularity_as_timedelta(dt_index).total_seconds() / 60

    @_validate_dt_index
    def validate_constant_granularity(self, dt_index: pd.DatetimeIndex, expected_hours: float) -> bool:
        """Return True if the step length of the index is `expected_hours`, within 1e-10 hours.

        The step length is `get_granularity_as_hours`. In strict mode, an
        index with more than one step length raises `GranularityError`.
        Without strict mode, it issues a warning, and only the step of the
        earliest timestamp is compared.
        """
        actual_hours = self.get_granularity_as_hours(dt_index)
        return abs(actual_hours - expected_hours) < 1e-10
