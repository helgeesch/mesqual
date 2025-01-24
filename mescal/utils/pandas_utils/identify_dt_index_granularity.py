import warnings
from functools import wraps
import pandas as pd


def _validate_dt_index(func):
    @wraps(func)
    def wrapper(dt_index, *args, **kwargs):
        if not isinstance(dt_index, pd.DatetimeIndex):
            try:
                dt_index = pd.DatetimeIndex(dt_index)
            except Exception as e:
                raise TypeError(f"Input must be convertible to DatetimeIndex, got {type(dt_index)}") from e
        return func(dt_index, *args, **kwargs)
    return wrapper


@_validate_dt_index
def get_granularity_as_series_of_timedeltas(dt_index: pd.DatetimeIndex) -> pd.Series:
    gran = dt_index.to_series().diff().shift(-1)
    if len(gran) > 1:
        gran.iloc[-1] = gran.iloc[-2]
    return gran


_warned_about_granularity = False


def get_granularity_as_timedelta(dt_index: pd.DatetimeIndex) -> pd.Timedelta:
    global _warned_about_granularity
    gran = get_granularity_as_series_of_timedeltas(dt_index)
    if len(dt_index) == 0:
        return pd.Timedelta(0)
    first_gran = gran.iloc[0]
    if len(gran.unique()) > 1 and not _warned_about_granularity:
        warnings.warn(
            f'Found multiple granularities in DatetimeIndex. '
            f'Either your Index has gaps, or the index has an inconsistent granularity.'
            f'Using {first_gran} as the index granularity. Other timedeltas present are: {gran.unique()}'
        )
        _warned_about_granularity = True
    return first_gran


def get_granularity_in_hrs(dt_index: pd.DatetimeIndex) -> float:
    granularity = get_granularity_as_timedelta(dt_index)
    return granularity.total_seconds() / 3600


if __name__ == '__main__':
    test_indices = {
        'hourly': pd.date_range('2024-01-01', periods=24, freq='h'),
        'quarter_hourly': pd.date_range('2024-01-01', periods=96, freq='15min'),
        'five_min': pd.date_range('2024-01-01', periods=288, freq='5min')
    }

    for name, index in test_indices.items():
        timedelta = get_granularity_as_timedelta(index)
        hours = get_granularity_in_hrs(index)
        print(f"{name}: {timedelta} ({hours}h)")
