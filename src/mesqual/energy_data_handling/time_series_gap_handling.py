import numpy as np
import pandas as pd


class TimeSeriesGapHandler:
    """Inserts NaN values in gaps of a time series, so that line plots show the gap.

    A gap is a step longer than `max_gap_in_minutes`. The handler inserts one
    NaN value `max_gap_in_minutes` after the last value before the gap. A line
    plot then shows a break instead of a line that bridges the gap.

    Args:
        max_gap_in_minutes: Maximum allowed gap duration in minutes. Gaps longer than
            this threshold will have NaN values inserted. Default is 60 minutes.

    Examples:
        >>> dates = pd.date_range('2024-01-01', periods=5, freq='h').delete([2, 3])
        >>> TimeSeriesGapHandler(max_gap_in_minutes=90).insert_nans_at_gaps(pd.Series([1.0, 2.0, 3.0], index=dates))
        2024-01-01 00:00:00    1.0
        2024-01-01 01:00:00    2.0
        2024-01-01 02:30:00    NaN
        2024-01-01 04:00:00    3.0
        dtype: float64
    """

    def __init__(self, max_gap_in_minutes: float = 60):
        """Initialize the gap handler with specified maximum gap threshold.

        Args:
            max_gap_in_minutes: Maximum allowed gap duration in minutes before
                inserting NaN markers.
        """
        self._max_gap_in_minutes = max_gap_in_minutes

    def insert_nans_at_gaps(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        """Insert NaN values at the beginning of gaps that exceed the maximum threshold.

        Args:
            data: Time series data with DatetimeIndex. Can be either a Series or DataFrame.

        Returns:
            Without a gap, `data` itself. Else a new object of the same type
                with the NaN values inserted, sorted by index.

        Raises:
            TypeError: If data index is not a DatetimeIndex.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError(f'Data index must be DatetimeIndex, got {type(data.index)}')

        data = data.sort_index()
        diffs = data.index.to_series().diff()
        mask = diffs > pd.Timedelta(minutes=self._max_gap_in_minutes)
        gap_indices = np.where(mask)[0]

        if len(gap_indices) == 0:
            return data  # No gaps found

        new_timestamps = data.index[gap_indices - 1] + pd.Timedelta(minutes=self._max_gap_in_minutes)

        if isinstance(data, pd.Series):
            new_values = pd.Series(np.nan, index=new_timestamps, name=data.name)
        else:
            new_values = pd.DataFrame(np.nan, index=new_timestamps, columns=data.columns)

        return pd.concat([data, new_values]).sort_index()
