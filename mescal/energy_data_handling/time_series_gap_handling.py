import numpy as np
import pandas as pd


class TimeSeriesGapHandler:
    """Handles gaps in time series data by inserting NaN values at detected gaps.
    
    This class is useful for identifying and marking gaps in time series data that exceed
    a specified threshold. The handler inserts NaN values at the beginning of detected gaps
    to make gaps visible for downstream processing. This is particularly useful for line plots,
    where you often prefer to have a visual line-gap in a data gap instead of the line bridging
    the two surrounding values.
    
    Args:
        max_gap_in_minutes: Maximum allowed gap duration in minutes. Gaps longer than
            this threshold will have NaN values inserted. Default is 60 minutes.
    
    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> dates = pd.date_range('2024-01-01', periods=5, freq='30min')
        >>> dates = dates.delete(2)  # Create a gap
        >>> values = np.random.rand(len(dates))
        >>> series = pd.Series(values, index=dates)
        >>> handler = TimeSeriesGapHandler(max_gap_in_minutes=30)
        >>> result = handler.insert_nans_at_gaps(series)
        >>> print(result)  # Will show NaN inserted at gap location
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
        
        This method identifies time gaps in the data that are longer than the configured
        maximum gap duration and inserts NaN values at the start of these gaps. This
        approach ensures that gaps are explicitly marked in the data rather than being
        silently filled by interpolation methods.
        
        Args:
            data: Time series data with DatetimeIndex. Can be either a Series or DataFrame.
        
        Returns:
            Same type as input with NaN values inserted at gap locations. The returned
            data will be sorted by index.
        
        Raises:
            TypeError: If data index is not a DatetimeIndex.
        
        Example:
            
            >>> dates = pd.date_range('2024-01-01', freq='1H', periods=5)
            >>> dates = dates.delete([2, 3])  # Create 2-hour gap
            >>> series = pd.Series([1, 2, 3], index=dates[[0, 1, 4]])
            >>> handler = TimeSeriesGapHandler(max_gap_in_minutes=90)
            >>> result = handler.insert_nans_at_gaps(series)
            >>> print(result)  # Shows NaN inserted at gap start
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError(f"Data index must be DatetimeIndex, got {type(data.index)}")
            
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


if __name__ == '__main__':
    dates = pd.date_range('2024-01-01', periods=10, freq='30min')
    dates = dates.delete([3, 4, 5, 6])  # Create data gap of 4 periods
    values = np.random.rand(len(dates))
    sample_series = pd.Series(values, index=dates)

    handler = TimeSeriesGapHandler(max_gap_in_minutes=30)
    result = handler.insert_nans_at_gaps(sample_series)
    print("See how we created a gap of 4 periods, while there is only 1 NaN value inserted.")
    print("If you now create a line plot, you'll see a visual gap instead of a bridge between the surrounding values.")
    print(result)
