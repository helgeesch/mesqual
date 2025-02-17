import numpy as np
import pandas as pd


class TimeSeriesGapHandler:
    def __init__(self, max_gap_in_hours: float = 1.0):
        self._max_gap_in_hours = max_gap_in_hours

    def insert_nans_at_gaps(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        diffs = data.index.to_series().diff()
        mask = diffs > pd.Timedelta(hours=self._max_gap_in_hours)
        gap_indices = np.where(mask)[0]
        new_timestamps = data.index[gap_indices - 1] + pd.Timedelta(hours=self._max_gap_in_hours)

        if isinstance(data, pd.Series):
            new_values = pd.Series(np.nan, index=new_timestamps)
        else:
            new_values = pd.DataFrame(np.nan, index=new_timestamps, columns=data.columns)

        return pd.concat([data, new_values]).sort_index()


if __name__ == '__main__':
    dates = pd.date_range('2024-01-01', periods=5, freq='30min')
    dates = dates.delete(2)
    values = np.random.rand(len(dates))
    sample_series = pd.Series(values, index=dates)

    handler = TimeSeriesGapHandler(max_gap_in_hours=0.75)
    result = handler.insert_nans_at_gaps(sample_series)
    print(result)
