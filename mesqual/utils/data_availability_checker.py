import pandas as pd


class DataAvailabilityChecker:
    """Analyzes temporal data availability in DataFrame columns across time intervals.

    Takes a DataFrame with DatetimeIndex and detects patterns of data availability,
    compressing consecutive periods with the same availability status into intervals.
    Handles irregular time series by inferring the dominant frequency and supports
    custom aggregation frequencies for availability checking.

    Example:
        df = pd.DataFrame({'col1': [1, np.nan, 2],
                          'col2': [np.nan, 1, 2]},
                          index=pd.date_range('2023-01-01', periods=3))
        checker = DataAvailabilityChecker(df)
        availability = checker.check_availability(freq='D')
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()
        self._validate_input()
        self._ensure_complete_index()

    def check_availability(self, freq: str = None) -> pd.DataFrame:
        availability_df = self._df.resample(freq).apply(lambda x: x.notna().any()) if freq else self._df.notna()
        return self._compress_intervals(availability_df)

    def _validate_input(self) -> None:
        if not isinstance(self._df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")

    def _get_dominant_frequency(self) -> str:
        diff_seconds = self._df.index.to_series().diff().dt.total_seconds()
        most_common_diff = diff_seconds.value_counts().index[0]

        freq_mapping = {
            60: 'T',
            3600: 'h',
            86400: 'D',
            604800: 'W',
            2592000: 'M',
            31536000: 'Y'
        }

        return freq_mapping[min(freq_mapping.keys(), key=lambda x: abs(x - most_common_diff))]

    def _ensure_complete_index(self) -> None:
        freq = self._get_dominant_frequency()
        self._df = self._df.reindex(
            pd.date_range(
                start=self._df.index.min(),
                end=self._df.index.max(),
                freq=freq
            )
        )

    def _compress_intervals(self, availability_df: pd.DataFrame) -> pd.DataFrame:
        freq = pd.infer_freq(availability_df.index)

        status_changes = (availability_df != availability_df.shift()).any(axis=1)
        change_points = availability_df.index[status_changes | (availability_df.index == availability_df.index[-1])]

        intervals: list[dict] = []
        for i in range(len(change_points) - 1):
            start_date = change_points[i]
            if i < len(change_points) - 2:
                end_date = (change_points[i + 1] - pd.Timedelta(availability_df.index.freq))
            else:
                end_date = change_points[i + 1]

            if freq in ['D', 'B', 'W', 'M', 'Y']:
                _start_date = start_date.date()
                _end_date = end_date.date()
            else:
                _start_date = start_date
                _end_date = end_date

            intervals.append({
                "start": _start_date,
                "end": _end_date,
                **availability_df.loc[start_date].to_dict()
            })

        result_df = pd.DataFrame(intervals)
        return result_df.set_index(["start", "end"])


if __name__ == "__main__":
    # Create sample DataFrame with hourly data
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="h")[:-1]
    data = {
        "col_1": [1.0] * 240 + [float("nan")] * 7800 + [1.0] * 700 + [float("nan")] * 20,
        "col_2": [1.0] * 240 + [float("nan")] * 8520,
        "col_3": [1.0] * 240 + [float("nan")] * 8280 + [1.0] * 240
    }
    df = pd.DataFrame(data, index=dates)
    df = pd.concat([df.iloc[:240], df.iloc[360:]], axis=0)

    # Initialize checker and get availability
    checker = DataAvailabilityChecker(df)

    # Check availability at raw (hourly) frequency
    hourly_availability = checker.check_availability()
    print("Hourly Availability:")
    print(hourly_availability)
    print("\n")

    # Check availability at daily frequency
    daily_availability = checker.check_availability(freq="D")
    print("Daily Availability:")
    print(daily_availability)