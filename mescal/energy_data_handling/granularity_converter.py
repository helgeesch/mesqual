from enum import Enum

import pandas as pd

from mescal.enums import QuantityTypeEnum
from mescal.energy_data_handling.granularity_analyzer import TimeSeriesGranularityAnalyzer


class GranularityConversionError(Exception):
    """Exception raised when granularity conversion operations fail.
    
    This exception is raised when conversion between different time granularities
    cannot be performed due to incompatible data formats, unsupported granularities,
    or other conversion-specific errors.
    """
    pass


class SamplingMethodEnum(Enum):
    """Enumeration of sampling methods for granularity conversion.
    
    Attributes:
        UPSAMPLING: Converting from coarser to finer granularity (e.g., hourly to 15-min)
        DOWNSAMPLING: Converting from finer to coarser granularity (e.g., 15-min to hourly)
        KEEP: No conversion needed - source and target granularities are the same
    """
    UPSAMPLING = 'upsampling'
    DOWNSAMPLING = 'downsampling'
    KEEP = 'keep'


class TimeSeriesGranularityConverter:
    """Converts time series between different granularities while respecting the nature of the quantity.

    This class handles the conversion of time series data between different granularities
    (e.g., hourly to 15-min or vice versa) while properly accounting for the physical
    nature of the quantity being converted:

    - Intensive quantities (e.g., prices, power levels) are replicated when increasing
      granularity and averaged when decreasing granularity.

    - Extensive quantities (e.g., volumes, welfare) are split when increasing
      granularity and summed when decreasing granularity.

    Features:
        - Automatic granularity detection using TimeGranularityAnalyzer
        - Per-day processing to handle missing periods properly and prevent incorrect autofilling of missing days
        - Support for both intensive and extensive quantities
        - Timezone-aware processing including daylight saving transitions
    """
    def __init__(self):
        """Initialize the granularity converter with analyzer instances.
        
        Creates both strict and non-strict granularity analyzers for different
        validation requirements during conversion operations.
        """
        self._strict_gran_analyzer = TimeSeriesGranularityAnalyzer(strict_mode=True)
        self._non_strict_gran_analyzer = TimeSeriesGranularityAnalyzer(strict_mode=False)

    def _validate_series_format(self, series: pd.Series) -> None:
        """Validate that the input series has the required DatetimeIndex format.
        
        Args:
            series: Time series to validate
            
        Raises:
            TypeError: If series index is not a DatetimeIndex
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError(f"Series index must be DatetimeIndex, got {type(series.index)}")

    def upsample_through_fillna(
            self,
            data: pd.DataFrame | pd.Series,
            quantity_type: QuantityTypeEnum
    ) -> pd.DataFrame | pd.Series:
        """Upsample data using forward-fill strategy with quantity-type-aware scaling.
        
        This method handles upsampling of sparse data where some values are missing.
        It uses forward-fill to propagate values and applies appropriate scaling
        based on the quantity type:
        
        - For INTENSIVE quantities: Values are replicated without scaling
        - For EXTENSIVE quantities: Values are divided by the number of periods
          they are spread across within each hour-segment group
        
        The method processes data per day and hour to handle missing periods properly
        and prevent incorrect auto-filling across day boundaries.
        
        Args:
            data: Time series data to upsample (Series or DataFrame)
            quantity_type: Type of quantity being converted (INTENSIVE or EXTENSIVE)
            
        Returns:
            Upsampled data with same type as input
            
        Example:
            
            >>> # For extensive quantities (energy), values are divided
            >>> series = pd.Series([100, np.nan, np.nan, np.nan, 200, np.nan, np.nan, np.nan],
            ...                   index=pd.date_range('2024-01-01', freq='15min', periods=5))
            >>> converter.upsample_through_fillna(series, QuantityTypeEnum.EXTENSIVE)
            # Results in [25, 25, 25, 25, 50, 50, 50, 50]
        """
        if isinstance(data, pd.Series):
            return self._upsample_series(data, quantity_type)

        tmp = data.copy().sort_index()
        idx = tmp.index.tz_convert('UTC') if tmp.index.tz is not None else tmp.index

        if quantity_type == QuantityTypeEnum.EXTENSIVE:
            segment_patterns = tmp.notna().cumsum()

            # Group columns by their segment pattern
            pattern_to_cols = {}
            for col in tmp.columns:
                pattern = tuple(segment_patterns[col].values)  # Convert to tuple to make it hashable
                pattern_to_cols.setdefault(pattern, []).append(col)

            # Process each group of columns with same pattern
            result_pieces = []
            for pattern, cols in pattern_to_cols.items():
                segments = segment_patterns[cols[0]]  # Take segments from first column (all are same)
                piece = (
                    tmp[cols]
                    .groupby([idx.date, idx.hour, segments])
                    .transform(lambda s: s.ffill() / len(s))
                )
                result_pieces.append(piece)

            return pd.concat(result_pieces, axis=1).loc[data.index].rename_axis(data.columns.names, axis=1)
        else:
            return tmp.groupby([idx.date, idx.hour]).ffill().loc[data.index]

    def _upsample_series(self, series: pd.Series, quantity_type: QuantityTypeEnum) -> pd.Series:
        """Helper method to upsample a Series using DataFrame-based upsampling.
        
        Args:
            series: Time series to upsample
            quantity_type: Type of quantity being converted
            
        Returns:
            Upsampled series
        """
        return self.upsample_through_fillna(
            series.to_frame(),
            quantity_type
        ).iloc[:, 0]

    def convert_to_target_index(
            self,
            series: pd.Series,
            target_index: pd.DatetimeIndex,
            quantity_type: QuantityTypeEnum
    ) -> pd.Series:
        """Convert a time series to match a specific target DatetimeIndex.
        
        This method converts the granularity of a time series to match the granularity
        of a target index. The target index must have consistent granularity within
        each day and consistent granularity across all days.
        
        Args:
            series: Source time series to convert
            target_index: DatetimeIndex defining the target granularity and timestamps
            quantity_type: Type of quantity (INTENSIVE or EXTENSIVE) for proper scaling
            
        Returns:
            Series converted to match target index granularity and timestamps
            
        Raises:
            ValueError: If target index has multiple granularities within days
                       or inconsistent granularity across days
            
        Example:
            
            >>> # Convert hourly to 15-min data
            >>> hourly_series = pd.Series([100, 150, 200], 
            ...                          index=pd.date_range('2024-01-01', freq='1H', periods=3))
            >>> target_idx = pd.date_range('2024-01-01', freq='15min', periods=12)
            >>> result = converter.convert_to_target_index(hourly_series, target_idx,
            ...                                          QuantityTypeEnum.INTENSIVE)
        """
        self._validate_series_format(series)
        target_gran_series = self._strict_gran_analyzer.get_granularity_as_series_of_timedeltas(target_index)
        _grouped = target_gran_series.groupby(target_gran_series.index.date)
        if _grouped.nunique().max() > 1:
            raise ValueError(f"Found some dates with multiple granularities within same day. Can't handle that!")
        if _grouped.first().nunique() > 1:
            raise ValueError(f"Found multiple granularities. Can't handle that!")
        target_granularity = pd.Timedelta(target_gran_series.values[0])
        return series.groupby(series.index.date).apply(
            lambda x: self._convert_date_to_target_granularity(x, target_granularity, quantity_type)
        ).droplevel(0).rename_axis(series.index.name).rename(series.name)

    def convert_to_target_granularity(
            self,
            series: pd.Series,
            target_granularity: pd.Timedelta,
            quantity_type: QuantityTypeEnum
    ) -> pd.Series:
        """Convert a time series to a specific target granularity.
        
        This method converts the temporal granularity of a time series while properly
        handling the physical nature of the quantity. The conversion is performed
        day-by-day to prevent incorrect handling of missing days or daylight saving
        time transitions.
        
        Args:
            series: Source time series to convert
            target_granularity: Target granularity as a pandas Timedelta
                               (e.g., pd.Timedelta(minutes=15) for 15-minute data)
            quantity_type: Type of quantity for proper scaling:
                          - INTENSIVE: Values are averaged/replicated (prices, power)
                          - EXTENSIVE: Values are summed/split (volumes, energy)
                          
        Returns:
            Series with converted granularity, maintaining original naming and metadata
            
        Raises:
            GranularityConversionError: If conversion cannot be performed due to
                                       unsupported granularities or data issues
                                       
        Example:
            
            >>> # Convert 15-minute to hourly data (downsampling)
            >>> quarter_hourly = pd.Series([25, 30, 35, 40], 
            ...                           index=pd.date_range('2024-01-01', freq='15min', periods=4))
            >>> hourly = converter.convert_to_target_granularity(
            ...     quarter_hourly, pd.Timedelta(hours=1), QuantityTypeEnum.EXTENSIVE)
            >>> print(hourly)  # Result: [130] (25+30+35+40)
        """
        self._validate_series_format(series)
        return series.groupby(series.index.date).apply(
            lambda x: self._convert_date_to_target_granularity(x, target_granularity, quantity_type)
        ).droplevel(0).rename_axis(series.index.name).rename(series.name)

    def _convert_date_to_target_granularity(
        self,
        series: pd.Series,
        target_granularity: pd.Timedelta,
        quantity_type: QuantityTypeEnum
    ) -> pd.Series:
        """Convert granularity for a single date's worth of data.
        
        This internal method handles the actual conversion logic for data within
        a single date range. It determines whether upsampling, downsampling, or
        no conversion is needed, then applies the appropriate method.
        
        Args:
            series: Time series data for a single date (may span into next date)
            target_granularity: Target granularity as Timedelta
            quantity_type: Type of quantity for scaling decisions
            
        Returns:
            Converted series for the date period
            
        Raises:
            GranularityConversionError: If conversion parameters are invalid or
                                       unsupported granularities are encountered
        """
        if len(set(series.index.date)) > 2:
            raise GranularityConversionError('This method is intended for single-date conversion only.')
        source_gran = self._non_strict_gran_analyzer.get_granularity_as_series_of_minutes(series.index)
        if len(source_gran.unique()) > 1:
            raise GranularityConversionError('Cannot convert data with changing granularity within a single day.')
        source_gran_minutes = source_gran.values[0]
        target_gran_minutes = target_granularity.total_seconds() / 60

        _allowed_granularities = [1, 5, 15, 30, 60, 24*60]
        if target_gran_minutes not in _allowed_granularities:
            raise GranularityConversionError(
                f'Target granularity {target_gran_minutes} minutes not supported. '
                f'Allowed granularities: {_allowed_granularities} minutes'
            )

        if target_gran_minutes > source_gran_minutes:
            sampling = SamplingMethodEnum.DOWNSAMPLING
        elif target_gran_minutes < source_gran_minutes:
            sampling = SamplingMethodEnum.UPSAMPLING
        else:
            sampling = SamplingMethodEnum.KEEP

        if sampling == SamplingMethodEnum.UPSAMPLING:
            scaling_factor = source_gran_minutes / target_gran_minutes
            if (scaling_factor % 1) != 0:
                raise GranularityConversionError(
                    f'Source granularity ({source_gran_minutes} min) is not evenly divisible '
                    f'by target granularity ({target_gran_minutes} min)'
                )
            else:
                scaling_factor = int(scaling_factor)

            new_index = pd.date_range(
                start=series.index[0],
                end=series.index[-1],
                freq=f"{target_gran_minutes}min",
                tz=series.index.tz
            )

            # For intensive quantities, replicate values; for extensive, divide by scaling factor
            if quantity_type == QuantityTypeEnum.INTENSIVE:
                return series.reindex(new_index, method='ffill')
            else:  # EXTENSIVE
                return series.reindex(new_index, method='ffill') / scaling_factor

        elif sampling == SamplingMethodEnum.DOWNSAMPLING:
            groups = series.groupby(pd.Grouper(freq=f"{target_gran_minutes}min"))
            # For extensive quantities, sum the values; for intensive, take the mean
            func = 'sum' if quantity_type == QuantityTypeEnum.EXTENSIVE else 'mean'
            return groups.agg(func)
        
        else:  # SamplingMethodEnum.KEEP
            return series

        return series


if __name__ == '__main__':
    import time
    import numpy as np

    converter = TimeSeriesGranularityConverter()

    tz = 'Europe/Berlin'
    hourly_index = pd.date_range('2024-01-01', '2024-12-31 23:45', freq='h', tz=tz)
    quarter_hourly_index = pd.date_range('2024-01-01', '2024-12-31 23:45', freq='15min', tz=tz)

    for qt in [QuantityTypeEnum.INTENSIVE, QuantityTypeEnum.EXTENSIVE]:
        for idx in [hourly_index, quarter_hourly_index]:
            series = pd.Series(100, index=idx)
            series = series.loc[series.index.difference(series.loc['2024-02'].index)]
            print(f'{qt} series: \n{series}')
            for target in [15, 60]:
                start = time.time()
                ts = converter.convert_to_target_granularity(series, pd.Timedelta(minutes=target), qt)
                print(f'To granularity {target}min took {time.time()-start:.2f} seconds:\n{ts}')
            for target_idx in [hourly_index, quarter_hourly_index]:
                start = time.time()
                ts = converter.convert_to_target_index(series, target_idx, qt)
                print(f'To target idx {target_idx[:2]} min took {time.time()-start:.2f} seconds:\n{ts}')

    # Test fillna upsampling
    _values = [100, np.nan, np.nan, np.nan, 200, np.nan, 300, np.nan, 50, np.nan, np.nan, np.nan, np.nan, np.nan]
    mixed_series = pd.Series(_values, index=quarter_hourly_index[:len(_values)])

    for qt in [QuantityTypeEnum.INTENSIVE, QuantityTypeEnum.EXTENSIVE]:
        ts = converter.upsample_through_fillna(mixed_series, qt)
        print(f'Upsampled as {qt}:\n{ts}')