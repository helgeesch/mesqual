from typing import Union, List, Literal, Callable, Dict, Any, Hashable
from datetime import time
import copy
import calendar
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mesqual.utils.pandas_utils.move_level import move_top_level_last

X_AXIS_AGGS = Literal['date', 'year_month', 'year_week', 'week', 'month', 'year']
X_AXIS_TYPES = Union[X_AXIS_AGGS, List[X_AXIS_AGGS]]
GROUPBY_AGG_TYPES = Union[str, List[str]]
_DEFAULT_COLOR_SCALE = 'Turbo'


class DashboardConfig:
    """Configuration of a time series heatmap dashboard.

    Holds the parameters of `TimeSeriesDashboardGenerator`: x-axis aggregation,
    facets, KPI statistics and color settings. The generator builds the
    configuration from its own arguments.

    The statistics column shows one value per statistic. `stat_aggs` maps a
    statistic name to a function that takes a Series and returns one number.
    `DEFAULT_STATISTICS` is the default set. `STATISTICS_LIBRARY` holds more
    statistics to combine with it.

    Examples:
        >>> from mesqual.visualizations import TimeSeriesDashboardGenerator
        >>> stats = {
        ...     'Peak': lambda x: x.max(),
        ...     'Spread': lambda x: x.max() - x.min(),
        ...     'Median': DashboardConfig.STATISTICS_LIBRARY['Median'],
        ... }
        >>> generator = TimeSeriesDashboardGenerator(x_axis='date', facet_row='Country', stat_aggs=stats)
        >>> fig = generator.get_figure(dataset.fetch('Country.Results.price'))
        >>> list(fig.data[1].y)
        [' ', 'Peak', 'Spread', 'Median']
    """

    DEFAULT_STATISTICS = {
        'Datums': lambda x: len(x),
        'Abs max': lambda x: x.abs().max(),
        'Abs mean': lambda x: x.abs().mean(),
        'Max': lambda x: x.max(),
        'Mean': lambda x: x.mean(),
        'Min': lambda x: x.min(),
    }

    STATISTICS_LIBRARY = {
        '# Values': lambda x: (~x.isna()).sum(),
        '# NaNs': lambda x: x.isna().sum(),
        '% == 0': lambda x: (x.round(2) == 0).sum() / (~x.isna()).sum() * 100,
        '% != 0': lambda x: ((x.round(2) != 0) & (~x.isna())).sum() / (~x.isna()).sum() * 100,
        '% > 0': lambda x: (x.round(2) > 0).sum() / (~x.isna()).sum() * 100,
        '% < 0': lambda x: (x.round(2) < 0).sum() / (~x.isna()).sum() * 100,
        'Mean of v>0': lambda x: x.where(x > 0, np.nan).mean(),
        'Mean of v<0': lambda x: x.where(x < 0, np.nan).mean(),
        'Median': lambda x: x.median(),
        'Q0.99': lambda x: x.quantile(0.99),
        'Q0.95': lambda x: x.quantile(0.95),
        'Q0.05': lambda x: x.quantile(0.05),
        'Q0.01': lambda x: x.quantile(0.01),
        'Std': lambda x: x.std(),
    }

    def __init__(
        self,
        x_axis: X_AXIS_TYPES = 'date',
        facet_col: str = None,
        facet_row: str = None,
        facet_col_wrap: int = None,
        facet_col_order: list[str] = None,
        facet_row_order: list[str] = None,
        ratio_of_stat_col: float = 0.1,
        stat_aggs: Dict[str, Callable[[pd.Series], float | int]] = None,
        groupby_aggregation: GROUPBY_AGG_TYPES = 'mean',
        title: str = None,
        color_continuous_scale: str | list[str] | list[tuple[float, str]] = _DEFAULT_COLOR_SCALE,
        color_continuous_midpoint: int | float = None,
        range_color: list[int | float] = None,
        per_facet_col_colorscale: bool = False,
        per_facet_row_colorscale: bool = False,
        facet_row_color_settings: dict = None,
        facet_col_color_settings: dict = None,
        subplots_vertical_spacing: float = None,
        subplots_horizontal_spacing: float = None,
        time_series_figure_kwargs: dict = None,
        stat_figure_kwargs: dict = None,
        universal_figure_kwargs: dict = None,
        use_string_for_axis: bool = False,
        **figure_kwargs: Any,
    ):
        """Initialize dashboard configuration.

        Args:
            x_axis: X-axis aggregation type ('date', 'year_month', 'year_week', 'week', 'month', 'year')
                   or list of aggregation types for faceting.
            facet_col: Column name to use for column faceting, or 'x_axis'/'groupby_aggregation'
                      for parameter-based faceting. With no facet set, each column of a one-level
                      input becomes a column facet.
            facet_row: Column name to use for row faceting, or 'x_axis'/'groupby_aggregation'
                      for parameter-based faceting.
            facet_col_wrap: Maximum number of columns per row when using column faceting.
            facet_col_order: Custom ordering for column facets.
            facet_row_order: Custom ordering for row facets.
            ratio_of_stat_col: Width ratio of statistics column relative to heatmap column.
            stat_aggs: Dictionary of statistic names to aggregation functions for KPI calculation.
            groupby_aggregation: Aggregation method for grouping data ('mean', 'sum', etc.) or list
                               of methods for faceting.
            title: Dashboard title.
            color_continuous_scale: Plotly colorscale name or custom colorscale. Default 'Turbo'.
            color_continuous_midpoint: Midpoint value for diverging colorscales.
            range_color: Fixed color range [min, max] for heatmaps.
            per_facet_col_colorscale: Whether to use separate colorscales per column facet.
            per_facet_row_colorscale: Whether to use separate colorscales per row facet.
            facet_row_color_settings: Custom color settings per row facet category.
            facet_col_color_settings: Custom color settings per column facet category.
            subplots_vertical_spacing: Vertical spacing between subplots.
            subplots_horizontal_spacing: Horizontal spacing between subplots.
            time_series_figure_kwargs: Additional kwargs for heatmap traces.
            stat_figure_kwargs: Additional kwargs for statistics traces.
            universal_figure_kwargs: kwargs applied to all traces.
            use_string_for_axis: Whether to convert axis values to strings.
            **figure_kwargs: Additional figure-level kwargs.
        """
        self.x_axis = x_axis
        self.facet_col = facet_col
        self.facet_row = facet_row
        self.facet_col_wrap = facet_col_wrap
        self.facet_col_order = facet_col_order
        self.facet_row_order = facet_row_order
        self.ratio_of_stat_col = ratio_of_stat_col
        self.stat_aggs = stat_aggs or self.DEFAULT_STATISTICS
        self.groupby_aggregation = groupby_aggregation
        self.title = title

        self.per_facet_col_colorscale = per_facet_col_colorscale
        self.per_facet_row_colorscale = per_facet_row_colorscale

        if per_facet_col_colorscale and per_facet_row_colorscale:
            raise ValueError('Cannot use both per_facet_col_colorscale and per_facet_row_colorscale simultaneously')
        if facet_row_color_settings and not per_facet_row_colorscale:
            raise ValueError('Set per_facet_row_colorscale to True in order to use facet_row_color_settings.')
        if facet_col_color_settings and not per_facet_col_colorscale:
            raise ValueError('Set per_facet_col_colorscale to True in order to use facet_col_color_settings.')

        self.facet_row_color_settings = facet_row_color_settings or {}
        self.facet_col_color_settings = facet_col_color_settings or {}

        self.time_series_figure_kwargs = time_series_figure_kwargs or {}
        self.stat_figure_kwargs = stat_figure_kwargs or {}

        self.subplots_vertical_spacing = subplots_vertical_spacing
        self.subplots_horizontal_spacing = subplots_horizontal_spacing

        universal_figure_kwargs = universal_figure_kwargs or {}

        self.figure_kwargs = {
            'color_continuous_scale': color_continuous_scale,
            'color_continuous_midpoint': color_continuous_midpoint,
            'range_color': range_color,
            **universal_figure_kwargs,
            **figure_kwargs,
        }
        self.use_string_for_axis = use_string_for_axis


class DataProcessor:
    """Checks and reshapes the input data of `TimeSeriesDashboardGenerator` for its facets and aggregations."""

    @staticmethod
    def validate_input_data_and_config(data: pd.DataFrame, config: DashboardConfig) -> None:
        x_axis = config.x_axis
        groupby_aggregation = config.groupby_aggregation
        facet_col = config.facet_col
        facet_row = config.facet_row
        facet_col_wrap = config.facet_col_wrap

        if facet_col_wrap is not None and facet_row is not None:
            raise ValueError('You cannot set facet_row if you are setting a facet_col_wrap')

        if isinstance(data, pd.Series):
            if sum(facet not in [None, 'x_axis', 'groupby_aggregation'] for facet in [facet_col, facet_row]):
                raise ValueError('You can not define facet_col or facet_row if you just have a pd.Series')
        elif data.columns.nlevels > 2:
            raise ValueError('Your data must not have more than 2 column index levels.')
        elif data.columns.nlevels == 2:
            if (facet_col is None) and (facet_row is None):
                raise ValueError('If you have two column levels, you must define both, facet_col and facet_row.')
            if isinstance(x_axis, list) or isinstance(groupby_aggregation, list):
                raise ValueError(
                    'You cannot set x_axis or groupby_aggregation to a list if your data already has 2 levels.'
                )
        elif data.columns.nlevels == 1:
            if sum(facet not in [None, 'x_axis', 'groupby_aggregation'] for facet in [facet_col, facet_row]) > 1:
                raise ValueError('You only have 1 column level. You can only define facet_col or facet_row')
            if isinstance(x_axis, list) and isinstance(groupby_aggregation, list):
                raise ValueError(
                    'You cannot set x_axis and groupby_aggregation to a list if your data already has 1 level.'
                )

        if isinstance(x_axis, list):
            if not any('x_axis' == facet for facet in [facet_col, facet_row]):
                raise ValueError("x_axis must be either 'facet_col' or 'facet_row' when provided as a list.")
        else:
            if any('x_axis' == facet for facet in [facet_col, facet_row]):
                raise ValueError(
                    'You provided a str for x_axis, '
                    "but set facet_col or facet_row to 'x_axis'. This is not allowed! \n"
                    'You must provide a List[str] and in order to use facet_row / facet_col '
                    'for different x_axis.'
                )

        if isinstance(groupby_aggregation, list):
            if not any('groupby_aggregation' == facet for facet in [facet_col, facet_row]):
                raise ValueError(
                    "groupby_aggregation must be either 'facet_col' or 'facet_row' when provided as a list."
                )
        else:
            if any('groupby_aggregation' == facet for facet in [facet_col, facet_row]):
                raise ValueError(
                    'You provided a str for groupby_aggregation, '
                    "but set facet_col or facet_row to 'groupby_aggregation'. This is not allowed! \n"
                    'You must provide a List[str] and in order to use facet_row / facet_col '
                    'for different groupby_aggregation.'
                )

    @staticmethod
    def prepare_dataframe_for_facet(data: pd.DataFrame, config: DashboardConfig) -> pd.DataFrame:
        for k in ['x_axis', 'groupby_aggregation']:
            config_value = getattr(config, k)
            if isinstance(config_value, list):
                data = pd.concat(
                    {i: data.copy(deep=True) for i in config_value},
                    axis=1,
                    names=[k],
                )
        return data

    @classmethod
    def ensure_df_has_two_column_levels(cls, data: pd.DataFrame, config: DashboardConfig) -> pd.DataFrame:
        if isinstance(data, pd.Series):
            data = data.to_frame(data.name or 'Time series')

        if data.columns.nlevels == 1:
            data.columns.name = data.columns.name or 'variable'
            data = cls._insert_empty_column_index_level(data)

        if cls._resolve_facet_col(data, config) in [data.columns.names[0]]:
            data = move_top_level_last(data, axis=1)

        return data

    @staticmethod
    def _resolve_facet_col(data: pd.DataFrame, config: DashboardConfig) -> str | None:
        if config.facet_col is None and config.facet_row is None:
            named_levels = [name for name in data.columns.names if name is not None]
            return named_levels[0] if named_levels else None
        return config.facet_col

    @classmethod
    def update_facet_config(cls, data: pd.DataFrame, config: DashboardConfig) -> None:
        facet_col = cls._resolve_facet_col(data, config)
        unique_facet_col_keys = data.columns.get_level_values(facet_col).unique().to_list()
        if config.facet_col_order is None:
            config.facet_col_order = unique_facet_col_keys
        else:
            config.facet_col_order = config.facet_col_order + [
                c for c in unique_facet_col_keys if c not in config.facet_col_order
            ]

        unique_facet_row_keys = data.columns.get_level_values(config.facet_row).unique().to_list()
        if config.facet_row_order is None:
            config.facet_row_order = unique_facet_row_keys
        else:
            config.facet_row_order = config.facet_row_order + [
                c for c in unique_facet_row_keys if c not in config.facet_row_order
            ]

        if config.facet_col_wrap is None:
            config.facet_col_wrap = len(config.facet_col_order)

    @staticmethod
    def get_grouped_data(series: pd.Series, x_axis: str, groupby_aggregation: str) -> pd.DataFrame:
        """Group and aggregate time series data into heatmap format.

        Transforms timeseries data into a matrix suitable for heatmap visualization
        with hour-of-day on y-axis and specified time aggregation on x-axis.

        Args:
            series: Input timeseries data with datetime index. The index name is ignored.
            x_axis: Time aggregation method ('date', 'week', 'month', etc.).
            groupby_aggregation: Aggregation function name ('mean', 'sum', etc.).

        Returns:
            DataFrame with time categories as columns and hour-of-day as rows.
        """
        temp = series.rename_axis(None).to_frame('value')
        temp.loc[:, 'time'] = temp.index.time
        temp.loc[:, 'minute'] = temp.index.minute
        temp.loc[:, 'hour'] = temp.index.hour + 1
        temp.loc[:, 'date'] = temp.index.date
        temp.loc[:, 'month'] = temp.index.month
        temp.loc[:, 'year'] = temp.index.year
        temp.loc[:, 'week'] = temp.index.isocalendar().week
        temp.loc[:, 'year_month'] = temp.index.strftime('%Y-%m')
        temp.loc[:, 'year_week'] = temp.index.strftime('%Y-CW%U')

        y_axis = 'time'
        groupby = [y_axis, x_axis]
        temp = temp.groupby(groupby)['value'].agg(groupby_aggregation)
        temp = temp.unstack(x_axis)
        temp_data = temp.sort_index(ascending=False)
        return temp_data

    @staticmethod
    def _insert_empty_column_index_level(df: pd.DataFrame, level_name: str = None) -> pd.DataFrame:
        level_value = ''
        return pd.concat({level_value: df}, axis=1, names=[level_name])

    @staticmethod
    def _prepend_empty_row(df: pd.DataFrame) -> pd.DataFrame:
        empty_row = pd.DataFrame([[np.nan] * len(df.columns)], index=[' '], columns=df.columns)
        return pd.concat([empty_row, df])


class ColorManager:
    """Manages color settings and scale computation for dashboard visualizations.

    Handles colorscale selection, range computation, and facet-specific color
    customization for heatmap and statistics traces.
    """

    @staticmethod
    def get_color_settings_for_facet_category(
        config: DashboardConfig, facet_key: tuple[Hashable, Hashable]
    ) -> dict[str, Any]:
        """Get color settings for a specific facet category.

        Args:
            config: Dashboard configuration object.
            facet_key: Tuple of (row_key, col_key) identifying the facet.

        Returns:
            Dictionary of color settings for the specified facet category.
        """
        row_key, col_key = facet_key
        settings = {
            'color_continuous_scale': config.figure_kwargs.get('color_continuous_scale'),
            'color_continuous_midpoint': config.figure_kwargs.get('color_continuous_midpoint'),
            'range_color': config.figure_kwargs.get('range_color'),
        }

        if config.per_facet_row_colorscale and row_key in config.facet_row_color_settings:
            settings.update(config.facet_row_color_settings.get(row_key, {}))
        elif config.per_facet_col_colorscale and col_key in config.facet_col_color_settings:
            settings.update(config.facet_col_color_settings.get(col_key, {}))

        return settings

    @staticmethod
    def compute_color_params(
        data: pd.DataFrame,
        config: DashboardConfig,
        facet_key: tuple[Hashable, Hashable] | None = None,
    ) -> dict[str, Any]:
        """Compute color parameters for heatmap traces.

        Calculates colorscale, min/max values, and other color-related parameters
        based on data range and configuration settings.

        Args:
            data: Input data for color range calculation.
            config: Dashboard configuration object.
            facet_key: Optional facet identifier for per-facet colorscales.

        Returns:
            Dictionary of color parameters for plotly traces.
        """
        if facet_key is not None:
            settings = ColorManager.get_color_settings_for_facet_category(config, facet_key)
        else:
            settings = config.figure_kwargs

        if facet_key is not None:
            if config.per_facet_row_colorscale:
                row_key, _ = facet_key
                filtered_data = data.loc[:, (row_key, slice(None))]
            elif config.per_facet_col_colorscale:
                _, col_key = facet_key
                filtered_data = data.loc[:, (slice(None), col_key)]
            else:
                filtered_data = data
        else:
            filtered_data = data

        color_continuous_scale = settings.get('color_continuous_scale')
        if color_continuous_scale is None:
            color_continuous_scale = _DEFAULT_COLOR_SCALE
        color_continuous_midpoint = settings.get('color_continuous_midpoint')
        range_color = settings.get('range_color')

        result = {}
        if color_continuous_scale:
            result['colorscale'] = color_continuous_scale

        if range_color:
            result['zmin'] = range_color[0]
            result['zmax'] = range_color[1]
        elif color_continuous_midpoint == 0:
            _absmax = filtered_data.abs().max().max()
            result['zmin'] = -_absmax
            result['zmax'] = _absmax
        elif color_continuous_midpoint:
            raise NotImplementedError('color_continuous_midpoint other than 0 is not implemented')
        else:
            result['zmin'] = filtered_data.min().min()
            result['zmax'] = filtered_data.max().max()

        return result


class TraceGenerator:
    """Generates plotly trace objects for dashboard visualization.

    Creates heatmap traces for timeseries data, statistics traces for KPI display,
    and colorscale traces for custom color legends.
    """

    @classmethod
    def get_heatmap_trace(
        cls,
        data: pd.DataFrame,
        ts_kwargs: dict[str, Any],
        color_kwargs: dict[str, Any],
        use_string_for_axis: bool,
        **kwargs: Any,
    ) -> go.Heatmap:
        """Create a heatmap trace for timeseries data visualization.

        Args:
            data: DataFrame with time categories as columns and hour-of-day as rows.
            ts_kwargs: Additional kwargs for the heatmap trace.
            color_kwargs: Color-related parameters (colorscale, zmin, zmax).
            use_string_for_axis: Whether to convert axis values to strings.
            **kwargs: Additional plotly Heatmap parameters.

        Returns:
            Plotly Heatmap trace object for timeseries visualization.
        """
        if set(data.columns).issubset(list(range(1, 13))):
            x = [calendar.month_abbr[m] for m in range(1, 13)]
        else:
            if use_string_for_axis:
                x = [str(i).replace('-', '_') for i in data.columns]
            else:
                x = data.columns

        trace_kwargs = {**color_kwargs, **ts_kwargs, **kwargs}

        assert 'colorscale' in trace_kwargs and 'zmin' in trace_kwargs and 'zmax' in trace_kwargs

        trace_heatmap = go.Heatmap(x=x, z=data.values, y=data.index, **trace_kwargs)
        return trace_heatmap

    @classmethod
    def get_stats_trace(
        cls,
        series: pd.Series,
        stat_aggs: dict[str, Callable[[pd.Series], float | int]],
        stat_kwargs: dict[str, Any],
        color_kwargs: dict[str, Any],
        **kwargs: Any,
    ) -> go.Heatmap:
        """Create a heatmap trace for displaying KPI statistics.

        Args:
            series: Input timeseries data for statistics calculation.
            stat_aggs: Dictionary of statistic names to aggregation functions.
            stat_kwargs: Additional kwargs for the statistics trace.
            color_kwargs: Color-related parameters (colorscale, zmin, zmax).
            **kwargs: Additional plotly Heatmap parameters.

        Returns:
            Plotly Heatmap trace object displaying calculated statistics.
        """
        data_stats = pd.Series({agg: func(series) for agg, func in stat_aggs.items()})
        data_stats = data_stats.to_frame('stats')
        data_stats = DataProcessor._prepend_empty_row(data_stats)

        if 'ygap' not in stat_kwargs:
            stat_kwargs['ygap'] = 5

        text_data = data_stats.map(lambda x: f'{x:.0f}')
        text_data = text_data.replace('nan', '').replace('null', '')

        trace_kwargs = {**color_kwargs, **stat_kwargs, **kwargs}
        trace_kwargs['showscale'] = False  # Stats should never have a colorbar

        assert 'colorscale' in trace_kwargs and 'zmin' in trace_kwargs and 'zmax' in trace_kwargs

        trace_stats = go.Heatmap(
            z=data_stats.values,
            x=data_stats.columns,
            y=data_stats.index,
            text=text_data.values,
            texttemplate='%{text}',
            **trace_kwargs,
        )
        return trace_stats

    @staticmethod
    def create_colorscale_trace(
        z_min: float,
        z_max: float,
        colorscale: str | list[str] | list[tuple[float, str]],
        orientation: Literal['v', 'h'] = 'v',
        title: Hashable | None = None,
    ) -> go.Heatmap:
        """Create a colorscale trace for custom color legend display.

        Args:
            z_min: Minimum value for colorscale range.
            z_max: Maximum value for colorscale range.
            colorscale: Plotly colorscale specification.
            orientation: Colorscale orientation ('v' for vertical, 'h' for horizontal).
            title: Optional title for the colorscale.

        Returns:
            Plotly Heatmap trace object representing the colorscale legend.
        """
        if orientation == 'v':
            z_vals = np.linspace(z_min, z_max, 100).reshape(-1, 1)
        else:
            z_vals = np.linspace(z_min, z_max, 100).reshape(1, -1)

        axis_vals = np.linspace(z_min, z_max, 100)

        colorbar_settings = {'thickness': 15, 'title': title or ''}

        if orientation == 'h':
            colorbar_settings.update({'orientation': 'h', 'y': -0.15, 'xanchor': 'center', 'x': 0.5})
            x = axis_vals
            y = None
        else:
            x = None
            y = axis_vals

        return go.Heatmap(
            x=x,
            y=y,
            z=z_vals,
            colorscale=colorscale,
            showscale=False,
            zmin=z_min,
            zmax=z_max,
            colorbar=colorbar_settings,
        )


class TimeSeriesDashboardGenerator:
    """Generates time series heatmap dashboards.

    Each subplot is a heatmap with the hour of the day on the y-axis and a time
    aggregation (date, week, month, ...) on the x-axis, next to a column of KPI
    statistics. Facets come from the named column levels of the data, or from the
    parameters `x_axis` and `groupby_aggregation` when `facet_col` or `facet_row`
    is `'x_axis'` or `'groupby_aggregation'`. Colorscales can be shared or set per
    facet.

    The input data is a pandas Series or DataFrame with a DatetimeIndex. To facet
    by data, the columns have named levels; the facet arguments take these level
    names.

    Examples:
        One dashboard with a heatmap and a statistics column per scenario and
        country:

        >>> prices = study.scen.fetch('Country.Results.price')
        >>> prices.columns.names
        FrozenList(['dataset', 'Country'])
        >>> generator = TimeSeriesDashboardGenerator(x_axis='date', facet_col='dataset', facet_row='Country')
        >>> fig = generator.get_figure(prices)
        >>> len(fig.data)
        16
        >>> [annotation.text for annotation in fig.layout.annotations][:3]
        ['C1 - base', 'C1 - high_demand', 'C1 - without_N4']

        Facet by aggregation method:

        >>> generator = TimeSeriesDashboardGenerator(
        ...     x_axis='date',
        ...     groupby_aggregation=['min', 'max'],
        ...     facet_col='groupby_aggregation',
        ...     facet_row='Country',
        ... )
        >>> fig = generator.get_figure(dataset.fetch('Country.Results.price'))
        >>> [annotation.text for annotation in fig.layout.annotations]
        ['C1 - min', 'C1 - max', 'C2 - min', 'C2 - max']

        Facet by x-axis aggregation:

        >>> generator = TimeSeriesDashboardGenerator(x_axis=['date', 'week'], facet_col='x_axis')
        >>> fig = generator.get_figure(dataset.fetch('Country.Results.price')['C1'])
        >>> [annotation.text for annotation in fig.layout.annotations]
        ['date', 'week']

        One colorscale per row facet, with a fixed color range for `C1`:

        >>> generator = TimeSeriesDashboardGenerator(
        ...     x_axis='date',
        ...     facet_col='dataset',
        ...     facet_row='Country',
        ...     per_facet_row_colorscale=True,
        ...     facet_row_color_settings={'C1': {'range_color': [0, 100]}},
        ... )
        >>> fig = generator.get_figure(prices)
        >>> sorted({(round(trace.zmin), round(trace.zmax)) for trace in fig.data})
        [(0, 100), (27, 87)]
    """

    def __init__(
        self,
        x_axis: X_AXIS_TYPES = 'date',
        facet_col: str = None,
        facet_row: str = None,
        facet_col_wrap: int = None,
        facet_col_order: list[str] = None,
        facet_row_order: list[str] = None,
        ratio_of_stat_col: float = 0.1,
        stat_aggs: Dict[str, Callable[[pd.Series], float | int]] = None,
        groupby_aggregation: GROUPBY_AGG_TYPES = 'mean',
        title: str = None,
        color_continuous_scale: str | list[str] | list[tuple[float, str]] = _DEFAULT_COLOR_SCALE,
        color_continuous_midpoint: int | float = None,
        range_color: tuple[float, float] | list[int | float] = None,
        per_facet_col_colorscale: bool = False,
        per_facet_row_colorscale: bool = False,
        facet_row_color_settings: dict = None,
        facet_col_color_settings: dict = None,
        subplots_vertical_spacing: float = None,
        subplots_horizontal_spacing: float = None,
        time_series_figure_kwargs: dict = None,
        stat_figure_kwargs: dict = None,
        universal_figure_kwargs: dict = None,
        use_string_for_axis: bool = False,
        config_cls: type[DashboardConfig] = DashboardConfig,
        data_processor_cls: type[DataProcessor] = DataProcessor,
        color_manager_cls: type[ColorManager] = ColorManager,
        trace_generator_cls: type[TraceGenerator] = TraceGenerator,
        **figure_kwargs: Any,
    ):
        """Initialize the timeseries dashboard generator.

        Args:
            x_axis: Time aggregation for x-axis or list for faceting.
            facet_col: Column faceting specification. With no facet set, each column of a one-level
                input becomes a column facet.
            facet_row: Row faceting specification.
            facet_col_wrap: Maximum columns per row in faceted layout.
            facet_col_order: Custom ordering for column facets.
            facet_row_order: Custom ordering for row facets.
            ratio_of_stat_col: Width ratio of statistics column to heatmap.
            stat_aggs: Custom KPI aggregation functions.
            groupby_aggregation: Data aggregation method or list for faceting.
            title: Dashboard title.
            color_continuous_scale: Plotly colorscale specification. Default 'Turbo'.
            color_continuous_midpoint: Midpoint for diverging colorscales.
            range_color: Fixed color range for heatmaps.
            per_facet_col_colorscale: Enable separate colorscales per column facet.
            per_facet_row_colorscale: Enable separate colorscales per row facet.
            facet_row_color_settings: Custom color settings per row facet.
            facet_col_color_settings: Custom color settings per column facet.
            subplots_vertical_spacing: Vertical spacing between subplots.
            subplots_horizontal_spacing: Horizontal spacing between subplots.
            time_series_figure_kwargs: Additional heatmap trace parameters.
            stat_figure_kwargs: Additional statistics trace parameters.
            universal_figure_kwargs: Parameters applied to all traces.
            use_string_for_axis: Convert axis values to strings.
            config_cls: Configuration class for dependency injection.
            data_processor_cls: Data processor class for dependency injection.
            color_manager_cls: Color manager class for dependency injection.
            trace_generator_cls: Trace generator class for dependency injection.
            **figure_kwargs: Additional figure-level parameters.
        """
        self.config = config_cls(
            x_axis=x_axis,
            facet_col=facet_col,
            facet_row=facet_row,
            facet_col_wrap=facet_col_wrap,
            facet_col_order=facet_col_order,
            facet_row_order=facet_row_order,
            ratio_of_stat_col=ratio_of_stat_col,
            stat_aggs=stat_aggs,
            groupby_aggregation=groupby_aggregation,
            title=title,
            color_continuous_scale=color_continuous_scale,
            color_continuous_midpoint=color_continuous_midpoint,
            range_color=range_color,
            per_facet_col_colorscale=per_facet_col_colorscale,
            per_facet_row_colorscale=per_facet_row_colorscale,
            facet_row_color_settings=facet_row_color_settings,
            facet_col_color_settings=facet_col_color_settings,
            subplots_vertical_spacing=subplots_vertical_spacing,
            subplots_horizontal_spacing=subplots_horizontal_spacing,
            time_series_figure_kwargs=time_series_figure_kwargs,
            stat_figure_kwargs=stat_figure_kwargs,
            universal_figure_kwargs=universal_figure_kwargs,
            use_string_for_axis=use_string_for_axis,
            **figure_kwargs,
        )
        self.data_processor_cls = data_processor_cls
        self.color_manager_cls = color_manager_cls
        self.trace_generator_cls = trace_generator_cls

    def get_figure(self, data: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Generate a complete dashboard figure from timeseries data.

        Creates a plotly figure containing heatmaps with associated statistics,
        properly formatted axes, and optional faceting. Applies all configured
        styling, color schemes, and layout settings.

        Args:
            data: Input timeseries DataFrame or Series with datetime index.
            **kwargs: Runtime configuration overrides.

        Returns:
            Plotly Figure object containing the complete dashboard visualization.
        """
        original_config = copy.deepcopy(self.config)

        try:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

            if not kwargs.get('_skip_validation', False):
                self.data_processor_cls.validate_input_data_and_config(data, self.config)
            data = self.data_processor_cls.prepare_dataframe_for_facet(data.copy(), self.config)
            data = self.data_processor_cls.ensure_df_has_two_column_levels(data, self.config)
            self.data_processor_cls.update_facet_config(data, self.config)

            fig = self._create_figure_layout_with_subplots(data)

            self._add_heatmap_and_stat_traces_to_figure(data, fig)

            if self.config.per_facet_col_colorscale:
                self._add_column_colorscales(data, fig)
                fig.update_traces(showlegend=False)
            elif self.config.per_facet_row_colorscale:
                self._add_row_colorscales(data, fig)
                fig.update_traces(showlegend=False)

            if self.config.title:
                fig.update_layout(
                    title=f'<b>{self.config.title}</b>',
                    title_x=0.5,
                )
            return fig
        finally:
            self.config = original_config

    def get_figures_chunked(
        self,
        data: pd.DataFrame,
        max_n_rows_per_figure: int = None,
        n_figures: int = None,
        chunk_title_suffix: bool = True,
        **kwargs: Any,
    ) -> list[go.Figure]:
        """Generate multiple figures by splitting facet rows into chunks.

        Useful for handling large datasets with many row facets by creating
        multiple smaller figures instead of one large figure.

        Args:
            data: Input timeseries DataFrame with datetime index.
            max_n_rows_per_figure: Maximum number of row facets per figure.
            n_figures: Total number of figures to create (alternative to max_n_rows_per_figure).
            chunk_title_suffix: Whether to add "(Part X/Y)" suffix to titles.
            **kwargs: Runtime configuration overrides.

        Returns:
            List of plotly Figure objects, each containing a subset of row facets.

        Raises:
            ValueError: If both or neither of max_n_rows_per_figure and n_figures are provided.
        """
        original_config = copy.deepcopy(self.config)

        try:
            if sum(x is not None for x in [max_n_rows_per_figure, n_figures]) != 1:
                raise ValueError('Provide exactly one of: max_n_rows_per_figure or n_figures')

            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

            data = self.data_processor_cls.prepare_dataframe_for_facet(data, self.config)
            data = self.data_processor_cls.ensure_df_has_two_column_levels(data, self.config)
            self.data_processor_cls.update_facet_config(data, self.config)

            if self.config.facet_row is None:
                return [self.get_figure(data, **kwargs)]

            total_rows = len(self.config.facet_row_order)

            if max_n_rows_per_figure:
                n_chunks = math.ceil(total_rows / max_n_rows_per_figure)
                chunk_size = max_n_rows_per_figure
            else:
                n_chunks = n_figures
                chunk_size = math.ceil(total_rows / n_figures)

            figures = []
            original_title = self.config.title

            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, total_rows)
                chunk_rows = self.config.facet_row_order[start_idx:end_idx]

                if not chunk_rows:
                    continue

                chunk_kwargs = kwargs.copy()
                chunk_kwargs['facet_row_order'] = chunk_rows

                if chunk_title_suffix and original_title:
                    chunk_kwargs['title'] = f'{original_title} (Part {i + 1}/{n_chunks})'
                elif chunk_title_suffix:
                    chunk_kwargs['title'] = f'Part {i + 1}/{n_chunks}'

                cols_in_chunk = [
                    c
                    for c, facet_row_category in zip(data.columns, data.columns.get_level_values(self.config.facet_row))
                    if facet_row_category in chunk_rows
                ]
                data_chunk = data[cols_in_chunk]

                fig = self.get_figure(data_chunk, **chunk_kwargs, _skip_validation=True)
                figures.append(fig)
            return figures
        finally:
            self.config = original_config

    def _create_figure_layout_with_subplots(self, data: pd.DataFrame) -> go.Figure:
        facet_col_wrap = max([1, self.config.facet_col_wrap])
        ratio_of_stat_col = self.config.ratio_of_stat_col

        has_colorscale_col = self.config.per_facet_row_colorscale
        has_colorscale_row = self.config.per_facet_col_colorscale

        num_facet_rows = max([1, len(self.config.facet_row_order)])
        num_facet_cols = max([1, len(self.config.facet_col_order)])

        num_rows = math.ceil(num_facet_cols / facet_col_wrap) * num_facet_rows
        num_cols = facet_col_wrap * 2  # Each facet gets a heatmap + stats column

        if has_colorscale_col:
            num_cols += 1
        if has_colorscale_row:
            num_rows += 1

        subplot_titles = self._generate_subplot_titles(has_colorscale_col, has_colorscale_row)
        column_widths = self._get_column_widths(facet_col_wrap, has_colorscale_col, ratio_of_stat_col)
        row_heights = self._get_row_heights(has_colorscale_row, num_rows)
        specs = [[{} for _ in range(num_cols)] for _ in range(num_rows)]

        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=subplot_titles,
            column_widths=column_widths,
            row_heights=row_heights,
            specs=specs,
            vertical_spacing=self.config.subplots_vertical_spacing,
            horizontal_spacing=self.config.subplots_horizontal_spacing,
        )
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', margin=dict(t=50, b=50))

        return fig

    def _get_row_heights(self, has_colorscale_row: bool, num_rows: int) -> list[float]:
        row_heights = None
        if has_colorscale_row:
            regular_height = 1.0
            colorscale_height = 0.15

            total_regular_rows = num_rows - 1
            total_height = total_regular_rows * regular_height + colorscale_height
            norm_regular = regular_height / total_height
            norm_colorscale = colorscale_height / total_height

            row_heights = [norm_regular] * total_regular_rows + [norm_colorscale]
        return row_heights

    def _get_column_widths(self, facet_col_wrap, has_colorscale_col, ratio_of_stat_col) -> list[float]:
        if has_colorscale_col:
            colorscale_width = 0.03
            adjusted_width = 1 - colorscale_width
            column_widths = []

            for _ in range(facet_col_wrap):
                heatmap_width = (adjusted_width - ratio_of_stat_col) / facet_col_wrap
                stats_width = ratio_of_stat_col / facet_col_wrap
                column_widths.extend([heatmap_width, stats_width])

            column_widths.append(colorscale_width)
        else:
            column_widths = [
                (1 - ratio_of_stat_col) / facet_col_wrap,
                ratio_of_stat_col / facet_col_wrap,
            ] * facet_col_wrap
        return column_widths

    def _generate_subplot_titles(self, has_colorscale_col, has_colorscale_row):
        subplot_titles = []
        for row_name in self.config.facet_row_order:
            for col_name in self.config.facet_col_order:
                if row_name and col_name:
                    title = f'{row_name} - {col_name}'
                else:
                    title = row_name or col_name
                subplot_titles.append(title)  # Title for heatmap
                subplot_titles.append(None)  # Title for stats

            if has_colorscale_col:
                subplot_titles.append(row_name)

        if has_colorscale_row:
            for col_name in self.config.facet_col_order:
                subplot_titles.append(col_name)
                subplot_titles.append(None)
        return subplot_titles

    def _add_heatmap_and_stat_traces_to_figure(self, data, fig):
        facet_col_wrap = self.config.facet_col_wrap

        disable_main_colorbars = self.config.per_facet_col_colorscale or self.config.per_facet_row_colorscale
        if disable_main_colorbars:
            self.config.time_series_figure_kwargs['showscale'] = False

        global_color_params = {}
        if not (self.config.per_facet_col_colorscale or self.config.per_facet_row_colorscale):
            global_color_params = self.color_manager_cls.compute_color_params(data, self.config)

        current_row = 1
        row_offset = 0

        for row_idx, row_key in enumerate(self.config.facet_row_order):
            col_offset = 0
            for col_idx, col_key in enumerate(self.config.facet_col_order):
                facet_pos = col_idx % facet_col_wrap
                if facet_pos == 0 and col_idx > 0:
                    row_offset += 1

                fig_row = current_row + row_offset
                fig_col = col_offset + facet_pos * 2 + 1  # +1 because plotly indexing starts at 1

                data_col = facet_key = (row_key, col_key)
                if data_col not in data.columns:
                    continue
                series = data[data_col]

                x_axis = self._get_effective_param_for_data_col('x_axis', data_col)
                groupby_aggregation = self._get_effective_param_for_data_col('groupby_aggregation', data_col)

                self._set_hovertemplates(x_axis)

                grouped_data = self.data_processor_cls.get_grouped_data(series, x_axis, groupby_aggregation)

                color_params = self._get_color_params_for_facet(data, facet_key, global_color_params)

                show_colorbar = False
                if not disable_main_colorbars:
                    show_colorbar = row_idx == 0 and col_idx == 0

                heatmap_trace = self.trace_generator_cls.get_heatmap_trace(
                    grouped_data,
                    self.config.time_series_figure_kwargs,
                    color_params,
                    showscale=show_colorbar,
                    use_string_for_axis=self.config.use_string_for_axis,
                )

                fig.add_trace(heatmap_trace, row=fig_row, col=fig_col)

                fig.update_yaxes(
                    tickvals=[time(hour=h, minute=0) for h in [0, 6, 12, 18]] + [max(grouped_data.index)],
                    ticktext=['0', '6', '12', '18', '24'],
                    row=fig_row,
                    col=fig_col,
                    autorange='reversed',
                )

                if x_axis == 'year_week':
                    fig.update_xaxes(dtick=8, row=fig_row, col=fig_col)

                stats_trace = self.trace_generator_cls.get_stats_trace(
                    series, self.config.stat_aggs, self.config.stat_figure_kwargs, color_params
                )

                fig.add_trace(stats_trace, row=fig_row, col=fig_col + 1)

                fig.update_xaxes(showgrid=False, row=fig_row, col=fig_col + 1)
                fig.update_yaxes(showgrid=False, autorange='reversed', row=fig_row, col=fig_col + 1)

            if col_offset == 0:
                current_row += math.ceil(len(self.config.facet_col_order) / facet_col_wrap)

    def _get_color_params_for_facet(
        self, data: pd.DataFrame, facet_key: tuple[Hashable, Hashable], global_color_params: dict
    ) -> dict:
        if self.config.per_facet_col_colorscale or self.config.per_facet_row_colorscale:
            color_params = self.color_manager_cls.compute_color_params(data, self.config, facet_key)
        else:
            color_params = global_color_params
        return color_params

    def _add_row_colorscales(self, data, fig):
        colorscale_col = self.config.facet_col_wrap * 2 + 1  # Column after all heatmaps and stats

        for row_idx, row_key in enumerate(self.config.facet_row_order):
            row_pos = row_idx * math.ceil(len(self.config.facet_col_order) / self.config.facet_col_wrap) + 1

            facet_key = (row_key, self.config.facet_col_order[0])
            colorscale, z_max, z_min = self._get_color_settings_for_category(data, facet_key)
            colorscale_trace = self.trace_generator_cls.create_colorscale_trace(z_min, z_max, colorscale, 'v', row_key)

            fig.add_trace(colorscale_trace, row=row_pos, col=colorscale_col)
            fig.update_xaxes(showticklabels=False, showgrid=False, row=row_pos, col=colorscale_col)
            fig.update_yaxes(showticklabels=True, showgrid=False, row=row_pos, col=colorscale_col, side='right')

    def _get_color_settings_for_category(self, data, facet_key):
        color_params = self.color_manager_cls.compute_color_params(data, self.config, facet_key)
        colorscale = color_params.get('colorscale', 'viridis')
        z_min = color_params.get('zmin', 0)
        z_max = color_params.get('zmax', 1)
        return colorscale, z_max, z_min

    def _add_column_colorscales(self, data, fig):
        colorscale_row = (
            math.ceil(len(self.config.facet_col_order) / self.config.facet_col_wrap) * len(self.config.facet_row_order)
            + 1
        )

        for col_idx, col_key in enumerate(self.config.facet_col_order):
            col_pos = (col_idx % self.config.facet_col_wrap) * 2 + 1

            facet_key = (self.config.facet_row_order[0], col_key)

            colorscale, z_max, z_min = self._get_color_settings_for_category(data, facet_key)

            colorscale_trace = self.trace_generator_cls.create_colorscale_trace(z_min, z_max, colorscale, 'h', col_key)

            fig.add_trace(colorscale_trace, row=colorscale_row, col=col_pos)
            fig.update_xaxes(showticklabels=True, showgrid=False, row=colorscale_row, col=col_pos)
            fig.update_yaxes(showticklabels=False, showgrid=False, row=colorscale_row, col=col_pos)

    def _get_effective_param_for_data_col(self, param_name, data_col):
        param_value = getattr(self.config, param_name)
        if not isinstance(param_value, list):
            return param_value
        else:
            return list(set(param_value).intersection(list(data_col)))[0]

    def _set_hovertemplates(self, x_axis):
        ts_kwargs = self.config.time_series_figure_kwargs
        stat_kwargs = self.config.stat_figure_kwargs

        ts_kwargs['hovertemplate'] = f'{x_axis}: %{{x}}<br>Hour of day: %{{y}}<br>Value: %{{z}}<extra></extra>'
        stat_kwargs['hovertemplate'] = f'aggregation: %{{y}}<br>Value: %{{z}}<extra></extra>'
