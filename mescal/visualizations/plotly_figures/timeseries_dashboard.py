from typing import Union, List, Literal, Callable, Dict, Any
from datetime import time
import calendar
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from eda.utils.pandas_utils.sort_multiindex import sort_multiindex


X_AXIS_AGGS = Literal['date', 'year_month', 'year_week', 'month', 'year']
X_AXIS_TYPES = Union[X_AXIS_AGGS, List[X_AXIS_AGGS]]
GROUPBY_AGG_TYPES = Union[str, List[str]]


def _prepend_empty_row_in_df(df):
    _empty_row = pd.DataFrame([[np.nan]*len(df.columns)], index=[' '], columns=df.columns)
    _data = pd.concat([_empty_row, df])
    return _data


def _insert_empty_column_index_level(df: pd.DataFrame, level_name: str = None) -> pd.DataFrame:
    level_value = ''
    return pd.concat({level_value: df}, axis=1, names=[level_name])


class TimeSeriesDashboard:
    DEFAULT_STATISTICS: Dict[str, Callable[[pd.Series], Union[float, int]]] = {
        'Datums': lambda x: len(x),
        'Abs max': lambda x: x.abs().max(),
        'Abs mean': lambda x: x.abs().mean(),
        'Max': lambda x: x.max(),
        'Mean': lambda x: x.mean(),
        'Min': lambda x: x.min(),
    }

    STATISTICS_LIBRARY: Dict[str, Callable[[pd.Series], Union[float, int]]] = {
        '# Values': lambda x: (~x.isna()).sum(),
        '# NaNs': lambda x: x.isna().sum(),
        '% == 0': lambda x: (x.round(2) == 0).sum() / (~x.isna()).sum() * 100,
        '% != 0': lambda x: ((x.round(2) != 0) & (~x.isna())).sum() / (~x.isna()).sum() * 100,
        '% > 0': lambda x: (x.round(2) > 0).sum() / (~x.isna()).sum() * 100,
        '% < 0': lambda x: (x.round(2) < 0).sum() / (~x.isna()).sum() * 100,
        'mean of v>0': lambda x: x.where(x > 0, np.nan).mean(),
        'mean of v<0': lambda x: x.where(x < 0, np.nan).mean(),
        'Q0.99': lambda x: x.quantile(0.99),
        'Q0.95': lambda x: x.quantile(0.95),
        'Q0.05': lambda x: x.quantile(0.05),
        'Q0.01': lambda x: x.quantile(0.01),
        'Std': lambda x: x.std(),
    }

    def __init__(
            self,
            x_axis: X_AXIS_TYPES = 'date',
            facet_col: Union[str, int] = None,
            facet_row: Union[str, int] = None,
            facet_col_wrap: int = None,
            facet_col_order: List[Union[str, int]] = None,
            facet_row_order: List[Union[str, int]] = None,
            ratio_of_stat_col: float = 0.1,
            stat_aggs: Dict[str, Callable[[pd.Series], Union[float, int]]] = None,
            groupby_aggregation: GROUPBY_AGG_TYPES = 'mean',
            title: str = None,
            color_continuous_scale: Union[str, list[str], list[tuple[float, str]]] = None,
            color_continuous_midpoint: Union[int, float] = None,
            range_color: List[Union[int, float]] = None,
            time_series_figure_kwargs: Dict[str, Any] = None,
            stat_figure_kwargs: Dict[str, Any] = None,
            universal_figure_kwargs: Dict[str, Any] = None,
            **figure_kwargs
    ):
        super().__init__()

        self.processing_kwargs = {
            'x_axis': x_axis,
            'facet_col': facet_col,
            'facet_row': facet_row,
            'facet_col_wrap': facet_col_wrap,
            'facet_col_order': facet_col_order,
            'facet_row_order': facet_row_order,
            'ratio_of_stat_col': ratio_of_stat_col,
            'stat_aggs': self.DEFAULT_STATISTICS if stat_aggs is None else stat_aggs,
            'groupby_aggregation': groupby_aggregation,
            'title': title,
        }

        self.time_series_figure_kwargs = time_series_figure_kwargs if time_series_figure_kwargs is not None else dict()
        self.stat_figure_kwargs = stat_figure_kwargs if stat_figure_kwargs is not None else dict()

        if universal_figure_kwargs is None:
            universal_figure_kwargs = dict()
        if figure_kwargs is None:
            figure_kwargs = dict()

        self.figure_kwargs = {
            'color_continuous_scale': color_continuous_scale,
            'color_continuous_midpoint': color_continuous_midpoint,
            'range_color': range_color,
            **universal_figure_kwargs,
            **figure_kwargs,
        }

    def get_figure(
            self,
            data: Union[pd.Series, pd.DataFrame],
            **kwargs
    ) -> go.Figure:

        processing_kwargs = self.processing_kwargs.copy()
        time_series_figure_kwargs = self.time_series_figure_kwargs.copy()
        stat_figure_kwargs = self.stat_figure_kwargs.copy()
        figure_kwargs = self.figure_kwargs.copy()

        for key, value in kwargs.copy().items():
            for local_kwargs in [processing_kwargs, time_series_figure_kwargs, stat_figure_kwargs, figure_kwargs]:
                if key in local_kwargs:
                    local_kwargs[key] = value
                    kwargs.pop(key)
        figure_kwargs = {**figure_kwargs, **kwargs}

        self._check_data_input(data, processing_kwargs)
        data = self._prepare_dataframe_in_case_of_method_call_facet(data, processing_kwargs)
        data = self._ensure_dataframe_format_with_two_column_levels(data)

        self._update_facet_kwargs(data, processing_kwargs)

        data = sort_multiindex(data, processing_kwargs['facet_col_order'], level=processing_kwargs['facet_col'], axis=1)
        data = sort_multiindex(data, processing_kwargs['facet_row_order'], level=processing_kwargs['facet_row'], axis=1)

        x_axis = processing_kwargs['x_axis']
        facet_col_wrap = processing_kwargs['facet_col_wrap']
        stat_aggs = processing_kwargs['stat_aggs']
        groupby_aggregation = processing_kwargs['groupby_aggregation']

        self._update_color_kwargs(data, figure_kwargs)

        fig = self._get_figure_layout(data, processing_kwargs)

        fig_row, fig_col = 0, 0
        for i, data_col in enumerate(data.columns):
            if (i % facet_col_wrap) == 0:
                fig_row += 1
                fig_col = 1
            else:
                fig_col += 2

            temp_ts = data[data_col]

            if not isinstance(x_axis, list):
                _x_axis = x_axis
            else:
                _x_axis = list(set(x_axis).intersection(list(data_col)))[0]
            if not isinstance(groupby_aggregation, list):
                _groupby_aggregation = groupby_aggregation
            else:
                _groupby_aggregation = list(set(groupby_aggregation).intersection(list(data_col)))[0]

            self._set_hovertemplates(_x_axis, time_series_figure_kwargs, stat_figure_kwargs)
            temp_data = self._get_grouped_and_aggregated_data_df(temp_ts, _x_axis, _groupby_aggregation)

            trace_heatmap = self._get_heatmap_trace(
                temp_data,
                time_series_figure_kwargs,
                figure_kwargs
            )
            trace_heatmap.update(showscale=i == 0)
            fig.add_trace(trace_heatmap, row=fig_row, col=fig_col)
            fig.update_yaxes(
                tickvals=[time(hour=h, minute=0) for h in [0, 6, 12, 18]] + [max(temp_data.index)],
                ticktext=['0', '6', '12', '18', '24'],
                row=fig_row,
                col=fig_col,
                autorange='reversed',
            )

            if _x_axis == 'year_week':
                fig.update_xaxes(dtick=8, row=fig_row, col=fig_col)

            trace_stats = self._get_stats_trace(temp_ts, stat_aggs, stat_figure_kwargs, figure_kwargs)
            fig.add_trace(trace_stats, row=fig_row, col=fig_col + 1)
            fig.update_xaxes(showgrid=False, row=fig_row, col=fig_col + 1)
            fig.update_yaxes(showgrid=False, autorange='reversed', row=fig_row, col=fig_col + 1)

        if ('title' in processing_kwargs) and (processing_kwargs['title'] is not None):
            fig.update_layout(
                title=f'<b>{processing_kwargs["title"]}</b>',
                title_x=0.5,
            )

        return fig

    @staticmethod
    def _prepare_dataframe_in_case_of_method_call_facet(data, processing_kwargs):
        for k in ['x_axis', 'groupby_aggregation']:
            if isinstance(processing_kwargs[k], list):
                data = pd.concat(
                    {
                        i: data.copy(deep=True)
                        for i in processing_kwargs[k]
                    },
                    axis=1,
                    names=[k],
                )
        return data

    @staticmethod
    def _get_grouped_and_aggregated_data_df(temp_ts, x_axis, groupby_aggregation):
        temp = temp_ts.to_frame('value')
        temp.loc[:, 'time'] = temp.index.time
        temp.loc[:, 'minute'] = temp.index.minute
        temp.loc[:, 'hour'] = temp.index.hour + 1
        temp.loc[:, 'date'] = temp.index.date
        temp.loc[:, 'month'] = temp.index.month
        temp.loc[:, 'year_month'] = temp.index.strftime('%Y-%m')
        temp.loc[:, 'year_week'] = temp.index.strftime('%Y-CW%U')
        y_axis = 'time'
        groupby = [y_axis, x_axis]
        temp = temp.groupby(groupby)['value'].agg(groupby_aggregation)
        temp = temp.unstack(x_axis)
        temp_data = temp.sort_index(ascending=False)
        return temp_data

    @staticmethod
    def _ensure_dataframe_format_with_two_column_levels(data: pd.DataFrame) -> pd.DataFrame:
        if isinstance(data, pd.Series):
            _name = data.name
            if _name is None:
                _name = 'Time series'
            data = data.to_frame(_name)
        if data.columns.nlevels == 1:
            if data.columns.name is None:
                data.columns.name = 'variable'
            data = _insert_empty_column_index_level(data)
        # TODO: ensure each combination of the columns exist so that facet col doesn't get out of order in case of missing data
        return data

    @staticmethod
    def _update_facet_kwargs(data: pd.DataFrame, processing_kwargs: Dict):
        level_0, level_1 = data.columns.names

        if level_0 is None:
            level_0 = 0
        if level_1 is None:
            level_1 = 1
        if (processing_kwargs['facet_row'] is None) and (processing_kwargs['facet_col'] is None):
            if len(set(data.columns.get_level_values(level_0))) > len(set(data.columns.get_level_values(level_1))):
                processing_kwargs['facet_row'], processing_kwargs['facet_col'] = level_0, level_1
            else:
                processing_kwargs['facet_row'], processing_kwargs['facet_col'] = level_1, level_0
        elif processing_kwargs['facet_row'] is None:
            if processing_kwargs['facet_col'] == level_0:
                processing_kwargs['facet_row'] = level_1
            else:
                processing_kwargs['facet_row'] = level_0
        elif processing_kwargs['facet_col'] is None:
            if processing_kwargs['facet_row'] == level_0:
                processing_kwargs['facet_col'] = level_1
            else:
                processing_kwargs['facet_col'] = level_0
        if processing_kwargs['facet_col_order'] is None:
            processing_kwargs['facet_col_order'] = data.columns.get_level_values(
                processing_kwargs['facet_col']).unique().to_list()
        if processing_kwargs['facet_row_order'] is None:
            processing_kwargs['facet_row_order'] = data.columns.get_level_values(
                processing_kwargs['facet_row']).unique().to_list()
        if processing_kwargs['facet_col_wrap'] is None:
            processing_kwargs['facet_col_wrap'] = len(processing_kwargs['facet_col_order'])

    @staticmethod
    def _check_data_input(data: pd.DataFrame, processing_kwargs: Dict):
        x_axis = processing_kwargs['x_axis']
        groupby_aggregation = processing_kwargs['groupby_aggregation']
        facet_col = processing_kwargs['facet_col']
        facet_row = processing_kwargs['facet_row']
        facet_col_wrap = processing_kwargs['facet_col_wrap']

        if facet_col_wrap is not None:
            if facet_row is not None:
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
                raise ValueError(
                    "x_axis must be either 'facet_col' or 'facet_row' when provided as a list."
                )
        else:
            if any('x_axis' == facet for facet in [facet_col, facet_row]):
                raise ValueError(
                    "You provided a str for x_axis, "
                    "but set facet_col or facet_row to 'x_axis'. This is not allowed! \n"
                    "You must provide a List[str] and in order to use facet_row / facet_col "
                    "for different x_axis."
                )

        if isinstance(groupby_aggregation, list):
            if not any('groupby_aggregation' == facet for facet in [facet_col, facet_row]):
                raise ValueError(
                    "groupby_aggregation must be either 'facet_col' or 'facet_row' when provided as a list."
                )
        else:
            if any('groupby_aggregation' == facet for facet in [facet_col, facet_row]):
                raise ValueError(
                    "You provided a str for groupby_aggregation, "
                    "but set facet_col or facet_row to 'groupby_aggregation'. This is not allowed! \n"
                    "You must provide a List[str] and in order to use facet_row / facet_col "
                    "for different groupby_aggregation."
                )

    @staticmethod
    def _set_hovertemplates(x_axis, time_series_figure_kwargs, stat_figure_kwargs):
        time_series_figure_kwargs['hovertemplate'] = f"{x_axis}: %{{x}}<br>Hour of day: %{{y}}<br>Value: %{{z}}<extra></extra>"
        stat_figure_kwargs['hovertemplate'] = f"aggregation: %{{y}}<br>Value: %{{z}}<extra></extra>"

    @staticmethod
    def _get_figure_layout(data: pd.DataFrame, processing_kwargs: Dict) -> go.Figure:
        facet_col = processing_kwargs['facet_col']
        facet_col_wrap = processing_kwargs['facet_col_wrap']
        ratio_of_stat_col = processing_kwargs['ratio_of_stat_col']

        data_columns = data.columns.to_list()

        def _get_subplot_title(column):
            if set(data.columns.get_level_values(0)) == set(data.columns.get_level_values(facet_col)):
                col_name, row_name = 0, 1
            else:
                col_name, row_name = 1, 0

            if column[0] and column[1]:
                return f'{column[col_name]} - {column[row_name]}'
            elif column[0]:
                return f'{column[0]}'
            else:
                return f'{column[1]}'

        subplot_titles = [x for c in data_columns for x in [_get_subplot_title(c), None]]
        num_variables = len(data.columns)
        num_rows = math.ceil(num_variables / facet_col_wrap)
        column_widths = [(1 - ratio_of_stat_col) / facet_col_wrap, ratio_of_stat_col / facet_col_wrap] * facet_col_wrap
        fig = make_subplots(num_rows, facet_col_wrap * 2, subplot_titles=subplot_titles, column_widths=column_widths)
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)')

        return fig

    @staticmethod
    def _update_color_kwargs(data: pd.DataFrame, kwargs: Dict):
        color_continuous_scale = kwargs['color_continuous_scale']
        color_continuous_midpoint = kwargs['color_continuous_midpoint']
        range_color = kwargs['range_color']

        if color_continuous_scale:
            kwargs['colorscale'] = color_continuous_scale
        if range_color:
            kwargs['zmin'] = range_color[0]
            kwargs['zmax'] = range_color[1]
        elif color_continuous_midpoint == 0:
            _absmax = data.abs().max().max()
            kwargs['zmin'] = -_absmax
            kwargs['zmax'] = _absmax
        elif color_continuous_midpoint:
            raise NotImplemented
        else:
            kwargs['zmin'] = data.min().min()
            kwargs['zmax'] = data.max().max()

        for k in ['color_continuous_scale', 'color_continuous_midpoint', 'range_color']:
            kwargs.pop(k)

    @staticmethod
    def _get_heatmap_trace(grouped_and_aggregated_df: pd.DataFrame, ts_figure_kwargs, universal_figure_kwargs):
        data = grouped_and_aggregated_df
        if set(data.columns).issubset(list(range(1, 13))):
            x = [calendar.month_abbr[m] for m in range(1, 13)]
        else:
            x = data.columns
        trace_heatmap = go.Heatmap(
            x=x,
            z=data.values,
            y=data.index,
            **universal_figure_kwargs,
            **ts_figure_kwargs,
        )
        return trace_heatmap

    @staticmethod
    def _get_stats_trace(temp_ts, stat_aggs, stat_figure_kwargs, universal_figure_kwargs):
        data_stats = pd.Series({agg: func(temp_ts) for agg, func in stat_aggs.items()})
        data_stats = data_stats.to_frame('stats')
        data_stats = _prepend_empty_row_in_df(data_stats)
        if 'ygap' not in stat_figure_kwargs:
            stat_figure_kwargs['ygap'] = 5

        text_data = data_stats.map(lambda x: f'{x:.0f}')
        text_data = text_data.replace('nan', '').replace('null', '')

        trace_stats = go.Heatmap(
            z=data_stats.values,
            x=data_stats.columns,
            y=data_stats.index,
            text=text_data.values,
            texttemplate="%{text}",
            showscale=False,
            **universal_figure_kwargs,
            **stat_figure_kwargs,
        )

        return trace_stats


if __name__ == '__main__':
    # TODO: provide example
    pass
