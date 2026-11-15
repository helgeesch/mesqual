import copy

import pandas as pd
import plotly.graph_objects as go
import pytest

from mesqual.testing import make_dummy_study
from mesqual.visualizations.plotly_figures.timeseries_dashboard import (
    DataProcessor,
    TimeSeriesDashboardGenerator,
)

X_AXES = ['date', 'week', 'month', 'year_month', 'year_week', 'year']


def _data(index_name: str | None = None) -> pd.DataFrame:
    index = pd.date_range('2025-01-01', periods=96 * 3, freq='15min', tz='Europe/Berlin', name=index_name)
    return pd.DataFrame({'a': range(len(index))}, index=index)


class TestDashboardIndexName:
    @pytest.mark.parametrize('x_axis', X_AXES)
    @pytest.mark.parametrize('name_kind', ['time', 'x_axis', 'other'])
    def test_figure_equals_figure_for_unnamed_index(self, x_axis, name_kind):
        index_name = {'time': 'time', 'x_axis': x_axis, 'other': 'other'}[name_kind]
        generator = TimeSeriesDashboardGenerator(x_axis=x_axis, facet_col='variable')
        expected = generator.get_figure(_data(None))
        actual = generator.get_figure(_data(index_name))
        assert len(expected.data) > 0
        assert actual.to_json() == expected.to_json()


class TestDashboardDefaults:
    def test_x_axis_year(self):
        index = pd.date_range('2024-12-30', periods=96 * 4, freq='15min')
        series = pd.Series(range(len(index)), index=index)
        grouped = DataProcessor.get_grouped_data(series, 'year', 'mean')
        assert list(grouped.columns) == [2024, 2025]

    @pytest.mark.parametrize('as_series', [False, True])
    def test_default_config_draws_one_facet(self, as_series):
        data = _data()['a'] if as_series else _data()
        generator = TimeSeriesDashboardGenerator()
        config_before = copy.deepcopy(vars(generator.config))
        figure = generator.get_figure(data)
        heatmaps = [trace for trace in figure.data if isinstance(trace, go.Heatmap)]
        assert len(heatmaps) == 2  # one heatmap and one statistics column
        assert vars(generator.config) == config_before

    def test_default_config_draws_one_facet_per_column(self):
        data = _data().assign(b=1.0)
        figure = TimeSeriesDashboardGenerator().get_figure(data)
        assert len([trace for trace in figure.data if isinstance(trace, go.Heatmap)]) == 4

    def test_two_level_frame_with_facet_col_and_facet_row(self):
        data = pd.concat({'s1': _data(), 's2': _data()}, axis=1, names=['scenario', 'variable'])
        figure = TimeSeriesDashboardGenerator(facet_col='scenario', facet_row='variable').get_figure(data)
        assert len([trace for trace in figure.data if isinstance(trace, go.Heatmap)]) == 4

    def test_color_scale_none_gives_default_scale(self):
        none_figure = TimeSeriesDashboardGenerator(facet_col='variable', color_continuous_scale=None).get_figure(
            _data()
        )
        default_figure = TimeSeriesDashboardGenerator(facet_col='variable').get_figure(_data())
        assert none_figure.to_json() == default_figure.to_json()


class TestDashboardConfigRestore:
    def test_next_call_works_after_get_figure_raised(self):
        generator = TimeSeriesDashboardGenerator(facet_col='variable')
        config_before = copy.deepcopy(vars(generator.config))
        with pytest.raises(ValueError):
            generator.get_figure(_data(), facet_row='variable')
        assert vars(generator.config) == config_before
        other = _data().rename(columns={'a': 'b'})
        assert len(generator.get_figure(other).data) > 0

    def test_get_figures_chunked_with_one_figure_keeps_config(self):
        generator = TimeSeriesDashboardGenerator(facet_col='variable')
        config_before = copy.deepcopy(vars(generator.config))
        figures = generator.get_figures_chunked(_data(), n_figures=1)
        assert len(figures) == 1
        assert vars(generator.config) == config_before


class TestDashboardFacetLayout:
    @staticmethod
    def _two_columns() -> pd.DataFrame:
        data = _data()
        return data.assign(b=data['a'] + 1000)

    def test_per_facet_col_colorscale_gives_each_column_its_range(self):
        generator = TimeSeriesDashboardGenerator(
            facet_col='variable',
            per_facet_col_colorscale=True,
            facet_col_color_settings={'a': {'range_color': [0, 10]}},
        )
        data = self._two_columns()
        figure = generator.get_figure(data)
        heatmaps = [trace for trace in figure.data if isinstance(trace, go.Heatmap)]
        ranges = {(trace.zmin, trace.zmax) for trace in heatmaps}
        assert ranges == {(0, 10), (data['b'].min(), data['b'].max())}

    def test_facet_col_wrap_puts_facets_below_each_other(self):
        figure = TimeSeriesDashboardGenerator(facet_col='variable', facet_col_wrap=1).get_figure(self._two_columns())
        annotations = {annotation.text: annotation for annotation in figure.layout.annotations}
        assert set(annotations) == {'a', 'b'}
        assert annotations['a'].x == annotations['b'].x
        assert annotations['a'].y > annotations['b'].y

    @staticmethod
    def _study_prices() -> pd.DataFrame:
        return make_dummy_study().scen.fetch('Country.Results.price')

    def test_partial_facet_orders_come_first_and_the_rest_follows(self):
        generator = TimeSeriesDashboardGenerator(
            facet_col='dataset', facet_row='Country', facet_row_order=['C2'], facet_col_order=['short']
        )
        titles = [annotation.text for annotation in generator.get_figure(self._study_prices()).layout.annotations]
        assert titles == [
            'C2 - short',
            'C2 - base',
            'C2 - high_demand',
            'C2 - without_N4',
            'C1 - short',
            'C1 - base',
            'C1 - high_demand',
            'C1 - without_N4',
        ]

    def test_missing_facet_combination_draws_no_traces_for_it(self):
        prices = self._study_prices()
        generator = TimeSeriesDashboardGenerator(facet_col='dataset', facet_row='Country')
        full = generator.get_figure(prices)
        partial = generator.get_figure(prices.drop(columns=[('short', 'C2')]))
        assert len(partial.data) == len(full.data) - 2

    def test_midpoint_zero_in_facet_settings_gives_a_symmetric_range(self):
        prices = self._study_prices()
        generator = TimeSeriesDashboardGenerator(
            facet_col='dataset',
            facet_row='Country',
            per_facet_row_colorscale=True,
            facet_row_color_settings={'C1': {'color_continuous_midpoint': 0}},
        )
        figure = generator.get_figure(prices)
        absmax = prices.xs('C1', axis=1, level='Country').abs().max().max()
        c2 = prices.xs('C2', axis=1, level='Country')
        ranges = {(trace.zmin, trace.zmax) for trace in figure.data}
        assert ranges == {(-absmax, absmax), (c2.min().min(), c2.max().max())}
