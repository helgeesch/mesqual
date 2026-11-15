"""Smoke tests: the map and dashboard generators run on `mesqual.testing` data and write HTML files."""

import folium
import plotly.graph_objects as go
import pytest

from mesqual.kpis.attributes import KPIAttributes
from mesqual.kpis.collection import KPICollection
from mesqual.kpis.kpi import KPI
from mesqual.testing import DummyDataset, DummyScenario, make_dummy_study
from mesqual.units import Units
from mesqual.visualizations import HTMLDashboard
from mesqual.visualizations.folium_viz_system import (
    AreaFeatureResolver,
    AreaGenerator,
    KPICollectionMapVisualizer,
    LineFeatureResolver,
    LineGenerator,
    PropertyMapper,
    TextOverlayFeatureResolver,
    TextOverlayGenerator,
)
from mesqual.visualizations.plotly_figures.timeseries_dashboard import TimeSeriesDashboardGenerator
from mesqual.visualizations.value_mapping_system import DiscreteColorMapping, SegmentedContinuousColorscale

PALETTE = ['#123456', '#abcdef']


@pytest.fixture(scope='module')
def dataset() -> DummyDataset:
    return DummyDataset(DummyScenario(tz='Europe/Berlin'))


def model_map(dataset: DummyDataset) -> tuple[folium.Map, dict[str, folium.FeatureGroup]]:
    """Countries colored by name, node labels and border lines, one feature group each."""
    generators = {
        'Country.Model': AreaGenerator(
            AreaFeatureResolver(
                fill_color=PropertyMapper.from_item_attr('name', DiscreteColorMapping(palette=PALETTE)),
                tooltip=True,
            )
        ),
        'Node.Model': TextOverlayGenerator(
            TextOverlayFeatureResolver(
                text_print_content=PropertyMapper.from_item_attr('country'),
            )
        ),
        'CountryBorder.Model': LineGenerator(
            LineFeatureResolver(
                geometry=PropertyMapper.from_item_attr('geo_line_string'),
                line_color='#000000',
                tooltip=True,
            )
        ),
    }
    folium_map = folium.Map()
    groups = {}
    for flag, generator in generators.items():
        groups[flag] = folium.FeatureGroup(name=flag)
        generator.generate_objects_for_model_df(dataset.fetch(flag), groups[flag])
        groups[flag].add_to(folium_map)
    return folium_map, groups


def price_kpis(dataset: DummyDataset, sign: float) -> KPICollection:
    """One KPI per country: `sign` times the mean country price."""
    means = dataset.fetch('Country.Results.price').mean()
    return KPICollection(
        [
            KPI(
                value=sign * float(value),
                attributes=KPIAttributes(
                    flag='Country.Results.price',
                    unit=Units.EUR_per_MWh,
                    model_flag='Country.Model',
                    object_name=country,
                    dataset_name=dataset.name,
                    dataset_attributes=dataset.attributes,
                ),
                dataset=dataset,
            )
            for country, value in means.items()
        ]
    )


def test_model_maps_draw_one_feature_per_object_and_save_html(dataset, tmp_path):
    folium_map, groups = model_map(dataset)
    for flag, group in groups.items():
        assert len(group._children) == len(dataset.fetch(flag)), flag
    path = tmp_path / 'model_map.html'
    folium_map.save(str(path))
    html = path.read_text(encoding='utf-8')
    assert all(color in html for color in PALETTE)


@pytest.mark.parametrize('use_abs_kpi_value, sign', [(False, 1.0), (True, -1.0)])
def test_kpi_map_with_area_and_text_overlay_saves_html(dataset, tmp_path, use_abs_kpi_value, sign):
    kpis = price_kpis(dataset, sign)
    values = [abs(kpi.value) for kpi in kpis]
    scale = SegmentedContinuousColorscale.single_segment_autoscale_factory_from_array(values=values)
    visualizer = KPICollectionMapVisualizer(
        generators=[
            AreaGenerator(
                AreaFeatureResolver(
                    fill_color=PropertyMapper.from_kpi_value(scale, use_abs_kpi_value=use_abs_kpi_value),
                    tooltip=True,
                )
            ),
            TextOverlayGenerator(TextOverlayFeatureResolver()),
        ]
    )
    folium_map = folium.Map()
    (group,) = visualizer.generate_and_add_feature_groups_to_map(kpis, folium_map)
    assert len(group._children) == 2 * kpis.size
    path = tmp_path / 'kpi_map.html'
    folium_map.save(str(path))
    html = path.read_text(encoding='utf-8')
    colors = {scale(value) for value in values}
    assert len(colors) == 2
    assert all(color in html for color in colors)


def test_time_series_dashboard_and_html_dashboard_save_html(dataset, tmp_path):
    """Prices of a study with a DST day and a change from hourly to 15-minute steps."""
    prices = make_dummy_study().scen.fetch('Node.Results.price')
    figure = TimeSeriesDashboardGenerator(facet_col='dataset', facet_row='Node').get_figure(prices)
    n_facets = prices.columns.nunique()
    assert len([trace for trace in figure.data if isinstance(trace, go.Heatmap)]) == 2 * n_facets
    dashboard = HTMLDashboard(name='Testing data')
    dashboard.add_plotly_figure(figure, name='prices', tab='Prices')
    dashboard.add_folium_map(model_map(dataset)[0], name='model', tab='Model')
    path = tmp_path / 'dashboard.html'
    dashboard.save(str(path))
    html = path.read_text(encoding='utf-8')
    assert html.count('class="plotly-graph-div"') == 1
    assert 'L.map(' in html
