from types import SimpleNamespace

import folium
from shapely import Point

from mesqual.kpis.attributes import KPIAttributes
from mesqual.kpis.collection import KPICollection
from mesqual.kpis.kpi import KPI
from mesqual.units import QuantityToTextConverter, Units
from mesqual.visualizations.folium_viz_system.kpi_collection_map_visualizer import KPICollectionMapVisualizer
from mesqual.visualizations.folium_viz_system.viz_text_overlay import (
    TextOverlayFeatureResolver,
    TextOverlayGenerator,
)
from tests.helpers import FrameDataset


class _Converter:
    def convert(self, quantity) -> str:
        return quantity


def _kpi(name: str, value_text: str) -> SimpleNamespace:
    return SimpleNamespace(quantity=value_text, get_kpi_name_with_dataset_name=lambda: name)


class _Group(list):
    @property
    def empty(self) -> bool:
        return not self


class _GroupingManager:
    def get_related_kpi_groups(self, kpi, study_manager) -> dict:
        return {'Group <g>': _Group([_kpi('related <r>', '2 < 3'), _kpi('js', 'a`b ${x}')])}


def test_related_kpi_tooltip_escapes_names_and_values():
    visualizer = KPICollectionMapVisualizer(
        generators=[],
        study_manager=object(),
        include_related_kpis_in_tooltip=True,
        kpi_grouping_manager=_GroupingManager(),
    )
    mapper = visualizer._create_enhanced_tooltip_generator(_Converter())
    html = mapper.map_data_item(SimpleNamespace(kpi=_kpi('price <p>', '5 & 6')))
    for escaped in [
        'price &lt;p&gt;',
        '5 &amp; 6',
        'Group &lt;g&gt;',
        'related &lt;r&gt;',
        '2 &lt; 3',
        'a&#96;b &#36;{x}',
    ]:
        assert escaped in html
    assert '`' not in html
    assert '${' not in html
    assert '<p>' not in html.replace('<p>&nbsp;</p>', '')


def _price_kpi() -> KPI:
    dataset = FrameDataset(name='ds')
    attributes = KPIAttributes(
        flag='items.price',
        unit=Units.EUR_per_MWh,
        model_flag=FrameDataset.FLAG,
        object_name='a',
        dataset_name=dataset.name,
        dataset_attributes=dataset.attributes,
    )
    return KPI(value=12.3456789, attributes=attributes, dataset=dataset)


def _tooltip_and_popup(value_formatting, popup=True) -> tuple[str, str]:
    resolver = TextOverlayFeatureResolver(location=Point(8.0, 50.0), tooltip=True, popup=popup)
    visualizer = KPICollectionMapVisualizer(
        generators=TextOverlayGenerator(resolver), value_formatting=value_formatting
    )
    (feature_group,) = visualizer.get_feature_groups(KPICollection([_price_kpi()]))
    (marker,) = feature_group._children.values()
    children = list(marker._children.values())
    (tooltip,) = [child.text for child in children if isinstance(child, folium.Tooltip)]
    (popup,) = [child.html.render() for child in children if isinstance(child, folium.Popup)]
    return tooltip, popup


def test_default_kpi_tooltip_and_popup_format_values_with_value_formatting():
    for html in _tooltip_and_popup(QuantityToTextConverter(decimals=4)):
        assert '>12.3457 €/MWh<' in html


def test_default_kpi_tooltip_and_popup_without_value_formatting_show_the_default_text():
    for html in _tooltip_and_popup(None):
        assert '>12.3 €/MWh<' in html


def test_kpi_popup_of_the_user_stays_as_given():
    _, popup = _tooltip_and_popup(QuantityToTextConverter(decimals=4), popup='<b>mine</b>')
    assert '<b>mine</b>' in popup
