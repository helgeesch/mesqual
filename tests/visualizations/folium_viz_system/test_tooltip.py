import folium
import pandas as pd
from shapely import LineString, Point, Polygon

from mesqual.visualizations.folium_viz_system.base_viz_system import FeatureResolver, ResolvedFeature
from mesqual.visualizations.folium_viz_system.visualizable_data_item import ModelDataItem, VisualizableDataItem
from mesqual.visualizations.folium_viz_system.viz_areas import AreaFeatureResolver, AreaGenerator
from mesqual.visualizations.folium_viz_system.viz_lines import LineFeatureResolver, LineGenerator


def _item(**columns) -> ModelDataItem:
    return ModelDataItem(pd.Series(columns, name='node_1'))


def test_tooltip_data_leaves_out_geometries():
    item = _item(
        geometry=Polygon([(0, 0), (1, 0), (1, 1)]),
        location=Point(5, 5),
        capacity=100,
    )
    assert item.get_tooltip_data() == {'ID': 'node_1', 'capacity': '100'}


def test_default_tooltip_leaves_out_geometries():
    item = _item(geometry=Polygon([(0, 0), (1, 0), (1, 1)]), capacity=100)
    html = FeatureResolver(ResolvedFeature, tooltip=True).resolve_feature(item).tooltip
    assert 'POLYGON' not in html
    assert 'capacity' in html


def test_default_tooltip_escapes_keys_and_values():
    item = _item(**{'a<b>': 'x & <i>y</i>'})
    html = FeatureResolver(ResolvedFeature, tooltip=True).resolve_feature(item).tooltip
    assert '<strong>a&lt;b&gt;</strong>' in html
    assert '>x &amp; &lt;i&gt;y&lt;/i&gt;<' in html


def test_default_popup_escapes_keys_and_values():
    item = _item(**{'a<b>': 'x & <i>y</i>'})
    popup = FeatureResolver(ResolvedFeature, popup=True).resolve_feature(item).popup
    html = popup.html.render()
    assert 'a&lt;b&gt;' in html
    assert 'x &amp; &lt;i&gt;y&lt;/i&gt;' in html
    assert '<i>' not in html


class _ItemWithPlainTooltipData(VisualizableDataItem):
    def get_name(self) -> str:
        return 'item_1'

    def get_text_representation(self) -> str:
        return 'item_1'

    def get_tooltip_data(self) -> dict:
        return {'capacity': '100'}

    def get_object_attribute(self, attribute: str):
        return None

    def object_has_attribute(self, attribute: str) -> bool:
        return False


def test_default_tooltip_and_popup_take_tooltip_data_without_keyword_arguments():
    feature = FeatureResolver(ResolvedFeature, tooltip=True, popup=True).resolve_feature(_ItemWithPlainTooltipData())
    for html in [feature.tooltip, feature.popup.html.render()]:
        assert '<strong>capacity</strong>' in html
        assert '>100<' in html


def test_tooltip_data_shows_a_list_value_as_text():
    item = _item(tags=['a', 'b'], missing=None)
    assert item.get_tooltip_data() == {'ID': 'node_1', 'tags': "['a', 'b']"}


JS_VALUE = 'a`b ${x}'


def test_default_tooltip_and_popup_escape_js_template_characters():
    item = _item(note=JS_VALUE)
    feature = FeatureResolver(ResolvedFeature, tooltip=True, popup=True).resolve_feature(item)
    for html in [feature.tooltip, feature.popup.html.render()]:
        assert 'a&#96;b &#36;{x}' in html
        assert '`' not in html
        assert '${' not in html


def _render_map(generator, model_df: pd.DataFrame) -> str:
    m = folium.Map()
    fg = folium.FeatureGroup(name='test')
    generator.generate_objects_for_model_df(model_df, fg)
    fg.add_to(m)
    return m.get_root().render()


def test_line_map_does_not_render_js_template_characters_of_values():
    model_df = pd.DataFrame(
        {'geometry': [LineString([(7.0, 50.0), (7.5, 52.0)])], 'note': [JS_VALUE]},
        index=['line_1'],
    )
    generator = LineGenerator(LineFeatureResolver(line_color='#000000', tooltip=True, popup=True))
    html = _render_map(generator, model_df)
    assert 'a&#96;b &#36;{x}' in html
    assert JS_VALUE not in html
    assert '${x}' not in html


def test_area_map_tooltip_holds_the_escaped_text_once():
    model_df = pd.DataFrame(
        {'geometry': [Polygon([(0, 0), (1, 0), (1, 1)])], 'note': ['A<B ' + JS_VALUE]},
        index=['area_1'],
    )
    html = _render_map(AreaGenerator(AreaFeatureResolver(tooltip=True)), model_df)
    # GeoJsonTooltip sets the property as innerHTML; the GeoJSON encodes "&" as \u0026.
    assert r'A\u0026lt;B a\u0026#96;b \u0026#36;{x}' in html
    assert r'\u0026amp;' not in html


def test_line_map_does_not_render_backslashes_of_values():
    value = r'C:\units\x \u0041'
    model_df = pd.DataFrame(
        {'geometry': [LineString([(7.0, 50.0), (7.5, 52.0)])], 'note': [value]},
        index=['line_1'],
    )
    generator = LineGenerator(LineFeatureResolver(line_color='#000000', tooltip=True, popup=True))
    html = _render_map(generator, model_df)
    assert html.count('C:&#92;units&#92;x &#92;u0041') == 2
    assert value not in html
