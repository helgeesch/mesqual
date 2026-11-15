import folium
import pandas as pd
from shapely import Point

from mesqual.visualizations.folium_viz_system.base_viz_system import PropertyMapper
from mesqual.visualizations.folium_viz_system.viz_text_overlay import (
    TextOverlayFeatureResolver,
    TextOverlayGenerator,
)

MODEL = pd.DataFrame({'location': [Point(8.0, 50.0)], 'label': ['A<B & C']}, index=['node_1'])


def _label_html(resolver: TextOverlayFeatureResolver) -> str:
    feature_group = folium.FeatureGroup(name='labels')
    TextOverlayGenerator(resolver).generate_objects_for_model_df(MODEL, feature_group)
    (marker,) = feature_group._children.values()
    (icon,) = [child for child in marker._children.values() if isinstance(child, folium.DivIcon)]
    return icon.options['html']


def test_text_overlay_escapes_the_text():
    html = _label_html(TextOverlayFeatureResolver(text_print_content=PropertyMapper.from_item_attr('label')))
    assert 'A&lt;B &amp; C' in html
    assert 'A<B' not in html


def test_text_overlay_writes_html_text_as_given():
    resolver = TextOverlayFeatureResolver(
        text_print_content=PropertyMapper.from_item_attr('label', lambda text: f'<b>{text}</b>'),
        text_is_html=True,
    )
    assert '<b>A<B & C</b>' in _label_html(resolver)


def test_text_overlay_escapes_a_backslash():
    resolver = TextOverlayFeatureResolver(text_print_content=PropertyMapper.from_item_attr('label', lambda _: r'C:\x'))
    assert 'C:&#92;x' in _label_html(resolver)
