import folium
import pytest

from mesqual.visualizations.folium_viz_system.legends.continuous_color import ContinuousColorscaleLegend
from mesqual.visualizations.folium_viz_system.legends.continuous_line_width import ContinuousLineWidthMapLegend
from mesqual.visualizations.folium_viz_system.legends.discrete_color import DiscreteColorLegend
from mesqual.visualizations.folium_viz_system.legends.discrete_line_dash import DiscreteLineDashLegend
from mesqual.visualizations.folium_viz_system.legends.discrete_line_width import DiscreteLineWidthLegend
from mesqual.visualizations.value_mapping_system.continuous import (
    SegmentedContinuousColorscale,
    SegmentedContinuousLineWidthMapping,
)
from mesqual.visualizations.value_mapping_system.discrete import (
    DiscreteColorMapping,
    DiscreteLineDashPatternMapping,
    DiscreteLineWidthMapping,
)


def _suffix(value: float) -> str:
    return f'{value / 1e9:g}B' if value >= 1e9 else f'{value / 1e6:g}M'


@pytest.fixture
def colorscale() -> SegmentedContinuousColorscale:
    return SegmentedContinuousColorscale(segments={(5e6, 2e9): ['#0000ff', '#ff0000']})


@pytest.fixture
def line_width() -> SegmentedContinuousLineWidthMapping:
    return SegmentedContinuousLineWidthMapping(segments={(5e6, 2e9): [1.0, 5.0]})


@pytest.fixture
def discrete_color() -> DiscreteColorMapping:
    return DiscreteColorMapping(mapping={5e6: '#ff0000', 2e9: '#00ff00'})


@pytest.fixture
def discrete_width() -> DiscreteLineWidthMapping:
    return DiscreteLineWidthMapping(mapping={5e6: 1, 2e9: 3})


def test_continuous_legend_default_labels(colorscale):
    html = ContinuousColorscaleLegend(colorscale).render_content()
    assert '>5000000<' in html
    assert '>2000000000<' in html


def test_continuous_legend_custom_formatter(colorscale):
    html = ContinuousColorscaleLegend(colorscale, value_formatter=_suffix).render_content()
    assert '>5M<' in html
    assert '>2B<' in html
    assert '5000000' not in html


def test_continuous_legend_formatter_applies_to_every_tick(colorscale):
    legend = ContinuousColorscaleLegend(colorscale, n_ticks_per_segment=3, value_formatter=lambda v: 'x')
    assert legend.render_content().count('>x<') == 3


def test_continuous_line_width_legend_custom_formatter(line_width):
    html = ContinuousLineWidthMapLegend(line_width, value_formatter=_suffix).render_content()
    assert '>5M<' in html
    assert '>2B<' in html


def test_discrete_color_legend_default_labels(discrete_color):
    html = DiscreteColorLegend(discrete_color).render_content()
    assert '>5000000<' in html
    assert '>2000000000<' in html


def test_discrete_color_legend_custom_formatter(discrete_color):
    html = DiscreteColorLegend(discrete_color, value_formatter=_suffix).render_content()
    assert '>5M<' in html
    assert '>2B<' in html
    assert '5000000' not in html


def test_discrete_line_width_legend_custom_formatter(discrete_width):
    html = DiscreteLineWidthLegend(discrete_width, value_formatter=_suffix).render_content()
    assert '>5M<' in html
    assert '>2B<' in html


def test_value_formatter_none_keeps_default_float_labels():
    mapping = DiscreteColorMapping(mapping={1.5: '#ff0000', 2.0: '#00ff00'})
    html = DiscreteColorLegend(mapping, value_formatter=None).render_content()
    assert '>1.5<' in html
    assert '>2<' in html


def test_default_output_label_is_not_formatted():
    mapping = DiscreteColorMapping(mapping={5e6: '#ff0000'}, default_output='#aaaaaa')
    mapping('unknown')
    html = DiscreteColorLegend(mapping, value_formatter=lambda v: 'x').render_content()
    assert '>x<' in html
    assert '>Other<' in html


def test_discrete_dash_legend_formats_string_keys():
    mapping = DiscreteLineDashPatternMapping(mapping={'Type 1': '', 'Type 2': '5,5'})
    html = DiscreteLineDashLegend(mapping, value_formatter=lambda v: v.upper()).render_content()
    assert '>TYPE 1<' in html
    assert '>TYPE 2<' in html


def test_default_labels_for_string_keys():
    mapping = DiscreteLineDashPatternMapping(mapping={'Type 1': ''})
    assert '>Type 1<' in DiscreteLineDashLegend(mapping).render_content()


@pytest.mark.parametrize('merge, expected_ticks', [(True, 3), (False, 4)])
def test_formatter_applies_to_every_tick_of_two_segments(merge, expected_ticks):
    mapping = SegmentedContinuousColorscale(
        segments={(0, 10): ['#0000ff', '#00ffff'], (10, 20): ['#00ff00', '#ff0000']}
    )
    legend = ContinuousColorscaleLegend(mapping, merge_adjacent_ticks=merge, value_formatter=lambda v: 'x')
    assert legend.render_content().count('>x<') == expected_ticks


def test_continuous_line_width_legend_shows_pixel_values():
    mapping = SegmentedContinuousLineWidthMapping(segments={(0, 10): 1.0, (10, 20): [3.0, 10.0]})
    html = ContinuousLineWidthMapLegend(mapping, show_pixel_values=True).render_content()
    assert 'class="pixel-value">1px<' in html
    assert 'class="pixel-value">3-10px<' in html
    assert 'class="pixel-value">' not in ContinuousLineWidthMapLegend(mapping).render_content()


def test_discrete_line_width_legend_pixel_values_keep_labels(discrete_width):
    legend = DiscreteLineWidthLegend(discrete_width, show_pixel_values=True, value_formatter=_suffix)
    html = legend.render_content()
    assert '5M<span class="pixel-value">(1px)</span>' in html
    assert '2B<span class="pixel-value">(3px)</span>' in html


def test_discrete_line_width_legend_other_keeps_pixel_value():
    mapping = DiscreteLineWidthMapping(mapping={1: 1}, default_output=4)
    mapping(2)
    html = DiscreteLineWidthLegend(mapping, show_pixel_values=True).render_content()
    assert 'Other<span class="pixel-value">(4px)</span>' in html


def test_discrete_legend_rows_in_insertion_order_with_mixed_key_types():
    mapping = DiscreteColorMapping(mapping={'b': '#ff0000', 2: '#00ff00'}, palette=['#0000ff'])
    mapping('a')
    mapping(1)
    html = DiscreteColorLegend(mapping).render_content()
    positions = [html.index(f'>{label}<') for label in ['b', '2', 'a', '1']]
    assert positions == sorted(positions)


def test_discrete_legend_default_row_only_when_default_used():
    mapping = DiscreteColorMapping(mapping={'a': '#ff0000'}, palette=['#0000ff'], default_output='#aaaaaa')
    legend = DiscreteColorLegend(mapping)
    mapping('b')
    assert '>Other<' not in legend.render_content()
    mapping(None)
    assert '>Other<' in legend.render_content()


def test_discrete_legend_reads_mapping_at_render_time():
    mapping = DiscreteColorMapping(palette=['#0000ff'], default_output='#aaaaaa')
    m = folium.Map()
    m.add_child(DiscreteColorLegend(mapping))
    mapping('late value')
    mapping(float('nan'))
    html = m.get_root().render()
    assert '>late value<' in html
    assert 'background-color: #0000ff;' in html
    assert '>Other<' in html


def test_legend_title_is_escaped():
    m = folium.Map()
    m.add_child(DiscreteColorLegend(DiscreteColorMapping(mapping={'a': '#ff0000'}), title='Price <EUR> & tax'))
    html = m.get_root().render()
    assert '>Price &lt;EUR&gt; &amp; tax<' in html
    assert '<EUR>' not in html


def test_discrete_legend_labels_are_escaped():
    mapping = DiscreteColorMapping(mapping={'A<B': '#ff0000', 'C&D': '#00ff00'})
    html = DiscreteColorLegend(mapping).render_content()
    assert '>A&lt;B<' in html
    assert '>C&amp;D<' in html


def test_discrete_line_width_legend_escapes_label_and_keeps_pixel_value():
    mapping = DiscreteLineWidthMapping(mapping={'A<B': 2})
    html = DiscreteLineWidthLegend(mapping, show_pixel_values=True).render_content()
    assert 'A&lt;B<span class="pixel-value">(2px)</span>' in html


def test_value_formatter_output_is_escaped(colorscale):
    html = ContinuousColorscaleLegend(colorscale, value_formatter=lambda v: '<b>x</b>').render_content()
    assert '>&lt;b&gt;x&lt;/b&gt;<' in html
    assert '<b>' not in html


def test_value_formatter_may_return_a_number(colorscale):
    html = ContinuousColorscaleLegend(colorscale, value_formatter=lambda v: round(v / 1e6)).render_content()
    assert '>5<' in html
    assert '>2000<' in html


@pytest.mark.parametrize(
    'legend_type, mapping',
    [
        (DiscreteColorLegend, DiscreteColorMapping(mapping={'a': '#ff0000'}, default_output='#aaaaaa')),
        (DiscreteLineDashLegend, DiscreteLineDashPatternMapping(mapping={'a': '5,5'}, default_output='1,1')),
        (DiscreteLineWidthLegend, DiscreteLineWidthMapping(mapping={'a': 1}, default_output=4)),
    ],
)
def test_discrete_legend_default_label_names_the_default_row(legend_type, mapping):
    mapping('b')
    legend = legend_type(mapping, default_label='Rest <EU> & more', value_formatter=str.upper)
    html = legend.render_content()
    assert '>Rest &lt;EU&gt; &amp; more<' in html
    assert 'Other' not in html


def test_discrete_line_width_legend_default_label_keeps_pixel_value():
    mapping = DiscreteLineWidthMapping(mapping={1: 1}, default_output=4)
    mapping(2)
    html = DiscreteLineWidthLegend(mapping, show_pixel_values=True, default_label='Rest').render_content()
    assert 'Rest<span class="pixel-value">(4px)</span>' in html


def test_discrete_legend_default_label_may_be_a_number():
    mapping = DiscreteColorMapping(mapping={'a': '#ff0000'}, default_output='#aaaaaa')
    mapping('b')
    assert '>0<' in DiscreteColorLegend(mapping, default_label=0).render_content()
