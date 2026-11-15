import logging
import re

import folium
import pandas as pd
import plotly.graph_objects as go
import pytest

from mesqual.visualizations.html_dashboard import HTMLDashboard
from mesqual.visualizations.html_table import HTMLTable


def test_page_title_is_escaped(tmp_path):
    path = tmp_path / 'dashboard.html'
    HTMLDashboard(name='Prices <A & B>').save(str(path))
    assert '<title>Prices &lt;A &amp; B&gt;</title>' in path.read_text(encoding='utf-8')


def test_tab_labels_stay_raw_html(tmp_path):
    path = tmp_path / 'dashboard.html'
    dashboard = HTMLDashboard()
    dashboard.add_html('<p>content</p>', name='element', tab='<b>Prices</b>')
    dashboard.save(str(path))
    assert '><b>Prices</b></button>' in path.read_text(encoding='utf-8')


def test_two_dividers_with_the_same_title_both_appear(tmp_path):
    path = tmp_path / 'dashboard.html'
    dashboard = HTMLDashboard()
    first = dashboard.add_section_divider('Prices', subtitle='first', tab='One')
    second = dashboard.add_section_divider('Prices', subtitle='second', tab='Two')
    dashboard.save(str(path))

    assert first != second
    assert list(dashboard.content) == [first, second]
    text = path.read_text(encoding='utf-8')
    assert 'first' in text and 'second' in text


def test_explicit_duplicate_name_logs_warning_and_replaces(caplog):
    dashboard = HTMLDashboard()
    dashboard.add_html('<p>old</p>', name='element')
    with caplog.at_level(logging.WARNING):
        dashboard.add_html('<p>new</p>', name='element')

    assert dashboard.content['element'].element == '<p>new</p>'
    assert 'element' in caplog.text


def test_auto_names_of_untitled_tables_with_the_same_id_get_a_suffix():
    dashboard = HTMLDashboard()
    first = dashboard.add_table(HTMLTable(pd.DataFrame({'A': [1]}), table_id='grid'))
    second = dashboard.add_table(HTMLTable(pd.DataFrame({'A': [2]}), table_id='grid'))
    assert (first, second) == ('grid', 'grid_2')


def test_auto_names_of_tables_with_the_same_title_get_a_suffix():
    dashboard = HTMLDashboard()
    first = dashboard.add_table(HTMLTable(pd.DataFrame({'A': [1]}), title='Prices'))
    second = dashboard.add_table(HTMLTable(pd.DataFrame({'A': [2]}), title='Prices'))
    assert (first, second) == ('table_prices', 'table_prices_2')


def test_auto_names_of_folium_maps_get_a_suffix():
    dashboard = HTMLDashboard()
    names = [dashboard.add_folium_map(folium.Map()) for _ in range(3)]
    assert names == ['folium_map', 'folium_map_2', 'folium_map_3']


class _DashboardPageTest:
    @staticmethod
    def _save(dashboard, path):
        dashboard.save(str(path))
        return path.read_text(encoding='utf-8')


class TestPlotlyJsSource(_DashboardPageTest):
    CDN_SCRIPT = '<script charset="utf-8" src="https://cdn.plot.ly/'
    EMBEDDED_MARKER = '* plotly.js v'
    LAYOUTS = [[None, None, None], ['One', ('One', 'Sub'), 'Two']]

    @staticmethod
    def _figure():
        return go.Figure(go.Bar(x=['A', 'B'], y=[1, 2]))

    def _dashboard(self, tabs, **kwargs):
        dashboard = HTMLDashboard(**kwargs)
        for i, tab in enumerate(tabs):
            dashboard.add_plotly_figure(self._figure(), name=f'fig_{i}', tab=tab)
        return dashboard

    def _assert_embedded_once(self, text):
        assert text.count(self.EMBEDDED_MARKER) == 1
        assert self.CDN_SCRIPT not in text

    @pytest.mark.parametrize('tabs', LAYOUTS)
    def test_default_embeds_plotly_js_once(self, tmp_path, tabs):
        self._assert_embedded_once(self._save(self._dashboard(tabs), tmp_path / 'dashboard.html'))

    @pytest.mark.parametrize('tabs', LAYOUTS)
    def test_cdn_loads_plotly_js_with_one_script_tag(self, tmp_path, tabs):
        text = self._save(self._dashboard(tabs, plotly_js='cdn'), tmp_path / 'dashboard.html')
        assert text.count(self.CDN_SCRIPT) == 1
        assert self.EMBEDDED_MARKER not in text
        assert text.count('class="plotly-graph-div"') == len(tabs)
        assert text.index(self.CDN_SCRIPT) < text.index('Plotly.newPlot')

    def test_each_save_embeds_plotly_js_once(self, tmp_path):
        dashboard = self._dashboard(self.LAYOUTS[0])
        self._assert_embedded_once(self._save(dashboard, tmp_path / 'first.html'))
        self._assert_embedded_once(self._save(dashboard, tmp_path / 'second.html'))

    def test_switch_between_saves(self, tmp_path):
        dashboard = self._dashboard(self.LAYOUTS[0])
        self._assert_embedded_once(self._save(dashboard, tmp_path / 'embedded.html'))
        dashboard.plotly_js = 'cdn'
        text = self._save(dashboard, tmp_path / 'cdn.html')
        assert text.count(self.CDN_SCRIPT) == 1
        assert self.EMBEDDED_MARKER not in text
        dashboard.plotly_js = 'embed'
        self._assert_embedded_once(self._save(dashboard, tmp_path / 'embedded_again.html'))

    @pytest.mark.parametrize('value', ['inline', ['cdn']])
    def test_unknown_value_raises(self, value):
        with pytest.raises(ValueError, match='plotly_js'):
            HTMLDashboard(plotly_js=value)

    def test_unknown_value_set_after_init_raises_on_save(self, tmp_path):
        dashboard = self._dashboard(self.LAYOUTS[0])
        dashboard.plotly_js = 'inline'
        with pytest.raises(ValueError, match='plotly_js'):
            dashboard.save(str(tmp_path / 'dashboard.html'))


class TestPageAndFigureSizing(_DashboardPageTest):
    def test_page_renders_in_standards_mode(self, tmp_path):
        dashboard = HTMLDashboard()
        dashboard.add_html('<p>x</p>', name='element')
        text = self._save(dashboard, tmp_path / 'dashboard.html')
        assert text.startswith('<!doctype html>\n<html lang="en"><head>')
        assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in text

    def test_figures_are_responsive_with_and_without_force_height(self, tmp_path):
        dashboard = HTMLDashboard()
        dashboard.add_plotly_figure(go.Figure(go.Bar(y=[1]), layout_height=300), name='fixed')
        dashboard.add_plotly_figure(go.Figure(go.Bar(y=[2])), name='free')
        dashboard.add_plotly_figure(
            go.Figure(go.Bar(y=[3]), layout_height=300), height='500px', name='forced', force_height=True
        )
        text = self._save(dashboard, tmp_path / 'dashboard.html')
        assert text.count('{"responsive": true}') == 3
        assert '"responsive": false' not in text
        assert text.count('<div style="height:300px; width:100%;">') == 1
        assert text.count('<div style="height:100%; width:100%;">') == 1
        assert text.count('<div style="height:500px; width:100%;">') == 1

    @pytest.mark.parametrize(
        'tab, written, omitted',
        [
            (None, HTMLDashboard._scroll_css(), HTMLDashboard._tab_css()),
            ('One', HTMLDashboard._tab_css(), HTMLDashboard._scroll_css()),
        ],
    )
    def test_page_css_matches_the_layout(self, tmp_path, tab, written, omitted):
        dashboard = HTMLDashboard()
        dashboard.add_plotly_figure(go.Figure(go.Bar(y=[1])), name='fig', tab=tab)
        text = self._save(dashboard, tmp_path / 'dashboard.html')
        assert written in text
        assert omitted not in text

    def test_scrollable_css_gives_the_body_the_window_height(self, tmp_path):
        dashboard = HTMLDashboard()
        dashboard.add_plotly_figure(go.Figure(go.Bar(y=[1])), name='fig')
        text = self._save(dashboard, tmp_path / 'dashboard.html')
        assert re.search(r'html, body \{ height: 100%; \}', text)
        assert 'box-sizing: border-box' in text

    def test_scrollable_figure_is_direct_child_of_body(self, tmp_path):
        dashboard = HTMLDashboard()
        dashboard.add_plotly_figure(go.Figure(go.Bar(y=[1])), name='fig')
        text = self._save(dashboard, tmp_path / 'dashboard.html')
        assert '</head><body>\n<div style="height:100%; width:100%;">' in text
