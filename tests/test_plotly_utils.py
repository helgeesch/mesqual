import pandas as pd
import plotly.express as px
import plotly.io as pio
import pytest

from mesqual.utils.plotly_utils import PlotlyTheme, figure_utils, px_category_utils

DATA = pd.DataFrame(
    {
        'x': ['a', 'b'] * 4,
        'y': range(8),
        'row': ['r1'] * 4 + ['r2'] * 4,
        'col': ['c1', 'c1', 'c2', 'c2'] * 2,
    }
)


def facet_figure():
    return px.bar(DATA, x='x', y='y', facet_row='row', facet_col='col')


@pytest.mark.parametrize('row', ['r1', 'r2'])
@pytest.mark.parametrize('col', ['c1', 'c2'])
def test_subplot_of_a_category_holds_the_trace_of_the_category(row, col):
    fig = facet_figure()
    subplot_row, subplot_col = px_category_utils.get_subplot_row_and_col_for_category(fig, {'row': row, 'col': col})
    (trace,) = [t for t in fig.data if f'row={row}' in t.hovertemplate and f'col={col}' in t.hovertemplate]
    subplot = fig.get_subplot(subplot_row, subplot_col)
    assert (subplot.xaxis.plotly_name, subplot.yaxis.plotly_name) == (
        trace.xaxis.replace('x', 'xaxis'),
        trace.yaxis.replace('y', 'yaxis'),
    )


def test_category_index_on_an_axis_and_lookup_errors():
    fig = facet_figure()
    with pytest.raises(KeyError, match='No trace'):
        px_category_utils.get_x_y_axis_for_category(fig, {'row': 'r9'})
    fig.update_xaxes(categoryorder='array', categoryarray=['a', 'b'])
    assert px_category_utils.get_index_for_category_on_axis(fig, 'x', 'b') == 2
    with pytest.raises(KeyError, match='Unknown category'):
        px_category_utils.get_index_for_category_on_axis(fig, 'x', 'c')
    with pytest.raises(TypeError):
        px_category_utils.get_index_for_category_on_axis(fig, 'x', 1)


def test_annotations_lose_the_category_name_and_turn_bold():
    fig = facet_figure()
    figure_utils.remove_category_in_annotations(fig)
    figure_utils.make_annotations_bold(fig)
    assert sorted(annotation.text for annotation in fig.layout.annotations) == [
        '<b>c1</b>',
        '<b>c2</b>',
        '<b>r1</b>',
        '<b>r2</b>',
    ]


@pytest.fixture
def restore_plotly_template():
    default = pio.templates.default
    yield
    pio.templates.default = default
    if 'custom' in pio.templates:
        del pio.templates['custom']


def test_theme_apply_sets_the_default_template(restore_plotly_template):
    PlotlyTheme(default_colorway=['#000000'], watermark_text='draft').apply()
    assert pio.templates.default == 'plotly+custom'
    layout = pio.templates['custom'].layout
    assert layout.colorway == ('#000000',)
    assert layout.annotations[0].text == 'draft'
