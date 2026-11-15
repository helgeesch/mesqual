"""The guide pages run as shown.

The `python` blocks of each page in `docs/guides/` run in order, as one script
per page, in a new folder and under fixed pandas display options, so the printed
frames do not depend on the terminal width. Each python block that prints is
followed by a `text` block with the same output.
"""

from pathlib import Path

import pandas as pd
import pytest

from tests.code_blocks import fence_tags
from tests.helpers import ROOT, run_code_blocks, text_lines

GUIDES = ROOT / 'docs' / 'guides'
PAGES = sorted(path.name for path in GUIDES.glob('*.md'))


@pytest.fixture(scope='module')
def guide(request, tmp_path_factory) -> tuple[dict, Path, list[tuple[str, str, str]]]:
    """The run of the page `request.param`: namespace, working folder and blocks `(tag, body, printed output)`.

    Module scope: each page runs once, and a broken page fails only the tests of that page.
    """
    namespace, folder = {'__name__': 'guide_examples'}, tmp_path_factory.mktemp(Path(request.param).stem)
    return namespace, folder, run_code_blocks(GUIDES / request.param, namespace, folder)


@pytest.mark.parametrize('page', PAGES)
def test_the_guide_has_only_python_and_text_blocks(page):
    assert fence_tags((GUIDES / page).read_text(encoding='utf-8')) <= {'python', 'text'}


@pytest.mark.parametrize('guide', PAGES, indirect=True)
def test_the_guide_python_blocks_run_in_order_and_each_printed_output_is_the_text_block_after_it(guide):
    _, _, blocks = guide
    shown = set()
    for position, (tag, _, printed) in enumerate(blocks):
        if tag != 'python' or not printed:
            continue
        next_tag, next_body, _ = blocks[position + 1] if position + 1 < len(blocks) else ('', '', '')
        assert next_tag == 'text', f'block {position + 1} needs a text block'
        assert text_lines(printed) == text_lines(next_body), f'block {position + 2} differs from the output'
        shown.add(position + 1)
    assert shown == {position for position, (tag, _, _) in enumerate(blocks) if tag == 'text'}


@pytest.mark.parametrize('guide', ['use-in-a-study.md'], indirect=True)
def test_the_study_guide_saves_the_map_and_the_dashboard_with_the_kpis_of_each_scenario(guide):
    _, folder, _ = guide
    price_map = (folder / 'price_map.html').read_text(encoding='utf-8')
    for scenario in ('base', 'high_demand', 'without_N4'):
        for country in ('C1', 'C2'):
            assert f'{scenario} Country.Results.price Mean {country}' in price_map
    dashboard = (folder / 'study_dashboard.html').read_text(encoding='utf-8')
    assert 'Mean price (EUR/MWh)' in dashboard


@pytest.mark.parametrize('guide', ['extend-a-platform-in-a-study.md'], indirect=True)
def test_the_extend_guide_custom_variable_follows_the_config_of_the_call(guide):
    namespace, _, _ = guide
    dataset, first_solution = namespace['dataset'], namespace['first_solution']
    demand = dataset.fetch('Node.Results.demand_total')
    demand_change = dataset.fetch('Node.Results.demand_total', config=first_solution) - demand
    assert not demand_change.eq(0).all().all()
    expected = namespace['net_position'] - demand_change
    pd.testing.assert_frame_equal(namespace['net_position_first'], expected)
