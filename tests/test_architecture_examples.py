"""The worked example of the architecture docs runs as shown.

The `python` blocks of `docs/architecture/example-trade-balance.md` run in
order as one script, under fixed pandas display options, so the printed
frames do not depend on the terminal width. Each block that prints is
followed by a `text` block with the same output. The `text` block right
after the block that assigns `info` is the abridged `info` dict: it equals
the same keys of `info`.
"""

import ast
from pathlib import Path

import pandas as pd
import pytest

from mesqual.energy_data_handling.network_lines_data import NetworkLineFlowsData
from mesqual.energy_data_handling.variable_utils import RegionalTradeBalanceCalculator
from tests.code_blocks import fence_tags
from tests.helpers import run_code_blocks, text_lines

EXAMPLE = Path(__file__).parents[1] / 'docs' / 'architecture' / 'example-trade-balance.md'
INFO_ASSIGNMENT = 'info = '


@pytest.fixture(scope='module')
def example(tmp_path_factory) -> tuple[dict, list[tuple[str, str, str]]]:
    """Run the python blocks once; return the namespace and each block as `(tag, body, printed output)`."""
    namespace = {'__name__': '__main__'}
    return namespace, run_code_blocks(EXAMPLE, namespace, tmp_path_factory.mktemp('example'))


def test_the_example_page_has_only_python_and_text_blocks_with_unindented_fences():
    assert fence_tags(EXAMPLE.read_text(encoding='utf-8')) == {'python', 'text'}


def test_each_printed_output_and_the_info_record_are_shown_in_the_text_block_after_them(example):
    namespace, runs = example
    shown = set()
    for position, (tag, body, printed) in enumerate(runs):
        if tag != 'python' or not (printed or INFO_ASSIGNMENT in body):
            continue
        assert position + 1 < len(runs) and runs[position + 1][0] == 'text', f'block {position + 1} needs a text block'
        text = runs[position + 1][1]
        if printed:
            assert text_lines(printed) == text_lines(text), f'block {position + 2} differs from the printed output'
        else:
            record = ast.literal_eval(text)
            assert record == {key: namespace['info'][key] for key in record}, f'block {position + 2} differs from info'
        shown.add(position + 1)
    assert shown == {position for position, (tag, _, _) in enumerate(runs) if tag == 'text'}


def test_the_example_result_equals_the_calculator_and_is_in_the_cache(example):
    namespace, _ = example
    ds, record = namespace['ds'], namespace['record']
    assert ds.cache.entry(namespace['CacheKey'].of(record)) is not None
    flows = NetworkLineFlowsData(
        sent_up=ds.fetch('Line.Results.flow_sent_up'),
        received_up=ds.fetch('Line.Results.flow_received_up'),
        sent_down=ds.fetch('Line.Results.flow_sent_down'),
        received_down=ds.fetch('Line.Results.flow_received_down'),
    )
    calculator = RegionalTradeBalanceCalculator(ds.fetch('Line.Model'), ds.fetch('Node.Model'), 'country')
    expected = calculator.get_trade_balance(flows, 'at_own_end')
    result = namespace['trade_balance']
    pd.testing.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1), check_names=False)
