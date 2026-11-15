import re

from tests.code_blocks import fence_tags
from tests.helpers import ROOT, run_python_blocks

README = ROOT / 'README.md'


def test_the_readme_quickstart_fetches_all_scenarios_and_all_comparisons(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {'__name__': 'readme_examples'}
    run_python_blocks(README, namespace)
    scenarios = namespace['prices'].columns.unique('dataset')
    comparisons = namespace['price_deltas'].columns.unique('dataset')
    assert list(scenarios) == ['base', 'high_demand', 'without_N4', 'short']
    assert list(comparisons) == ['high_demand vs base', 'without_N4 vs base', 'short vs base']


def test_the_readme_code_blocks_are_python_or_bash():
    assert fence_tags(README.read_text(encoding='utf-8')) <= {'python', 'bash'}


def test_the_readme_links_are_absolute():
    targets = re.findall(r'\]\(([^)]*)\)', README.read_text(encoding='utf-8'))
    assert targets
    assert [target for target in targets if not target.startswith('https://')] == []
