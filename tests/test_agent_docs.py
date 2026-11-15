"""Links and API names resolve in the agent texts (skill, llms.txt, docs page) and the README; conventions hold."""

import importlib
import pkgutil
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

import mesqual
from mesqual.datasets import DatasetComparison, PlatformDataset
from mesqual.enums import ComparisonTypeEnum
from mesqual.testing import DummyDataset, DummyScenario, make_dummy_study
from tests.helpers import ROOT, SKILL

LLMS_TXT = ROOT / 'docs' / 'llms.txt'
PAGE = ROOT / 'docs' / 'use-with-agents.md'
AGENT_TEXTS = [SKILL, LLMS_TXT, PAGE]
TEXTS = pytest.mark.parametrize('path', [*AGENT_TEXTS, ROOT / 'README.md'], ids=lambda path: path.name)

DOCS_URL = 'https://docs.mesqual.io'
DOCS_LINK = re.compile(r'https://docs\.mesqual\.io[\w/#.\-]*')
RELATIVE_LINK = re.compile(r'\]\(([\w/\-]+\.md(?:#[\w.\-]+)?)\)')
SNIPPET = re.compile(r'^```markdown\n(.*?)^```$', re.MULTILINE | re.DOTALL)
FULL_CALL = re.compile(r'`([A-Za-z_][\w.]*\([^`]*\))`')
LLMS_ITEM = re.compile(r'^- \[[^\]]+\]\(https://docs\.mesqual\.io/\S*\): \S.*$')
BACKTICKED = re.compile(r'`([^`\n]+)`')
DOTTED = re.compile(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*')
MISSING = object()


@pytest.fixture(scope='session')
def site(tmp_path_factory) -> Path:
    """The docs site, built once per session. Needs the extra `docs`."""
    pytest.importorskip('mkdocs')
    folder = tmp_path_factory.mktemp('site')
    command = [sys.executable, '-m', 'mkdocs', 'build', '--quiet', '-f', str(ROOT / 'mkdocs.yml'), '-d', str(folder)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        pytest.fail(result.stderr)
    return folder


@pytest.fixture(scope='module')
def namespace() -> dict:
    """The public names of mesqual, the aliases `folviz` and `valmap`, and the names of the synthetic study.

    A public name of a module counts if it is defined in mesqual or is a module of mesqual. `study` and
    `dataset` come from the synthetic study. `StudyDataset`, the study subclass of the texts, is
    `PlatformDataset`, so only the names that every platform dataset has resolve.
    """
    names = {'mesqual': mesqual}
    for info in pkgutil.walk_packages(mesqual.__path__, 'mesqual.'):
        module = importlib.import_module(info.name)
        public = getattr(module, '__all__', None) or [name for name in vars(module) if not name.startswith('_')]
        names.update({name: getattr(module, name) for name in public if from_mesqual(getattr(module, name))})
    study = make_dummy_study()
    return names | {
        'folviz': importlib.import_module('mesqual.visualizations.folium_viz_system'),
        'valmap': importlib.import_module('mesqual.visualizations.value_mapping_system'),
        'study': study,
        'dataset': study.scen.get_dataset('base'),
        'StudyDataset': PlatformDataset,
    }


def from_mesqual(value: Any) -> bool:
    """`True` for a module of mesqual and for an object whose `__module__` is one."""
    module = value.__name__ if isinstance(value, ModuleType) else getattr(value, '__module__', None)
    return isinstance(module, str) and module.split('.')[0] == 'mesqual'


def link_resolves(site: Path, link: str) -> bool:
    """`True` if the site has the page `<path>/index.html` of the link, with its anchor."""
    path, _, anchor = link.removeprefix(DOCS_URL).lstrip('/').partition('#')
    page = site / path / 'index.html'
    return page.is_file() and (not anchor or f'id="{anchor}"' in page.read_text(encoding='utf-8'))


def api_names(text: str, namespace: dict) -> list[str]:
    """The backticked API names of a text, up to their first call.

    A name with a call is always an API name. A dotted name without a call is
    one if its first name is in `namespace`; other names, for example the flag
    `Node.Results.price`, are not. A leading `@` of a decorator is dropped.
    """
    names = []
    for token in BACKTICKED.findall(text):
        token = token.removeprefix('@')
        match = DOTTED.match(token)
        rest = token[match.end() :] if match else None
        if match and (rest.startswith('(') or (not rest and match.group().split('.')[0] in namespace)):
            names.append(match.group())
    return names


def name_resolves(name: str, namespace: dict) -> bool:
    """`True` if each part of `name` is an attribute, or a dataclass field without class default, of the part before."""
    first, *attributes = name.split('.')
    value = namespace.get(first, MISSING)
    for attribute in attributes:
        fields = getattr(value, '__dataclass_fields__', {})
        value = getattr(value, attribute, fields[attribute] if attribute in fields else MISSING)
    return value is not MISSING


def site_links(text: str) -> list[str]:
    """The docs.mesqual.io links of a text, and its relative `.md` links as docs.mesqual.io links."""
    relative = [DOCS_URL + '/' + re.sub(r'(index)?\.md', '/', link, count=1) for link in RELATIVE_LINK.findall(text)]
    return DOCS_LINK.findall(text) + relative


@TEXTS
def test_the_docs_links_resolve_on_the_built_site(site, path):
    links = site_links(path.read_text(encoding='utf-8'))
    assert links
    assert [link for link in links if not link_resolves(site, link)] == []


@TEXTS
def test_the_api_names_resolve(namespace, path):
    names = api_names(path.read_text(encoding='utf-8'), namespace)
    assert names
    assert [name for name in names if not name_resolves(name, namespace)] == []


def test_llms_txt_follows_the_format_and_links_only_the_docs_site():
    lines = LLMS_TXT.read_text(encoding='utf-8').splitlines()
    assert 30 <= len(lines) <= 50
    assert lines[0].startswith('# ')
    assert next(line for line in lines[1:] if line).startswith('> ')
    sections = [number for number, line in enumerate(lines) if line.startswith('## ')]
    assert sections
    items = [line for line in lines[sections[0] :] if line and not line.startswith('## ')]
    assert all(LLMS_ITEM.match(item) for item in items)


def test_the_calls_of_the_snippet_run_on_the_synthetic_study(namespace):
    (snippet,) = SNIPPET.findall(PAGE.read_text(encoding='utf-8'))
    calls = FULL_CALL.findall(snippet)
    assert calls
    for call in calls:
        eval(call, {**namespace, 'flag': 'Node.Results.price'})


def test_the_fetch_conventions_hold_on_the_synthetic_study():
    study = make_dummy_study()
    flag = 'Node.Results.price'
    prices, deltas, prices_and_deltas = (
        collection.fetch(flag) for collection in (study.scen, study.comp, study.scen_comp)
    )
    assert list(prices.columns.names) == list(deltas.columns.names) == ['dataset', 'Node']
    assert list(prices_and_deltas.columns.names) == ['type', 'dataset', 'Node']
    assert set(prices_and_deltas.columns.unique('type')) == {'scenario', 'comparison'}
    assert list(deltas.columns.unique('dataset')) == ['high_demand vs base', 'without_N4 vs base', 'short vs base']
    assert prices.attrs == {} and deltas.attrs == {}
    both = study.comp.fetch(flag, comparison_type=ComparisonTypeEnum.BOTH)
    assert both.columns.names[-1] == 'compared'
    assert 'compared' not in [level.name for level in study.comp.flag_info(flag).schema.columns]
    assert study.scen.get_dataset('base').fetch('Node.Model').columns.names == [None]
    assert study.scen.flag_info('Node.Model').schema.columns[-1].name is None

    outside_short = prices['short'].isna().all(axis=1)
    assert outside_short.any()
    gas = 'Node.Results.generation_gas'  # fill_value 0
    assert study.scen.get_dataset('base').flag_info(gas).fill_value == 0
    gas_deltas = study.comp.fetch(gas)['short vs base'][outside_short]
    pd.testing.assert_frame_equal(gas_deltas, -study.scen.fetch(gas)['base'][outside_short])


def test_the_text_markers_of_a_comparison():
    study = make_dummy_study()
    base, without_n4 = study.scen.get_dataset('base'), study.scen.get_dataset('without_N4')
    deleted = DatasetComparison(without_n4, base).fetch('Node.Model')['country']
    new = DatasetComparison(base, without_n4).fetch('Node.Model')['country']
    three_countries = DummyDataset(DummyScenario(name='three_countries', n_countries=3))
    changed = DatasetComparison(three_countries, DummyDataset(DummyScenario())).fetch('Node.Model')['country']
    assert deleted['N4'] == 'DELETED: C2' and new['N4'] == 'C2 (new)'
    assert deleted['N1'] == base.fetch('Node.Model')['country']['N1']
    assert changed.str.contains(' (was ', regex=False).any()
