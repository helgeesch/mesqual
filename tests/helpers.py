"""Helpers for the tests: small datasets and caches, and the Markdown texts with runners for their code blocks."""

import contextlib
import io
from pathlib import Path

import pandas as pd

from mesqual.cache import Cache, Entry
from mesqual.datasets.dataset import Dataset
from mesqual.flag import FrameSchema, Level, OtherFlagSpec
from tests.code_blocks import code_blocks, python_blocks


class FrameDataset(Dataset):
    FLAG = 'items'
    flag_specs = (
        OtherFlagSpec(name=FLAG, description='Test items.', schema=FrameSchema(index=(Level('item', 'dimension'),))),
    )

    def _fetch(self, flag, effective_config, **kwargs) -> pd.DataFrame:
        return pd.DataFrame({'value': [1.0, 2.0], 'country': ['DE', 'FR']}, index=['a', 'b'])


class InMemoryCache(Cache):
    """Keeps the entries in a dict: cache key to its `(record, value)` entry."""

    def __init__(self):
        self.store = {}

    def entry(self, key):
        return self.store.get(key)

    def set(self, key, value, record):
        self.store[key] = (record, value)

    def delete(self, dataset=None, flag=None):
        for key in self.list_keys(dataset, flag):
            del self.store[key]

    def list_keys(self, dataset=None, flag=None):
        return [key for key in self.store if key.matches(dataset, flag)]


def only_entry(cache: Cache, dataset: Dataset, flag) -> Entry:
    """The entry of the only key of `flag` of `dataset`."""
    (key,) = cache.list_keys(dataset, flag)
    return cache.entry(key)


# Markdown texts and their python blocks.

ROOT = Path(__file__).parents[1]
SKILL = ROOT / 'src' / 'mesqual' / 'skills' / 'mesqual-study' / 'SKILL.md'


def run_python_blocks(path: Path, namespace: dict) -> None:
    """Run the `python` code blocks of a Markdown file in order, as one script in `namespace`."""
    blocks = python_blocks(path.read_text(encoding='utf-8'))
    assert blocks, f'{path.name} has no python blocks'
    for number, block in enumerate(blocks, start=1):
        exec(compile(block, f'{path.name}, python block {number}', 'exec'), namespace)


def run_code_blocks(path: Path, namespace: dict, folder: Path) -> list[tuple[str, str, str]]:
    """Run the `python` code blocks of a Markdown file in order, as one script in `namespace`.

    The blocks run in `folder` and under fixed pandas display options, so
    printed frames do not depend on the terminal width.

    Returns:
        Every code block as `(language tag, body, printed output)`, in order.
        A block that is not `python` has no printed output.
    """
    runs = []
    with contextlib.chdir(folder), pd.option_context('display.max_columns', None, 'display.width', 1000):
        for number, (tag, body) in enumerate(code_blocks(path.read_text(encoding='utf-8')), start=1):
            stdout = io.StringIO()
            if tag == 'python':
                with contextlib.redirect_stdout(stdout):
                    exec(compile(body, f'{path.name}, block {number}', 'exec'), namespace)
            runs.append((tag, body, stdout.getvalue()))
    return runs


def text_lines(text: str) -> list[str]:
    """The lines of a text without trailing spaces and without leading and trailing empty lines."""
    return [line.rstrip() for line in text.strip('\n').splitlines()]
