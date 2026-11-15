"""Run the `>>>` examples in the docstrings of the package, one test per docstring.

An example sees the names of its module and the shared names of
`shared_names`. The test restores the quantity types of `Units`, and
`tests/conftest.py` clears the class configs, so that the examples cannot
change other tests.
"""

import doctest
import importlib
import pkgutil

import pandas as pd
import pytest

import mesqual
from mesqual.testing import make_dummy_study
from mesqual.units import Units

OPTION_FLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE


def docstring_examples() -> list[doctest.DocTest]:
    """The docstrings with `>>>` examples of the package."""
    finder = doctest.DocTestFinder()
    return [
        test
        for info in pkgutil.walk_packages(mesqual.__path__, 'mesqual.')
        for test in finder.find(importlib.import_module(info.name))
        if test.examples
    ]


def shared_names() -> dict:
    """The names that every example can use without an import."""
    study = make_dummy_study()
    return {
        'pd': pd,
        'study': study,
        'dataset': study.scen.get_dataset('base'),
    }


@pytest.mark.parametrize('test', docstring_examples(), ids=lambda test: test.name)
def test_docstring_example(test, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Units, '_QUANTITY_TYPES', dict(Units._QUANTITY_TYPES))
    test.globs = {**test.globs, **shared_names()}
    runner = doctest.DocTestRunner(optionflags=OPTION_FLAGS)
    report = []
    result = runner.run(test, out=report.append)
    assert result.failed == 0, ''.join(report)
