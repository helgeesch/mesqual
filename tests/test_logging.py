import logging
import subprocess
import sys

import pytest

from mesqual.utils.logging import ConsoleLogging

IMPORT_ALL_MODULES = """
import importlib, logging, pkgutil
import mesqual
for module in pkgutil.walk_packages(mesqual.__path__, 'mesqual.', onerror=lambda name: None):
    try:
        importlib.import_module(module.name)
    except ImportError:
        pass  # modules with optional dependencies
loggers = [logging.getLogger(name) for name in list(logging.root.manager.loggerDict) if name.startswith('mesqual')]
assert loggers, 'no mesqual loggers'
own = [l.name for l in loggers if isinstance(l, logging.Logger) and (l.handlers or l.level != logging.NOTSET or not l.propagate)]
assert not own, own
assert not logging.root.handlers, logging.root.handlers
"""


@pytest.fixture(autouse=True)
def reset_console_logging():
    yield
    ConsoleLogging.disable()


class _Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def test_importing_mesqual_adds_no_handler_level_or_propagation_change():
    subprocess.run([sys.executable, '-c', IMPORT_ALL_MODULES], check=True)


def test_a_root_handler_receives_mesqual_records():
    from mesqual.datasets import dataset

    handler = _Records()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        dataset.logger.warning('from mesqual')
    finally:
        root.removeHandler(handler)
    assert [record.name for record in handler.records] == ['mesqual.datasets.dataset']


class TestConsoleLogging:
    def test_enable_adds_one_handler_and_sets_the_level(self):
        ConsoleLogging.enable(logging.DEBUG)
        ConsoleLogging.enable(logging.DEBUG)
        logger = logging.getLogger('mesqual')
        assert len(logger.handlers) == 1
        assert logger.level == logging.DEBUG
        assert logger.propagate

    def test_disable_removes_the_handler_and_restores_the_level(self):
        logger = logging.getLogger('mesqual')
        logger.setLevel(logging.WARNING)
        try:
            ConsoleLogging.enable(logging.DEBUG)
            ConsoleLogging.enable(logging.INFO)
            ConsoleLogging.disable()
            assert logger.handlers == []
            assert logger.level == logging.WARNING
        finally:
            logger.setLevel(logging.NOTSET)

    def test_an_unknown_level_raises_and_adds_no_handler(self):
        with pytest.raises(ValueError):
            ConsoleLogging.enable('debug')
        logger = logging.getLogger('mesqual')
        assert logger.handlers == []
        assert logger.level == logging.NOTSET

    def test_an_unknown_level_on_a_second_enable_keeps_the_working_handler(self):
        logger = logging.getLogger('mesqual')
        logger.setLevel(logging.WARNING)
        try:
            ConsoleLogging.enable(logging.DEBUG)
            handlers = list(logger.handlers)
            with pytest.raises(ValueError):
                ConsoleLogging.enable('debug')
            assert logger.handlers == handlers
            assert logger.level == logging.DEBUG
            ConsoleLogging.disable()
            assert logger.level == logging.WARNING
        finally:
            logger.setLevel(logging.NOTSET)

    def test_disable_keeps_a_level_that_the_application_set_after_enable(self):
        logger = logging.getLogger('mesqual')
        try:
            ConsoleLogging.enable(logging.DEBUG)
            logger.setLevel(logging.ERROR)
            ConsoleLogging.disable()
            assert logger.handlers == []
            assert logger.level == logging.ERROR
        finally:
            logger.setLevel(logging.NOTSET)

    def test_without_colorlog_the_output_is_plain_text(self, monkeypatch):
        monkeypatch.setitem(sys.modules, 'colorlog', None)
        ConsoleLogging.enable(colored=True)
        assert type(logging.getLogger('mesqual').handlers[0].formatter) is logging.Formatter

    def test_records_print_to_the_console(self, capsys):
        ConsoleLogging.enable(logging.DEBUG, colored=False)
        logging.getLogger('mesqual.datasets.dataset').debug('stale entry')
        assert 'mesqual.datasets.dataset - DEBUG - stale entry' in capsys.readouterr().err

    def test_colored_uses_colorlog(self):
        colorlog = pytest.importorskip('colorlog')
        ConsoleLogging.enable(colored=True)
        assert isinstance(logging.getLogger('mesqual').handlers[0].formatter, colorlog.ColoredFormatter)
        ConsoleLogging.enable(colored=False)
        assert not isinstance(logging.getLogger('mesqual').handlers[0].formatter, colorlog.ColoredFormatter)
