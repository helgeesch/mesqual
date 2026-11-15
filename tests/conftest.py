import pytest

from mesqual.datasets.dataset import _DuplicateIndexWarnings, _NotStoredLogs, _UnknownConfigFieldWarnings
from mesqual.datasets.dataset_config import DatasetConfigManager


@pytest.fixture(autouse=True)
def clear_class_configs():
    DatasetConfigManager.clear_class_configs()
    yield
    DatasetConfigManager.clear_class_configs()


@pytest.fixture(autouse=True)
def clear_unknown_config_field_warnings():
    _UnknownConfigFieldWarnings.clear()
    yield
    _UnknownConfigFieldWarnings.clear()


@pytest.fixture(autouse=True)
def clear_duplicate_index_warnings():
    _DuplicateIndexWarnings.clear()
    yield
    _DuplicateIndexWarnings.clear()


@pytest.fixture(autouse=True)
def clear_not_stored_logs():
    _NotStoredLogs.clear()
    yield
    _NotStoredLogs.clear()
