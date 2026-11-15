import importlib
import importlib.metadata

import mesqual


def test_version_is_the_package_metadata_version():
    assert mesqual.__version__ == importlib.metadata.version('mesqual')


def test_version_without_metadata_is_unknown(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, 'version', missing)
    try:
        importlib.reload(mesqual)
        assert mesqual.__version__ == '0+unknown'
    finally:
        monkeypatch.undo()
        importlib.reload(mesqual)
    assert mesqual.__version__ == importlib.metadata.version('mesqual')
