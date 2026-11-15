from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from mesqual.study_manager import StudyManager
from mesqual.units import Units

__all__ = [
    'StudyManager',
    'Units',
]

try:
    __version__ = _distribution_version('mesqual')
except PackageNotFoundError:
    __version__ = '0+unknown'
