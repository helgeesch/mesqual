from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING


if TYPE_CHECKING:
    from mescal.flag.flag import FlagTypeProtocol
    from mescal.data_sets.data_set import DataSet
    from mescal.data_sets.data_set_config import DataSetConfig
    from mescal.flag.flag_index import FlagIndex
    from mescal.kpis.kpi_base import KPI
    from mescal.kpis.aggs import OperationOfTwoValues
    from mescal.kpis.kpi_collection import KPICollection


FlagType = TypeVar('FlagType', bound='FlagTypeProtocol')
DataSetType = TypeVar('DataSetType', bound='DataSet')
DataSetConfigType = TypeVar('DataSetConfigType', bound='DataSetConfig')
FlagIndexType = TypeVar('FlagIndexType', bound='FlagIndex')
KPIType = TypeVar('KPIType', bound='KPI')
ValueOperationType = TypeVar('ValueOperationType', bound='OperationOfTwoValues')
