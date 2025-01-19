from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING

from mescal.flag.flag import Flagtype

if TYPE_CHECKING:
    from mescal.data_sets.data_set import DataSet
    from mescal.data_sets.data_set_config import DataSetConfig
    from mescal.kpis.kpi_base import KPI
    from mescal.kpis.aggs import OperationOfTwoValues
    from mescal.kpis.kpi_collection import KPICollection


DataSetType = TypeVar('DataSetType', bound='DataSet')
DataSetConfigType = TypeVar('DataSetConfigType', bound='DataSetConfig')
KPIType = TypeVar('KPIType', bound='KPI')
ValueOperationType = TypeVar('ValueOperationType', bound='_TwoValueOperation')
KPICollectionType = TypeVar('KPICollectionType', bound='KPICollection')
