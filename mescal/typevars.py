from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING


if TYPE_CHECKING:
    from mescal.flag.flag import FlagTypeProtocol
    from mescal.datasets.dataset import Dataset
    from mescal.datasets.dataset_config import DatasetConfig
    from mescal.flag.flag_index import FlagIndex
    from mescal.kpis.kpi_base import KPI
    from mescal.kpis.aggs import OperationOfTwoValues
    from mescal.kpis.kpi_collection import KPICollection
    from mescal.visualizations.folium_viz_system.system_base import ResolvedStyle, StyleResolver

FlagType = TypeVar('FlagType', bound='FlagTypeProtocol')
DatasetType = TypeVar('DatasetType', bound='Dataset')
DatasetConfigType = TypeVar('DatasetConfigType', bound='DatasetConfig')
FlagIndexType = TypeVar('FlagIndexType', bound='FlagIndex')
KPIType = TypeVar('KPIType', bound='KPI')
ValueOperationType = TypeVar('ValueOperationType', bound='OperationOfTwoValues')
StyleResolverType = TypeVar('StyleResolverType', bound='StyleResolver')
ResolvedStyleType = TypeVar('ResolvedStyleType', bound='ResolvedStyle')
