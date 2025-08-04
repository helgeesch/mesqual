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
    from mescal.visualizations.folium_viz_system.base_viz_system import ResolvedFeature, FeatureResolver
    from mescal.visualizations.value_mapping_system.base import BaseMapping
    from mescal.visualizations.value_mapping_system.discrete import DiscreteInputMapping
    from mescal.visualizations.value_mapping_system.continuous import SegmentedContinuousInputMappingBase
    from mescal.visualizations.folium_legend_system.base import BaseLegend

FlagType = TypeVar('FlagType', bound='FlagTypeProtocol')
DatasetType = TypeVar('DatasetType', bound='Dataset')
DatasetConfigType = TypeVar('DatasetConfigType', bound='DatasetConfig')
FlagIndexType = TypeVar('FlagIndexType', bound='FlagIndex')
KPIType = TypeVar('KPIType', bound='KPI')
ValueOperationType = TypeVar('ValueOperationType', bound='OperationOfTwoValues')
FeatureResolverType = TypeVar('FeatureResolverType', bound='FeatureResolver')
ResolvedFeatureType = TypeVar('ResolvedFeatureType', bound='ResolvedFeature')
ValueMappingType = TypeVar('ValueMappingType', bound='BaseMapping')
DiscreteMappingType = TypeVar('DiscreteMappingType', bound='DiscreteInputMapping')
ContinuousMappingType = TypeVar('ContinuousMappingType', bound='SegmentedContinuousInputMappingBase')
FoliumLegendType = TypeVar('FoliumLegendType', bound='BaseLegend')
