"""Type variables of the generic classes of mesqual."""

from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING


if TYPE_CHECKING:
    from mesqual.flag.flag import FlagTypeProtocol
    from mesqual.flag.flag_index import FlagIndex
    from mesqual.datasets.dataset import Dataset
    from mesqual.datasets.dataset_config import DatasetConfig
    from mesqual.kpis.definitions.base import KPIDefinition
    from mesqual.visualizations.folium_viz_system.base_viz_system import FeatureResolver, ResolvedFeature
    from mesqual.visualizations.value_mapping_system.base import BaseMapping
    from mesqual.visualizations.value_mapping_system.discrete import DiscreteInputMapping
    from mesqual.visualizations.value_mapping_system.continuous import SegmentedContinuousInputMappingBase
    from mesqual.visualizations.folium_viz_system.legends.base import BaseLegend

FlagType = TypeVar('FlagType', bound='FlagTypeProtocol')
"""Type of the flags that identify data in a dataset, bound to `FlagTypeProtocol`. Typically `str`.

A type parameter of `Dataset`, `PlatformDataset`, the dataset collections and `DatasetComparison`.
"""

DatasetType = TypeVar('DatasetType', bound='Dataset')
"""Type of a dataset, a subclass of `Dataset`. On `PlatformDataset`: the base class of its interpreters.

A type parameter of `PlatformDataset`, the dataset collections and `DatasetComparison`.
"""

DatasetConfigType = TypeVar('DatasetConfigType', bound='DatasetConfig')
"""Type of the configuration of a dataset, a subclass of `DatasetConfig`.

A type parameter of `Dataset`, `PlatformDataset`, the dataset collections and `DatasetComparison`.
"""

FlagIndexType = TypeVar('FlagIndexType', bound='FlagIndex')
"""Type of the flag index of a dataset, a subclass of `FlagIndex`.

A type parameter of `Dataset`, `PlatformDataset`, the dataset collections and `DatasetComparison`.
"""

KPIDefinitionType = TypeVar('KPIDefinitionType', bound='KPIDefinition')
"""Type of a KPI definition, a subclass of `KPIDefinition`. The type parameter of `KPIBuilder`."""

FeatureResolverType = TypeVar('FeatureResolverType', bound='FeatureResolver')
"""Type of the feature resolver that a map object generator uses, a subclass of `FeatureResolver`.

The type parameter of `FoliumObjectGenerator`.
"""

ResolvedFeatureType = TypeVar('ResolvedFeatureType', bound='ResolvedFeature')
"""Type of the resolved feature that a feature resolver produces, for example `ResolvedAreaFeature`.

Subclass of `ResolvedFeature`; type parameter of `FeatureResolver`.
"""

ValueMappingType = TypeVar('ValueMappingType', bound='BaseMapping')
"""Type of a value mapping, a subclass of `BaseMapping`. The type parameter of `BaseLegend`."""

DiscreteMappingType = TypeVar('DiscreteMappingType', bound='DiscreteInputMapping')
"""Type of a discrete input mapping, a subclass of `DiscreteInputMapping`.

The type parameter of `DiscreteLegendBase`.
"""

ContinuousMappingType = TypeVar('ContinuousMappingType', bound='SegmentedContinuousInputMappingBase')
"""Type of a segmented continuous input mapping, a subclass of `SegmentedContinuousInputMappingBase`.

The type parameter of `ContinuousLegendBase`.
"""

FoliumLegendType = TypeVar('FoliumLegendType', bound='BaseLegend')
"""Type of a folium legend, a subclass of `BaseLegend`. For user code that is generic over the legend class."""
