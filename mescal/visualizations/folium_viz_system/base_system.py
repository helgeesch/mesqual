from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union, Callable, Any, Generic, List, Type, Dict

import folium
import pandas as pd

from mescal.kpis import KPICollection, KPI
from mescal.typevars import ResolvedStyleType, StyleResolverType
from mescal.visualizations.folium_viz_system.simple_generators import TooltipGenerator, PopupGenerator
from mescal.visualizations.folium_viz_system.map_data_item import MapDataItem, ModelDataItem, KPIDataItem


@dataclass
class StyleMapper:
    """Maps a data column to a visual property using a mapping function."""
    property: str  # e.g., 'color', 'width', 'opacity', 'height', 'arrow_speed', 'shadow_color'
    column: str | None = None  # column to get value from data source, None for static values
    mapping: Union[Callable, Any] = None  # mapper function or static value
    return_type: type = object  # expected return type for validation


@dataclass
class ResolvedStyle:
    """Container for resolved style properties."""
    properties: dict = field(default_factory=dict)

    def get(self, property: str, default=None):
        return self.properties.get(property, default)

    def __getitem__(self, key):
        return self.properties[key]

    def __setitem__(self, key, value):
        self.properties[key] = value

    def __contains__(self, key):
        return key in self.properties


class StyleResolver(Generic[ResolvedStyleType]):
    """Resolves styling for map data items using flexible property mappings."""

    def __init__(self, style_mappers: List[StyleMapper] = None, style_type: Type[ResolvedStyleType] = ResolvedStyle):
        self.style_mappers = {mapper.property: mapper for mapper in style_mappers}
        self.style_type = style_type

    def resolve_style(self, data_item: MapDataItem) -> ResolvedStyleType:
        """Resolve styling for a data item."""
        resolved = self.style_type()

        for prop, mapper in self.style_mappers.items():
            if mapper.column:
                value = data_item.get_styling_value(mapper.column)
            else:
                value = None

            if callable(mapper.mapping):
                resolved[prop] = mapper.mapping(value)
            else:
                resolved[prop] = mapper.mapping

        return resolved

    @classmethod
    def _validate_mapper_namings(cls, mappers: Dict[str, StyleMapper]) -> None:
        for key, mapper in mappers.items():
            if mapper.property != key:
                raise ValueError(
                    f"StyleMapper property not set correctly; StyleMapper for {key} must have property set to '{key}'."
                )

    @classmethod
    def _transform_static_values_to_style_mappers(cls, mappers: Dict[str, Any]) -> Dict[str, StyleMapper]:
        for key, mapper in list(mappers.items()):
            if not isinstance(mapper, StyleMapper):
                mappers[key] = StyleMapper(key, None, mapper, type(mapper) if mapper is not None else object)
        return mappers


class FoliumObjectGenerator(Generic[StyleResolverType], ABC):
    """Abstract base for generating folium objects."""

    def __init__(
            self,
            style_resolver: StyleResolverType = None,
            tooltip_generator: TooltipGenerator = None,
            popup_generator: PopupGenerator = None,
    ):
        self.style_resolver: StyleResolverType = style_resolver or self._style_resolver_type()()
        self.tooltip_generator = tooltip_generator or TooltipGenerator()
        self.popup_generator = popup_generator

    @abstractmethod
    def _style_resolver_type(self) -> Type[StyleResolverType]:
        return StyleResolver

    @abstractmethod
    def generate(self, data_item: MapDataItem, feature_group: folium.FeatureGroup) -> None:
        """Generate folium object and add it to the feature group."""
        pass

    def generate_objects_for_model_df(
            self,
            model_df: pd.DataFrame,
            feature_group: folium.FeatureGroup,
            **kwargs
    ) -> folium.FeatureGroup:
        """Add model DataFrame data to the map."""
        for _, row in model_df.iterrows():
            data_item = ModelDataItem(row, **kwargs)
            self.generate(data_item, feature_group)
        return feature_group

    def generate_objects_for_kpi_collection(
            self,
            kpi_collection: 'KPICollection',
            feature_group: folium.FeatureGroup,
            **kwargs
    ) -> folium.FeatureGroup:
        """Add KPI data to the map."""
        for kpi in kpi_collection:
            data_item = KPIDataItem(kpi, kpi_collection, **kwargs)
            self.generate(data_item, feature_group)
        return feature_group

    def generate_object_for_single_kpi(
            self,
            kpi: 'KPI',
            feature_group: folium.FeatureGroup,
            kpi_collection: 'KPICollection' = None,
            **kwargs
    ) -> folium.FeatureGroup:
        """Add a single KPI to the map with optional context."""
        data_item = KPIDataItem(kpi, kpi_collection, **kwargs)
        self.generate(data_item, feature_group)
        return feature_group
