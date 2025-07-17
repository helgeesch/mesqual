from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union, Callable, Any, Generic, List, Type, Dict

import folium
import pandas as pd

from mescal.kpis import KPICollection, KPI
from mescal.typevars import ResolvedStyleType, StyleResolverType
from mescal.visualizations.folium_viz_system.element_generators import TooltipGenerator, PopupGenerator
from mescal.visualizations.folium_viz_system.map_data_item import MapDataItem, ModelDataItem, KPIDataItem


class StyleMapper(ABC):
    def __init__(self, return_type: type = object):
        self.return_type = return_type

    @abstractmethod
    def resolve(self, data_item: MapDataItem) -> Any:
        pass

    @classmethod
    def for_attribute(
            cls,
            attribute: str,
            mapping: Callable[[Any], Any] = None,
            return_type: type = object
    ) -> 'AttributeStyleMapper':
        return AttributeStyleMapper(attribute, mapping, return_type)

    @classmethod
    def for_data_item(
            cls,
            mapping: Callable[[MapDataItem], Any],
            return_type: type = object
    ) -> 'DataItemStyleMapper':
        return DataItemStyleMapper(mapping, return_type)

    @classmethod
    def for_static(cls, value: Any) -> 'StaticStyleMapper':
        return StaticStyleMapper(value, type(value))


class StaticStyleMapper(StyleMapper):
    def __init__(self, value: Any, return_type: type = object):
        super().__init__(return_type)
        self.value = value

    def resolve(self, data_item: MapDataItem) -> Any:
        return self.value


class AttributeStyleMapper(StyleMapper):
    def __init__(self, attribute: str, mapping: Callable[[Any], Any] = None, return_type: type = object):
        super().__init__(return_type)
        self.attribute = attribute
        self.mapping = mapping or (lambda x: x)

    def resolve(self, data_item: MapDataItem) -> Any:
        value = data_item.get_styling_value(self.attribute)
        return self.mapping(value)


class DataItemStyleMapper(StyleMapper):
    def __init__(self, mapping: Callable[[MapDataItem], Any], return_type: type = object):
        super().__init__(return_type)
        self.mapping = mapping

    def resolve(self, data_item: MapDataItem) -> Any:
        return self.mapping(data_item)


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

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.properties)
        for name in dir(self.__class__):
            attr = getattr(self.__class__, name)
            if isinstance(attr, property):
                out[name] = getattr(self, name)
        return out


class StyleResolver(Generic[ResolvedStyleType]):
    def __init__(self, style_type: Type[ResolvedStyleType] = None, **style_mappers: StyleMapper | Any):
        self.style_type: Type[ResolvedStyleType] = style_type or ResolvedStyleType.__constraints__[0]
        self.style_mappers: dict[str, StyleMapper] = self._normalize_style_mappers(style_mappers)

    def resolve_style(self, data_item: MapDataItem) -> ResolvedStyleType:
        resolved = self.style_type()
        for prop, mapper in self.style_mappers.items():
            resolved[prop] = mapper.resolve(data_item)
        return resolved

    @staticmethod
    def _normalize_style_mappers(mappers: dict[str, StyleMapper | Any]) -> dict[str, StyleMapper]:
        return {
            key: mapper if isinstance(mapper, StyleMapper) else StyleMapper.for_static(mapper)
            for key, mapper in mappers.items()
        }


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
