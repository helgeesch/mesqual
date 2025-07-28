from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union, Callable, Any, Generic, List, Type, Dict

from shapely import Point, Polygon, MultiPolygon, LineString, MultiLineString
import pandas as pd
import folium

from mescal.kpis import KPICollection, KPI
from mescal.typevars import ResolvedFeatureType, FeatureResolverType
from mescal.visualizations.folium_viz_system.map_data_item import MapDataItem, ModelDataItem, KPIDataItem


class PropertyMapper(ABC):
    @abstractmethod
    def map_data_item(self, data_item: MapDataItem) -> Any:
        pass

    @classmethod
    def from_item(
            cls,
            mapping: Callable[[MapDataItem], Any],
    ) -> 'ItemToPropertyMapper':
        return ItemToPropertyMapper(mapping)

    @classmethod
    def from_item_attr(
            cls,
            attribute: str,
            mapping: Callable[[Any], Any] = None,
    ) -> 'ItemAttributeToPropertyMapper':
        return ItemAttributeToPropertyMapper(attribute, mapping)

    @classmethod
    def from_static_value(cls, value: Any) -> 'StaticPropertyMapper':
        return StaticPropertyMapper(value)


class StaticPropertyMapper(PropertyMapper):
    def __init__(self, value: Any):
        self.value = value

    def map_data_item(self, data_item: MapDataItem) -> Any:
        return self.value


class ItemAttributeToPropertyMapper(PropertyMapper):
    def __init__(self, attribute: str, mapping: Callable[[Any], Any] = None):
        self.attribute = attribute
        self.mapping = mapping or (lambda x: x)

    def map_data_item(self, data_item: MapDataItem) -> Any:
        value = data_item.get_object_attribute(self.attribute)
        return self.mapping(value)


class ItemToPropertyMapper(PropertyMapper):
    def __init__(self, mapping: Callable[[MapDataItem], Any]):
        self.mapping = mapping

    def map_data_item(self, data_item: MapDataItem) -> Any:
        return self.mapping(data_item)


@dataclass
class ResolvedFeature:
    """Container for resolved feature properties."""
    properties: dict = field(default_factory=dict)
    tooltip: str = None
    popup: folium.Popup = None
    text_print_content: str = None

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


class FeatureResolver(Generic[ResolvedFeatureType]):
    def __init__(self, feature_type: Type[ResolvedFeatureType] = None, **property_mappers: PropertyMapper | Any):
        self.feature_type: Type[ResolvedFeatureType] = feature_type or ResolvedFeatureType.__constraints__[0]

        _defaults_if_true = dict(
            tooltip=self._default_tooltip_generator,
            popup=self._default_popup_generator,
            text_print_content=self._default_text_print_generator,
        )
        for k, mapper in _defaults_if_true.items():
            if property_mappers.get(k, None) is True:
                property_mappers[k] = mapper()
            elif property_mappers.get(k, None) is False:
                property_mappers[k] = None

        self.property_mappers: dict[str, PropertyMapper] = self._normalize_property_mappers(property_mappers)

    def resolve_feature(self, data_item: MapDataItem) -> ResolvedFeatureType:
        resolved = self.feature_type()
        for prop, mapper in self.property_mappers.items():
            resolved[prop] = mapper.map_data_item(data_item)
            if prop in ['tooltip', 'popup', 'text_print_content']:
                setattr(resolved, prop, mapper.map_data_item(data_item))
        return resolved

    @staticmethod
    def _normalize_property_mappers(mappers: dict[str, PropertyMapper | Any]) -> dict[str, PropertyMapper]:
        return {
            key: mapper if isinstance(mapper, PropertyMapper) else PropertyMapper.from_static_value(mapper)
            for key, mapper in mappers.items()
        }

    @staticmethod
    def _explicit_or_fallback(explicit: Any, fallback: PropertyMapper = None) -> PropertyMapper:
        if explicit is not None:
            return explicit
        return fallback

    @staticmethod
    def _default_tooltip_generator() -> ItemToPropertyMapper:

        def get_tooltip(data_item: MapDataItem) -> str:
            tooltip_data = data_item.get_tooltip_data()

            html = '<table style="border-collapse: collapse;">\n'
            for key, value in tooltip_data.items():
                html += f'  <tr><td style="padding: 4px 8px;"><strong>{key}</strong></td>' \
                        f'<td style="text-align: right; padding: 4px 8px;">{value}</td></tr>\n'
            html += '</table>'

            return html

        return ItemToPropertyMapper(get_tooltip)

    @staticmethod
    def _default_popup_generator() -> ItemToPropertyMapper:

        def get_popup(data_item: MapDataItem) -> folium.Popup:
            tooltip_data = data_item.get_tooltip_data()

            html = '<table style="border-collapse: collapse;">\n'
            for key, value in tooltip_data.items():
                html += f'  <tr><td style="padding: 4px 8px;"><strong>{key}</strong></td>' \
                        f'<td style="text-align: right; padding: 4px 8px;">{value}</td></tr>\n'
            html += '</table>'

            return folium.Popup(html, max_width=300)

        return ItemToPropertyMapper(get_popup)

    @staticmethod
    def _default_text_print_generator() -> ItemToPropertyMapper:
        return ItemToPropertyMapper(lambda d: d.get_text_representation())

    @staticmethod
    def _default_geometry_mapper() -> ItemToPropertyMapper:

        def get_geometry(data_item: MapDataItem) -> Polygon | None:
            if data_item.object_has_attribute('geometry'):
                return data_item.get_object_attribute('geometry')
            return None

        return ItemToPropertyMapper(get_geometry)

    @staticmethod
    def _default_location_mapper() -> ItemToPropertyMapper:

        def get_location(data_item: MapDataItem) -> Point | None:
            for k in ['location', 'projection_point', 'centroid', 'midpoint']:
                if data_item.object_has_attribute(k):
                    location = data_item.get_object_attribute(k)
                    if isinstance(location, Point):
                        return location

            for lat, lon in [('lat', 'lon'), ('latitude', 'longitude')]:
                if data_item.object_has_attribute(lat) and data_item.object_has_attribute(lon):
                    lat_value = data_item.get_object_attribute(lat)
                    lon_value = data_item.get_object_attribute(lon)
                    if all(isinstance(v, (int, float)) for v in [lat_value, lon_value]):
                        return Point([lon_value, lat_value])

            if data_item.object_has_attribute('geometry'):
                geometry = data_item.get_object_attribute('geometry')
                if isinstance(geometry, Point):
                    return geometry
                elif isinstance(geometry, (Polygon, MultiPolygon)):
                    return geometry.representative_point()
                elif isinstance(geometry, (LineString, MultiLineString)):
                    return geometry.interpolate(0.5, normalized=True)
            return None

        return ItemToPropertyMapper(get_location)

    @staticmethod
    def _default_line_string_mapper() -> ItemToPropertyMapper:

        def get_line_string(data_item: MapDataItem) -> LineString | None:
            for k in ['geometry', 'line_string']:
                if data_item.object_has_attribute(k):
                    line_string = data_item.get_object_attribute(k)
                    if isinstance(line_string, (LineString, MultiLineString)):
                        return line_string
            return None

        return ItemToPropertyMapper(get_line_string)


class FoliumObjectGenerator(Generic[FeatureResolverType], ABC):
    """Abstract base for generating folium objects."""

    def __init__(
            self,
            feature_resolver: FeatureResolverType = None,
    ):
        self.feature_resolver: FeatureResolverType = feature_resolver or self._feature_resolver_type()()

    @abstractmethod
    def _feature_resolver_type(self) -> Type[FeatureResolverType]:
        return FeatureResolver

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
