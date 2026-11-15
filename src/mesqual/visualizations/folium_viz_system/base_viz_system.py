from abc import ABC, abstractmethod
from html import escape
from dataclasses import dataclass, field
from typing import Union, Callable, Any, Generic, Type, TYPE_CHECKING

from shapely import Point, Polygon, MultiPolygon, LineString, MultiLineString
import pandas as pd
import folium

from mesqual.typevars import ResolvedFeatureType, FeatureResolverType
from mesqual.utils.pandas_utils import add_index_as_column
from mesqual.visualizations.folium_viz_system.visualizable_data_item import (
    VisualizableDataItem,
    ModelDataItem,
    KPIDataItem,
)

if TYPE_CHECKING:
    from mesqual.kpis import KPI, KPICollection
    from mesqual.units import QuantityToTextConverter


def _escape_text(value: Any) -> str:
    """Return `value` as text that is safe in HTML and in a JS template literal."""
    return escape(str(value)).replace('`', '&#96;').replace('$', '&#36;').replace('\\', '&#92;')


class PropertyMapper:
    """Maps a data item to the value of one visual property.

    A `PropertyMapper` wraps a function that takes a `VisualizableDataItem` and
    returns a value, such as a color, a width or a geometry. Feature resolvers
    call it for each map element. The class methods build mappers from a static
    value, from an attribute of the model object or from the KPI value.

    Args:
        mapping: Function that takes a data item and returns the property value.

    Examples:
        >>> from mesqual.kpis import Aggregations, FlagAggKPIBuilder
        >>> from mesqual.visualizations.folium_viz_system import KPIDataItem
        >>> dataset.add_kpis_from_definitions(
        ...     FlagAggKPIBuilder().for_flag('Country.Results.price').with_aggregation(Aggregations.Mean).build()
        ... )
        >>> price = KPIDataItem(dataset.kpi_collection.filter(object_name='C1')[0])
        >>> PropertyMapper.from_static_value('#000000').map_data_item(price)
        '#000000'

        A mapper with its own function:

        >>> label = PropertyMapper(lambda item: f"{item.get_object_attribute('name')}: {item.get_text_representation()}")
        >>> label.map_data_item(price)
        'Country 1: 42.1 €/MWh'
    """

    def __init__(self, mapping: Callable[[Union[VisualizableDataItem, KPIDataItem, ModelDataItem]], Any]):
        self.mapping = mapping

    def map_data_item(self, data_item: VisualizableDataItem) -> Any:
        """
        Map data item to property value.

        Args:
            data_item: Data item to map

        Returns:
            Mapped property value
        """
        return self.mapping(data_item)

    @classmethod
    def from_static_value(cls, value: Any) -> 'PropertyMapper':
        """
        Create mapper that returns the same value for all data items.

        Used for consistent styling across all map elements (e.g., all borders
        the same color, all markers the same size).

        Args:
            value: Static value to return for all data items

        Returns:
            PropertyMapper that always returns the static value

        """
        return cls(lambda data_item: value)

    @classmethod
    def from_item_attr(
        cls,
        attribute: str,
        mapping: Callable[[Any], Any] = None,
    ) -> 'PropertyMapper':
        """
        Create mapper from model/object attribute with optional transformation.

        Extracts values from model data attributes (geometry, capacity, name, etc.)
        and optionally applies a transformation function. The attribute is resolved
        from the underlying model DataFrame or object data.

        Args:
            attribute: Name of the attribute to extract (e.g., 'geometry', 'capacity'), must be an entry in the object pd.Series
            mapping: Optional transformation function to apply to the attribute value

        Returns:
            PropertyMapper that extracts and optionally transforms the attribute

        Examples:
            >>> from mesqual.visualizations.folium_viz_system import ModelDataItem
            >>> generator = ModelDataItem(dataset.fetch('Generator.Model').loc['G3'], object_type='Generator')
            >>> PropertyMapper.from_item_attr('fuel').map_data_item(generator)
            'gas'
            >>> print(PropertyMapper.from_item_attr('capacity', lambda mw: mw / 100).map_data_item(generator))
            10.0
        """
        if mapping is None:
            mapping = lambda x: x
        return cls(lambda data_item: mapping(data_item.get_object_attribute(attribute)))

    @classmethod
    def from_kpi_value(cls, mapping: Callable[[Any], Any], use_abs_kpi_value: bool = False) -> 'PropertyMapper':
        """
        Create mapper from KPI values with transformation function.

        Specifically designed for KPIDataItem objects, extracts the computed KPI value
        and applies a transformation. Used for styling based on energy system metrics
        like power flows, prices, or emissions.

        Args:
            mapping: Transformation function applied to the KPI value

        Returns:
            PropertyMapper that transforms KPI values

        Examples:
            >>> from mesqual.kpis import Aggregations, FlagAggKPIBuilder
            >>> from mesqual.visualizations.folium_viz_system import KPIDataItem
            >>> dataset.add_kpis_from_definitions(
            ...     FlagAggKPIBuilder().for_flag('CountryBorder.Results.flow_sent_net').with_aggregation(Aggregations.Mean).build()
            ... )
            >>> flow = KPIDataItem(dataset.kpi_collection.filter(object_name='C1 - C2')[0])
            >>> print(PropertyMapper.from_kpi_value(lambda v: v < 0).map_data_item(flow))
            True

            A colorscale on the absolute flow:

            >>> from mesqual.visualizations.value_mapping_system import SegmentedContinuousColorscale
            >>> colors = SegmentedContinuousColorscale(segments={(0, 1000): 'YlOrRd'})
            >>> PropertyMapper.from_kpi_value(colors, use_abs_kpi_value=True).map_data_item(flow)
            '#fd923e'
        """
        if use_abs_kpi_value:
            return cls(lambda data_item: mapping(abs(data_item.kpi.value)))
        else:
            return cls(lambda data_item: mapping(data_item.kpi.value))


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
    """
    Resolves the visual properties of a data item.

    The resolver calls one `PropertyMapper` per property and returns the values
    as a resolved feature. A property value that is not a `PropertyMapper` is a
    static value for all items. The subclasses for areas, lines and markers set
    the default mappers of their properties.

    Args:
        feature_type: Class of the resolved feature.
        **property_mappers: `PropertyMapper` or static value per property.
            `tooltip` and `popup` take these values:

            - True: the default, a table of `get_tooltip_data` of the item,
              with keys and values escaped as text.
            - False: no tooltip or popup.
            - A `PropertyMapper` or a static value: the tooltip is HTML and the
              popup is a `folium.Popup` or HTML, used as given.

    Examples:
        >>> from mesqual.kpis import Aggregations, FlagAggKPIBuilder
        >>> from mesqual.visualizations.folium_viz_system import AreaFeatureResolver, KPIDataItem
        >>> dataset.add_kpis_from_definitions(
        ...     FlagAggKPIBuilder().for_flag('Country.Results.price').with_aggregation(Aggregations.Mean).build()
        ... )
        >>> price = KPIDataItem(dataset.kpi_collection.filter(object_name='C1')[0])
        >>> resolver = AreaFeatureResolver(
        ...     fill_color=PropertyMapper.from_kpi_value(lambda v: 'red' if v > 45 else 'blue'),
        ...     fill_opacity=0.8,
        ... )
        >>> feature = resolver.resolve_feature(price)
        >>> feature.fill_color, feature.fill_opacity, feature.border_color
        ('blue', 0.8, 'white')
        >>> feature.geometry
        <POLYGON ((9 46, 9 50, 5 50, 5 46, 9 46))>
    """

    def __init__(self, feature_type: Type[ResolvedFeatureType] = None, **property_mappers: PropertyMapper | Any):
        self.feature_type: Type[ResolvedFeatureType] = feature_type or ResolvedFeatureType.__constraints__[0]

        _defaults_if_true = dict(
            tooltip=self._default_tooltip_generator,
            popup=self._default_popup_generator,
            text_print_content=self._default_text_print_generator,
        )
        for k, mapper_factory in _defaults_if_true.items():
            if property_mappers.get(k, None) is True:
                property_mappers[k] = mapper_factory()
            elif property_mappers.get(k, None) is False:
                property_mappers[k] = None

        self.property_mappers: dict[str, PropertyMapper] = self._normalize_property_mappers(property_mappers)

    def resolve_feature(self, data_item: VisualizableDataItem) -> ResolvedFeatureType:
        """
        Resolve feature properties from data item.

        Args:
            data_item: Data item to resolve

        Returns:
            Resolved feature with all computed properties
        """
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
    def _data_table_html(data: dict) -> str:
        """Return an HTML table with one row per entry of `data`, keys and values escaped."""
        rows = ''.join(
            f'  <tr><td style="padding: 4px 8px;"><strong>{_escape_text(key)}</strong></td>'
            f'<td style="text-align: right; padding: 4px 8px;">{_escape_text(value)}</td></tr>\n'
            for key, value in data.items()
        )
        return f'<table style="border-collapse: collapse;">\n{rows}</table>'

    @staticmethod
    def _default_tooltip_generator() -> PropertyMapper:
        """
        Create default tooltip generator showing data item information.

        Returns:
            PropertyMapper that generates an HTML table of the item data, keys and values escaped
        """
        return _DefaultTableMapper()

    @staticmethod
    def _default_popup_generator() -> PropertyMapper:
        """
        Create default popup generator with formatted data item information.

        Returns:
            PropertyMapper that generates a folium.Popup with the table of the default tooltip
        """
        return _DefaultPopupMapper()

    @staticmethod
    def _default_text_print_generator() -> PropertyMapper:
        """
        Create default text content generator for overlay labels.

        Returns:
            PropertyMapper that returns data item text representation
        """

        def get_text(data_item: VisualizableDataItem) -> str:
            return data_item.get_text_representation()

        return PropertyMapper(get_text)

    @staticmethod
    def _default_geometry_mapper() -> PropertyMapper:
        """
        Create default geometry mapper that extracts geometric objects.

        Returns:
            PropertyMapper that extracts 'geometry' attribute from data items
        """

        def get_geometry(data_item: VisualizableDataItem) -> Polygon | None:
            if data_item.object_has_attribute('geometry'):
                return data_item.get_object_attribute('geometry')
            return None

        return PropertyMapper(get_geometry)

    @staticmethod
    def _default_location_mapper() -> PropertyMapper:
        """
        Create the default location mapper.

        The location is the first `Point` of the item attributes `location`,
        `projection_point`, `centroid` and `midpoint`; else a point from the
        attributes `lat` and `lon` or `latitude` and `longitude` if both values
        are instances of Python `int` or `float` (`numpy.float64` is one,
        `numpy.int64` is not); else from the attribute `geometry`: the point,
        the representative point of a polygon, or the midpoint of a line.

        Returns:
            PropertyMapper that returns the Point location of an item
        """

        def get_location(data_item: VisualizableDataItem) -> Point | None:
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

        return PropertyMapper(get_location)

    @staticmethod
    def _default_line_string_mapper() -> PropertyMapper:
        """
        Create default LineString geometry mapper for line visualizations.

        Returns:
            PropertyMapper that extracts LineString geometries from data items
        """

        def get_line_string(data_item: VisualizableDataItem) -> LineString | None:
            for k in ['geometry', 'line_string']:
                if data_item.object_has_attribute(k):
                    line_string = data_item.get_object_attribute(k)
                    if isinstance(line_string, (LineString, MultiLineString)):
                        return line_string
            return None

        return PropertyMapper(get_line_string)


class _DefaultTableMapper(PropertyMapper):
    """Default table: the HTML table of `get_tooltip_data` of the item, values formatted with `converter`.

    The default tooltip.
    """

    def __init__(self, converter: 'QuantityToTextConverter | None' = None):
        self.converter = converter
        super().__init__(self._get_table)

    def _get_table(self, data_item: VisualizableDataItem) -> str:
        if self.converter is None:
            return FeatureResolver._data_table_html(data_item.get_tooltip_data())
        return FeatureResolver._data_table_html(data_item.get_tooltip_data(converter=self.converter))


class _DefaultPopupMapper(_DefaultTableMapper):
    """Default popup: the default table in a `folium.Popup`."""

    def map_data_item(self, data_item: VisualizableDataItem) -> folium.Popup:
        return folium.Popup(super().map_data_item(data_item), max_width=300)


class FoliumObjectGenerator(Generic[FeatureResolverType], ABC):
    """
    Abstract base class for generating folium map objects from data items.

    A generator adds one folium object per data item to a feature group, for
    example an area, a line or a marker. Its feature resolver gives the visual
    properties. The generators take the rows of a model DataFrame or the KPIs of
    a KPI collection.

    Examples:
        Areas from the country model, lines from the border model, in one feature
        group:

        >>> import folium
        >>> from mesqual.visualizations.folium_viz_system import AreaGenerator, LineFeatureResolver, LineGenerator
        >>> fg = folium.FeatureGroup('Model')
        >>> fg = AreaGenerator().generate_objects_for_model_df(dataset.fetch('Country.Model'), fg)
        >>> lines = LineGenerator(LineFeatureResolver(geometry=PropertyMapper.from_item_attr('geo_line_string')))
        >>> fg = lines.generate_objects_for_model_df(dataset.fetch('CountryBorder.Model'), fg)
        >>> m = folium.Map(location=[48, 9], zoom_start=6)
        >>> fg = fg.add_to(m)
    """

    def __init__(
        self,
        feature_resolver: FeatureResolverType = None,
    ):
        self.feature_resolver: FeatureResolverType = feature_resolver or self._feature_resolver_type()()

    @abstractmethod
    def _feature_resolver_type(self) -> Type[FeatureResolverType]:
        return FeatureResolver

    @abstractmethod
    def generate(self, data_item: VisualizableDataItem, feature_group: folium.FeatureGroup) -> None:
        """
        Generate folium object and add it to the feature group.

        Args:
            data_item: Data item to visualize
            feature_group: Feature group to add to
        """
        pass

    def generate_objects_for_model_df(
        self, model_df: pd.DataFrame, feature_group: folium.FeatureGroup, **kwargs
    ) -> folium.FeatureGroup:
        """Add model DataFrame data to the map."""
        model_dff = add_index_as_column(model_df)
        object_type = model_dff.index.name if isinstance(model_dff.index.name, str) else None
        for _, row in model_dff.iterrows():
            data_item = ModelDataItem(row, object_type=object_type, **kwargs)
            self.generate(data_item, feature_group)
        return feature_group

    def generate_objects_for_kpi_collection(
        self, kpi_collection: 'KPICollection', feature_group: folium.FeatureGroup, **kwargs
    ) -> folium.FeatureGroup:
        """Add KPI data to the map."""
        for kpi in kpi_collection:
            data_item = KPIDataItem(kpi, kpi_collection, **kwargs)
            self.generate(data_item, feature_group)
        return feature_group

    def generate_object_for_single_kpi(
        self, kpi: 'KPI', feature_group: folium.FeatureGroup, kpi_collection: 'KPICollection' = None, **kwargs
    ) -> folium.FeatureGroup:
        """Add a single KPI to the map with optional context."""
        data_item = KPIDataItem(kpi, kpi_collection, **kwargs)
        self.generate(data_item, feature_group)
        return feature_group
