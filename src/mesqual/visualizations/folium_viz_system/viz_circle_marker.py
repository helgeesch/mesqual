from dataclasses import dataclass
from typing import Type, Any

from shapely import Point
import folium

from mesqual.visualizations.folium_viz_system.visualizable_data_item import VisualizableDataItem
from mesqual.visualizations.folium_viz_system.base_viz_system import (
    ResolvedFeature,
    FeatureResolver,
    PropertyMapper,
    FoliumObjectGenerator,
)


@dataclass
class ResolvedCircleMarkerFeature(ResolvedFeature):
    """Specialized style container for circle marker visualizations."""

    @property
    def location(self) -> Point:
        return self.get('location')

    @property
    def fill_color(self) -> str:
        return self.get('fill_color')

    @property
    def border_color(self) -> str:
        return self.get('border_color')

    @property
    def radius(self) -> float:
        return self.get('radius')

    @property
    def border_width(self) -> float:
        return self.get('border_width')

    @property
    def fill_opacity(self) -> float:
        return self.get('fill_opacity')


class CircleMarkerFeatureResolver(FeatureResolver[ResolvedCircleMarkerFeature]):
    """Resolves the visual properties of circle markers at point locations.

    Args:
        fill_color: Fill color.
        border_color: Border color.
        radius: Radius in pixels.
        border_width: Border width in pixels.
        fill_opacity: Fill opacity from 0 to 1.
        tooltip: Tooltip content, see `FeatureResolver`.
        popup: Popup content, see `FeatureResolver`.
        location: Point of the marker. Default: the first `Point` of the item
            attributes `location`, `projection_point`, `centroid` and `midpoint`;
            else a point from the attributes `lat` and `lon` or `latitude` and
            `longitude` if both values are instances of Python `int` or `float`
            (`numpy.float64` is one, `numpy.int64` is not); else from the
            attribute `geometry`: the point, the
            representative point of a polygon, or the midpoint of a line. An
            item without location gets no marker.
        **property_mappers: Additional custom property mappings.

    Each argument takes a static value or a `PropertyMapper`.
    """

    def __init__(
        self,
        fill_color: PropertyMapper | str = '#D9D9D9',
        border_color: PropertyMapper | str = 'white',
        radius: PropertyMapper | float = 8.0,
        border_width: PropertyMapper | float = 1.0,
        fill_opacity: PropertyMapper | float = 1.0,
        tooltip: PropertyMapper | str | bool = True,
        popup: PropertyMapper | folium.Popup | bool = False,
        location: PropertyMapper | Point = None,
        **property_mappers: PropertyMapper | Any,
    ):
        mappers = dict(
            fill_color=fill_color,
            border_color=border_color,
            radius=radius,
            border_width=border_width,
            fill_opacity=fill_opacity,
            tooltip=tooltip,
            popup=popup,
            location=self._explicit_or_fallback(location, self._default_location_mapper()),
            **property_mappers,
        )
        super().__init__(feature_type=ResolvedCircleMarkerFeature, **mappers)


class CircleMarkerGenerator(FoliumObjectGenerator[CircleMarkerFeatureResolver]):
    """Generates folium CircleMarker objects for point geometries.

    Examples:
        One marker per node, with the radius from the voltage:

        >>> import folium
        >>> generator = CircleMarkerGenerator(
        ...     CircleMarkerFeatureResolver(radius=PropertyMapper.from_item_attr('voltage', lambda v: v / 40))
        ... )
        >>> folium_map = folium.Map(location=[48, 9], zoom_start=6)
        >>> fg = generator.generate_objects_for_model_df(dataset.fetch('Node.Model'), folium.FeatureGroup('Nodes'))
        >>> fg = fg.add_to(folium_map)
        >>> folium_map.get_root().render().count('L.circleMarker(')
        4
    """

    def _feature_resolver_type(self) -> Type[CircleMarkerFeatureResolver]:
        return CircleMarkerFeatureResolver

    def generate(self, data_item: VisualizableDataItem, feature_group: folium.FeatureGroup) -> None:
        style = self.feature_resolver.resolve_feature(data_item)
        if style.location is None:
            return

        folium.CircleMarker(
            location=(style.location.y, style.location.x),
            tooltip=style.tooltip,
            popup=style.popup,
            radius=style.radius,
            color=style.border_color,
            fillColor=style.fill_color,
            fillOpacity=style.fill_opacity,
            weight=style.border_width,
        ).add_to(feature_group)
