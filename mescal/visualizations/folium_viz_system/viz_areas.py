from dataclasses import dataclass
from typing import Type

import folium
from shapely import Polygon, MultiPolygon

from mescal.visualizations.folium_viz_system.map_data_item import MapDataItem
from mescal.visualizations.folium_viz_system.base_viz_system import (
    ResolvedStyle,
    StyleResolver,
    StyleMapper,
    FoliumObjectGenerator,
)


@dataclass
class ResolvedAreaStyle(ResolvedStyle):
    """Specialized style container for area visualizations."""

    @property
    def fill_color(self) -> str:
        return self.get('fill_color')

    @property
    def border_color(self) -> str:
        return self.get('border_color')

    @property
    def border_width(self) -> float:
        return self.get('border_width')

    @property
    def fill_opacity(self) -> float:
        return min(self.get('fill_opacity'), 1.0)

    @property
    def highlight_border_width(self) -> float:
        return self.get('highlight_border_width')

    @property
    def highlight_fill_opacity(self) -> float:
        return min(self.get('highlight_fill_opacity'), 1.0)


class AreaStyleResolver(StyleResolver[ResolvedAreaStyle]):
    def __init__(
            self,
            fill_color: StyleMapper | str = '#D2D2D2',
            border_color: StyleMapper | str = 'white',
            border_width: StyleMapper | float = 2.0,
            fill_opacity: StyleMapper | float = 0.8,
            highlight_border_width: StyleMapper | float = 3.0,
            highlight_fill_opacity: StyleMapper | float = 1.0,
            *style_mappers: StyleMapper,
    ):
        mappers = dict(
            fill_color=fill_color,
            border_color=border_color,
            border_width=border_width,
            fill_opacity=fill_opacity,
            highlight_border_width=highlight_border_width,
            highlight_fill_opacity=highlight_fill_opacity,
        )
        mappers = self._transform_static_values_to_style_mappers(mappers)
        self._validate_mapper_namings(mappers)
        super().__init__(list(mappers.values()) + list(style_mappers), style_type=ResolvedAreaStyle)


class AreaGenerator(FoliumObjectGenerator[AreaStyleResolver]):
    """Generates folium GeoJson objects for area geometries."""

    def _style_resolver_type(self) -> Type[AreaStyleResolver]:
        return AreaStyleResolver

    def generate(self, data_item: MapDataItem, feature_group: folium.FeatureGroup) -> None:
        geometry = data_item.get_geometry()
        if not isinstance(geometry, (Polygon, MultiPolygon)):
            return

        style = self.style_resolver.resolve_style(data_item)
        tooltip = self.tooltip_generator.generate_tooltip(data_item)
        popup = self.popup_generator.generate_popup(data_item) if self.popup_generator else None

        style_dict = {
            'fillColor': style.fill_color,
            'color': style.border_color,
            'weight': style.border_width,
            'fillOpacity': style.fill_opacity
        }

        highlight_dict = style_dict.copy()
        highlight_dict['weight'] = style.highlight_border_width
        highlight_dict['fillOpacity'] = style.highlight_fill_opacity

        geojson_data = {
            "type": "Feature",
            "geometry": geometry.__geo_interface__,
            "properties": {"tooltip": tooltip}
        }

        geojson_kwargs = {
            'style_function': lambda x, s=style_dict: s,
            'highlight_function': lambda x, h=highlight_dict: h,
            'tooltip': folium.GeoJsonTooltip(fields=['tooltip'], aliases=[''], sticky=True)
        }

        if popup:
            geojson_kwargs['popup'] = folium.Popup(popup, max_width=300)

        folium.GeoJson(geojson_data, **geojson_kwargs).add_to(feature_group)
