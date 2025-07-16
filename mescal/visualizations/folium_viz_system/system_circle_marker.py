from dataclasses import dataclass
from typing import Type

import folium

from mescal.visualizations.folium_viz_system.map_data_item import MapDataItem
from mescal.visualizations.folium_viz_system.system_base import ResolvedStyle, StyleResolver, StyleMapper, \
    FoliumObjectGenerator


@dataclass
class ResolvedCircleMarkerStyle(ResolvedStyle):
    """Specialized style container for circle marker visualizations."""

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


class CircleMarkerStyleResolver(StyleResolver[ResolvedCircleMarkerStyle]):
    def __init__(
            self,
            fill_color: StyleMapper | str = '#D9D9D9',
            border_color: StyleMapper | str = 'white',
            radius: StyleMapper | float = 8.0,
            border_width: StyleMapper | float = 1.0,
            fill_opacity: StyleMapper | float = 1.0,
            *style_mappers: StyleMapper,
    ):
        mappers = dict(
            fill_color=fill_color,
            border_color=border_color,
            radius=radius,
            border_width=border_width,
            fill_opacity=fill_opacity,
        )
        mappers = self._transform_static_values_to_style_mappers(mappers)
        self._validate_mapper_namings(mappers)
        super().__init__(list(mappers.values()) + list(style_mappers), style_type=ResolvedCircleMarkerStyle)


class CircleMarkerGenerator(FoliumObjectGenerator[CircleMarkerStyleResolver]):
    """Generates folium CircleMarker objects for point geometries."""

    def _style_resolver_type(self) -> Type[CircleMarkerStyleResolver]:
        return CircleMarkerStyleResolver

    def generate(self, data_item: MapDataItem, feature_group: folium.FeatureGroup) -> None:
        try:
            location = data_item.get_location()
        except ValueError:
            return

        style = self.style_resolver.resolve_style(data_item)
        tooltip = self.tooltip_generator.generate_tooltip(data_item)
        popup = self.popup_generator.generate_popup(data_item) if self.popup_generator else None

        marker_kwargs = {'location': location, 'tooltip': tooltip}
        if popup:
            marker_kwargs['popup'] = folium.Popup(popup, max_width=300)

        circle_kwargs = {
            'radius': style.radius,
            'color': style.border_color,
            'fillColor': style.fill_color,
            'fillOpacity': style.fill_opacity,
            'weight': style.border_width,
            **marker_kwargs
        }
        folium.CircleMarker(**circle_kwargs).add_to(feature_group)
