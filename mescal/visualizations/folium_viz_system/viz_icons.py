from abc import ABC

import folium

from mescal.visualizations.folium_viz_system.base_viz_system import FoliumObjectGenerator, StyleResolver
from mescal.visualizations.folium_viz_system.map_data_item import MapDataItem
from mescal.visualizations.folium_viz_system.element_generators import TooltipGenerator, PopupGenerator, IconGenerator


class PELIMINARY_IconGenerator(FoliumObjectGenerator, ABC):
    """Generates folium Marker or CircleMarker objects for point geometries."""

    def __init__(
            self,
            style_resolver: StyleResolver = None,
            tooltip_generator: TooltipGenerator = None,
            popup_generator: PopupGenerator = None,
            icon_generator: IconGenerator = None,
    ):
        super().__init__(style_resolver, tooltip_generator, popup_generator)
        self.icon_generator = icon_generator

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

        icon = self.icon_generator.generate_icon(data_item, style)
        folium.Marker(
            icon=icon,
            **marker_kwargs
        ).add_to(feature_group)

        folium.Marker(**marker_kwargs).add_to(feature_group)
