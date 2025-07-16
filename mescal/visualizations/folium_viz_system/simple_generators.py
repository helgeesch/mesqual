from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import folium

from mescal.visualizations.folium_viz_system.base_system import ResolvedStyle, StyleResolver, FoliumObjectGenerator
from mescal.visualizations.folium_viz_system.map_data_item import MapDataItem
from mescal.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class TooltipGenerator:
    """Generates HTML tooltips from map data items."""

    def generate_tooltip(self, data_item: MapDataItem) -> str:
        """Generate HTML tooltip from data item."""
        tooltip_data = data_item.get_tooltip_data()

        html = '<table style="border-collapse: collapse;">\n'
        for key, value in tooltip_data.items():
            html += f'  <tr><td style="padding: 4px 8px;"><strong>{key}</strong></td>' \
                    f'<td style="text-align: right; padding: 4px 8px;">{value}</td></tr>\n'
        html += '</table>'
        return html


class PopupGenerator:
    """Generates HTML popups from map data items."""

    def generate_popup(self, data_item: MapDataItem) -> str:
        """Generate HTML popup from data item."""
        popup_data = data_item.get_tooltip_data()  # Can use same data as tooltip

        html = '<div style="font-family: Arial, sans-serif; max-width: 300px;">\n'
        html += '<table style="border-collapse: collapse; width: 100%;">\n'
        for key, value in popup_data.items():
            html += f'  <tr><td style="padding: 6px 12px; border-bottom: 1px solid #ddd;"><strong>{key}</strong></td>' \
                    f'<td style="padding: 6px 12px; text-align: right; border-bottom: 1px solid #ddd;">{value}</td></tr>\n'
        html += '</table></div>'
        return html




class IconGenerator(ABC):
    """Abstract base for generating folium icons."""

    @abstractmethod
    def generate_icon(self, data_item: MapDataItem, style: ResolvedStyle) -> folium.DivIcon:
        """Generate a folium icon for the data item."""
        pass


class _TMPIconGenerator(FoliumObjectGenerator, ABC):
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


