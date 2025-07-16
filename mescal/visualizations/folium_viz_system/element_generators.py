from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import folium

from mescal.visualizations.folium_viz_system.map_data_item import MapDataItem
from mescal.utils.logging import get_logger

if TYPE_CHECKING:
    from mescal.visualizations.folium_viz_system.base_viz_system import ResolvedStyle

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
    def generate_icon(self, data_item: MapDataItem, style: 'ResolvedStyle') -> folium.DivIcon:
        """Generate a folium icon for the data item."""
        pass
