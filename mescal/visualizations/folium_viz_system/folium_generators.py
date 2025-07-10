from abc import ABC, abstractmethod
from typing import Callable

import folium
from shapely import Polygon, MultiPolygon, LineString

from mescal.visualizations.folium_viz_system.folium_styling import ResolvedStyle, StyleResolver, ResolvedAreaStyle, \
    ResolvedLineStyle, ResolvedCircleMarkerStyle, ResolvedTextOverlayStyle
from mescal.visualizations.folium_viz_system.map_data_item import MapDataItem, KPIDataItem


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


class FoliumObjectGenerator(ABC):
    """Abstract base for generating folium objects."""

    def __init__(
            self,
            style_resolver: StyleResolver = None,
            tooltip_generator: TooltipGenerator = None,
            popup_generator: PopupGenerator = None
    ):
        self.style_resolver = style_resolver or StyleResolver([])
        self.tooltip_generator = tooltip_generator or TooltipGenerator()
        self.popup_generator = popup_generator

    @abstractmethod
    def generate(self, data_item: MapDataItem, feature_group: folium.FeatureGroup) -> None:
        """Generate folium object and add it to the feature group."""
        pass


class AreaGenerator(FoliumObjectGenerator):
    """Generates folium GeoJson objects for area geometries."""

    def generate(self, data_item: MapDataItem, feature_group: folium.FeatureGroup) -> None:
        geometry = data_item.get_geometry()
        if not isinstance(geometry, (Polygon, MultiPolygon)):
            return

        style = self.style_resolver.resolve_style(data_item)
        tooltip = self.tooltip_generator.generate_tooltip(data_item)
        popup = self.popup_generator.generate_popup(data_item) if self.popup_generator else None

        # Use specialized area style properties if available
        if isinstance(style, ResolvedAreaStyle):
            style_dict = {
                'fillColor': style.fill_color,
                'color': style.border_color,
                'weight': style.border_width,
                'fillOpacity': style.fill_opacity
            }
        else:
            style_dict = {
                'fillColor': style.color,
                'color': style.get('border_color', 'white'),
                'weight': style.width,
                'fillOpacity': style.opacity
            }

        highlight_dict = style_dict.copy()
        highlight_dict['weight'] = style_dict['weight'] * 1.5
        highlight_dict['fillOpacity'] = min(style_dict['fillOpacity'] * 1.5, 1.0)

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


class LineGenerator(FoliumObjectGenerator):
    """Generates folium PolyLine objects for line geometries."""

    def generate(self, data_item: MapDataItem, feature_group: folium.FeatureGroup) -> None:
        geometry = data_item.get_geometry()
        if not isinstance(geometry, LineString):
            return

        style = self.style_resolver.resolve_style(data_item)
        tooltip = self.tooltip_generator.generate_tooltip(data_item)
        popup = self.popup_generator.generate_popup(data_item) if self.popup_generator else None

        coordinates = [(lat, lon) for lon, lat in geometry.coords]

        # Use specialized line style properties if available
        if isinstance(style, ResolvedLineStyle):
            line_kwargs = {
                'locations': coordinates,
                'color': style.line_color,
                'weight': style.line_width,
                'opacity': style.line_opacity,
                'tooltip': tooltip
            }
            if style.dash_pattern:
                line_kwargs['dashArray'] = style.dash_pattern
        else:
            line_kwargs = {
                'locations': coordinates,
                'color': style.color,
                'weight': style.width,
                'opacity': style.opacity,
                'tooltip': tooltip
            }
            if 'dash_pattern' in style:
                line_kwargs['dashArray'] = style['dash_pattern']

        if popup:
            line_kwargs['popup'] = folium.Popup(popup, max_width=300)

        folium.PolyLine(**line_kwargs).add_to(feature_group)


class NodeGenerator(FoliumObjectGenerator):
    """Generates folium Marker or CircleMarker objects for point geometries."""

    def __init__(
            self,
            style_resolver: StyleResolver = None,
            tooltip_generator: TooltipGenerator = None,
            popup_generator: PopupGenerator = None,
            icon_generator: IconGenerator = None,
            use_circle_marker: bool = True
    ):
        super().__init__(style_resolver, tooltip_generator, popup_generator)
        self.icon_generator = icon_generator
        self.use_circle_marker = use_circle_marker

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

        if self.icon_generator:
            icon = self.icon_generator.generate_icon(data_item, style)
            folium.Marker(
                icon=icon,
                **marker_kwargs
            ).add_to(feature_group)
        elif self.use_circle_marker:
            # Use specialized circle marker style properties if available
            if isinstance(style, ResolvedCircleMarkerStyle):
                circle_kwargs = {
                    'radius': style.radius,
                    'color': style.border_color,
                    'fillColor': style.fill_color,
                    'fillOpacity': style.fill_opacity,
                    'weight': style.border_width,
                    **marker_kwargs
                }
            else:
                circle_kwargs = {
                    'radius': style.width,
                    'color': style.get('border_color', 'white'),
                    'fillColor': style.color,
                    'fillOpacity': style.opacity,
                    'weight': style.get('border_width', 1),
                    **marker_kwargs
                }
            folium.CircleMarker(**circle_kwargs).add_to(feature_group)
        else:
            folium.Marker(**marker_kwargs).add_to(feature_group)


class TextOverlayGenerator(FoliumObjectGenerator):
    """Generates text overlays for map data items."""

    def __init__(
            self,
            style_resolver: StyleResolver = None,
            tooltip_generator: TooltipGenerator = None,
            popup_generator: PopupGenerator = None,
            text_formatter: Callable[[MapDataItem], str] = None
    ):
        super().__init__(style_resolver, tooltip_generator, popup_generator)
        self.text_formatter = text_formatter or self._default_text_formatter

    def _default_text_formatter(self, data_item: MapDataItem) -> str:
        if isinstance(data_item, KPIDataItem):
            return f"{data_item.kpi.value:.1f}"
        return str(data_item.object_id)

    def generate(self, data_item: MapDataItem, feature_group: folium.FeatureGroup) -> None:
        try:
            location = data_item.get_location()
        except ValueError:
            return

        style = self.style_resolver.resolve_style(data_item)
        text = self.text_formatter(data_item)
        popup = self.popup_generator.generate_popup(data_item) if self.popup_generator else None

        # Use specialized text overlay style properties if available
        if isinstance(style, ResolvedTextOverlayStyle):
            text_color = style.text_color
            shadow_color = style.shadow_color
            font_size = style.font_size
            font_weight = style.font_weight
        else:
            text_color = style.get('text_color', self._get_contrasting_color(style.color))
            shadow_color = style.get('shadow_color', self._get_shadow_color(text_color))
            font_size = style.get('font_size', '10pt')
            font_weight = style.get('font_weight', 'bold')

        icon_html = f'''
            <div style="
                position: absolute;
                left: 50%;
                top: 50%;
                transform: translate(-50%, -50%);
                text-align: center;
                font-size: {font_size};
                font-weight: {font_weight};
                color: {text_color};
                white-space: nowrap;
                text-shadow:
                   -0.5px -0.5px 0 {shadow_color},  
                    0.5px -0.5px 0 {shadow_color},
                   -0.5px  0.5px 0 {shadow_color},
                    0.5px  0.5px 0 {shadow_color};
            ">
                {text}
            </div>
        '''

        marker_kwargs = {
            'location': location,
            'icon': folium.DivIcon(html=icon_html)
        }

        if popup:
            marker_kwargs['popup'] = folium.Popup(popup, max_width=300)

        folium.Marker(**marker_kwargs).add_to(feature_group)

    def _get_contrasting_color(self, surface_color: str) -> str:
        """Get contrasting text color for a surface color."""
        if self._is_dark(surface_color):
            return '#F2F2F2'
        return '#3A3A3A'

    def _get_shadow_color(self, text_color: str) -> str:
        """Get shadow color for text."""
        if text_color == '#F2F2F2':
            return '#3A3A3A'
        return '#F2F2F2'

    @staticmethod
    def _is_dark(color: str) -> bool:
        """Check if a color is dark."""
        if not color.startswith('#'):
            return False
        try:
            r, g, b = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
            return (0.299 * r + 0.587 * g + 0.114 * b) < 160
        except (ValueError, IndexError):
            return False
