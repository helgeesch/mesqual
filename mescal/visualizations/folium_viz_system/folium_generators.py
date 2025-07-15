from abc import ABC, abstractmethod
from typing import Callable, Generic, TYPE_CHECKING, Type
import hashlib

import pandas as pd
import folium
from folium.plugins import PolyLineTextPath, PolyLineOffset
from shapely import Polygon, MultiPolygon, LineString

from mescal.typevars import StyleResolverType, ResolvedStyleType
from mescal.visualizations.folium_viz_system.folium_styling import (
    ResolvedStyle,
    StyleResolver,
    ResolvedAreaStyle,
    AreaStyleResolver,
    ResolvedLineStyle,
    LineStyleResolver,
    ResolvedCircleMarkerStyle,
    CircleMarkerStyleResolver,
    ResolvedTextOverlayStyle,
    TextOverlayStyleResolver,
)
from mescal.visualizations.folium_viz_system.map_data_item import MapDataItem, ModelDataItem, KPIDataItem

if TYPE_CHECKING:
    from mescal.kpis import KPI, KPICollection


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


class LineGenerator(FoliumObjectGenerator[LineStyleResolver]):
    """Generates folium PolyLine objects for line geometries with optional per-feature-group offset tracking."""

    def __init__(
            self,
            style_resolver: LineStyleResolver = None,
            tooltip_generator: TooltipGenerator = None,
            popup_generator: PopupGenerator = None,
            per_feature_group_offset_registry: bool = True,
            offset_increment: int = 5,
    ):
        super().__init__(style_resolver, tooltip_generator, popup_generator)
        self.offset_increment = offset_increment
        self.per_feature_group_registry = per_feature_group_offset_registry
        self._global_registry: dict[str, int] = {}
        self._group_registry: dict[int, dict[str, int]] = {}

    def reset_registry(self) -> None:
        self._global_registry.clear()
        self._group_registry.clear()

    def _style_resolver_type(self) -> Type[LineStyleResolver]:
        return LineStyleResolver

    def generate(self, data_item: MapDataItem, feature_group: folium.FeatureGroup) -> None:
        geometry = data_item.get_geometry()
        if not isinstance(geometry, LineString):
            return

        style = self.style_resolver.resolve_style(data_item)
        tooltip = self.tooltip_generator.generate_tooltip(data_item)
        popup = self.popup_generator.generate_popup(data_item) if self.popup_generator else None

        coordinates = [(lat, lon) for lon, lat in geometry.coords]
        line_hash = self._hash_coordinates(coordinates)

        registry = self._get_registry_for_group(feature_group)

        offset_index = registry.get(line_hash, 0)
        registry[line_hash] = offset_index + 1

        offset_px = offset_index * self.offset_increment
        offset_side = 1 if offset_index % 2 == 0 else -1
        effective_offset = offset_px * offset_side

        poly_line = PolyLineOffset(
            locations=coordinates,
            color=style.line_color,
            weight=style.line_width,
            opacity=style.line_opacity,
            offset=effective_offset,
            tooltip=tooltip,
            dash_array=style.dash_pattern or None,
            popup=folium.Popup(popup, max_width=300) if popup else None,
        )

        poly_line.add_to(feature_group)

        if style.line_text_path:
            PolyLineTextPath(
                poly_line,
                text=style.line_text_path,
                repeat=style.line_text_path_repeat,
                center=style.line_text_path_center,
                below=style.line_text_path_below,
                orientation=style.line_text_path_orientation,
                attributes={
                    'font-weight': style.line_text_path_font_weight,
                    'font-size': str(style.line_text_path_font_size)
                }
            ).add_to(feature_group)

    def _get_registry_for_group(self, feature_group: folium.FeatureGroup) -> dict[str, int]:
        if not self.per_feature_group_registry:
            return self._global_registry
        group_id = id(feature_group)
        if group_id not in self._group_registry:
            self._group_registry[group_id] = {}
        return self._group_registry[group_id]

    def _hash_coordinates(self, coordinates: list[tuple[float, float]]) -> str:
        rounded = [(round(lat, 6), round(lon, 6)) for lat, lon in coordinates]
        return hashlib.md5(str(rounded).encode("utf-8")).hexdigest()


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


class TextOverlayGenerator(FoliumObjectGenerator[TextOverlayStyleResolver]):
    """Generates text overlays for map data items."""

    def __init__(
            self,
            style_resolver: TextOverlayStyleResolver = None,
            tooltip_generator: TooltipGenerator = None,
            popup_generator: PopupGenerator = None,
            text_formatter: Callable[[MapDataItem], str] = None
    ):
        super().__init__(style_resolver or TextOverlayStyleResolver(), tooltip_generator, popup_generator)
        self.text_formatter = text_formatter or self._default_text_formatter

    def _style_resolver_type(self) -> Type[TextOverlayStyleResolver]:
        return TextOverlayStyleResolver

    def _default_text_formatter(self, data_item: MapDataItem) -> str:
        if isinstance(data_item, KPIDataItem):
            # TODO: use quantity and pretty value; auto-num-of-decimals and so on...
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

        text_color = style.text_color
        font_size = style.font_size
        font_weight = style.font_weight
        shadow_size = style.shadow_size
        shadow_color = style.shadow_color

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
                   -{shadow_size} -{shadow_size} 0 {shadow_color},  
                    {shadow_size} -{shadow_size} 0 {shadow_color},
                   -{shadow_size}  {shadow_size} 0 {shadow_color},
                    {shadow_size}  {shadow_size} 0 {shadow_color};
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
