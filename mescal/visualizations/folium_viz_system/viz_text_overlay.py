from dataclasses import dataclass
from typing import Callable, Type, Any

import folium

from mescal.visualizations.folium_viz_system.element_generators import TooltipGenerator, PopupGenerator
from mescal.visualizations.folium_viz_system.map_data_item import MapDataItem, KPIDataItem

from mescal.visualizations.folium_viz_system.base_viz_system import (
    ResolvedStyle,
    StyleResolver,
    StyleMapper,
    FoliumObjectGenerator,
)


@dataclass
class ResolvedTextOverlayStyle(ResolvedStyle):
    """Specialized style container for text overlay visualizations."""

    @property
    def text_color(self) -> str:
        return self.get('text_color')

    @property
    def font_size(self) -> str:
        return self.get('font_size')

    @property
    def font_weight(self) -> str:
        return self.get('font_weight')

    @property
    def background_color(self) -> str:
        return self.get('background_color')

    @property
    def shadow_size(self) -> float:
        return self.get('shadow_size')

    @property
    def shadow_color(self) -> str:
        return self.get('shadow_color')


class TextOverlayStyleResolver(StyleResolver[ResolvedTextOverlayStyle]):
    def __init__(
            self,
            text_color: StyleMapper | str = '#3A3A3A',
            font_size: StyleMapper | str = '10pt',
            font_weight: StyleMapper | str = 'bold',
            background_color: StyleMapper | str = None,
            shadow_size: StyleMapper | str = '0.5px',
            shadow_color: StyleMapper | str = '#F2F2F2',
            **style_mappers: StyleMapper | Any,
    ):
        mappers = dict(
            text_color=text_color,
            font_size=font_size,
            font_weight=font_weight,
            background_color=background_color,
            shadow_size=shadow_size,
            shadow_color=shadow_color,
            **style_mappers
        )
        super().__init__(style_type=ResolvedTextOverlayStyle, **mappers)


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
        return data_item.get_text_representation()

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
