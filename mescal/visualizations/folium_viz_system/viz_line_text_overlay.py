from dataclasses import dataclass
from typing import Callable, Type, Any

import folium
from folium.plugins import PolyLineTextPath
from shapely import LineString

from mescal.visualizations.folium_viz_system.map_data_item import MapDataItem, KPIDataItem
from mescal.visualizations.folium_viz_system.base_viz_system import (
    ResolvedStyle,
    StyleResolver,
    StyleMapper,
    FoliumObjectGenerator,
)


@dataclass
class ResolvedLineTextOverlayStyle(ResolvedStyle):
    @property
    def text_offset(self) -> float:
        offset = self.get('text_offset')
        if offset is None:
            return self.font_size / 2
        return offset

    @property
    def text_orientation(self) -> int:
        return self.get('text_orientation', 0)

    @property
    def text_repeat(self) -> bool:
        return self.get('text_repeat', False)

    @property
    def text_center(self) -> bool:
        return self.get('text_center', False)

    @property
    def text_below(self) -> bool:
        return self.get('text_below', False)

    @property
    def font_weight(self) -> str:
        return self.get('font_weight', 'bold')

    @property
    def font_size(self) -> int:
        return self.get('font_size', 12)

    @property
    def font_color(self) -> str:
        return self.get('font_color', '#000000')

    @property
    def reverse_path_direction(self) -> bool:
        return self.get('reverse_path_direction', False)


class LineTextOverlayStyleResolver(StyleResolver[ResolvedLineTextOverlayStyle]):
    def __init__(
            self,
            text_offset: StyleMapper | float = None,
            text_orientation: StyleMapper | int = 0,
            text_repeat: StyleMapper | bool = False,
            text_center: StyleMapper | bool = False,
            text_below: StyleMapper | bool = False,
            font_weight: StyleMapper | str = 'bold',
            font_size: StyleMapper | int = 12,
            font_color: StyleMapper | str = '#000000',
            reverse_path_direction: StyleMapper | bool = False,
            **style_mappers: StyleMapper | Any,
    ):
        mappers = dict(
            text_offset=text_offset,
            text_orientation=text_orientation,
            text_repeat=text_repeat,
            text_center=text_center,
            text_below=text_below,
            font_weight=font_weight,
            font_size=font_size,
            font_color=font_color,
            reverse_path_direction=reverse_path_direction,
            **style_mappers
        )
        super().__init__(style_type=ResolvedLineTextOverlayStyle, **mappers)


class LineTextOverlayGenerator(FoliumObjectGenerator[LineTextOverlayStyleResolver]):
    def __init__(
            self,
            style_resolver: LineTextOverlayStyleResolver = None,
            tooltip_generator=None,
            popup_generator=None,
            text_formatter: Callable[[MapDataItem], str] = None
    ):
        super().__init__(style_resolver or LineTextOverlayStyleResolver(), tooltip_generator, popup_generator)
        self.text_formatter = text_formatter or self._default_text_formatter

    def _default_text_formatter(self, data_item: MapDataItem) -> str:
        return data_item.get_text_representation()

    def _style_resolver_type(self) -> Type[LineTextOverlayStyleResolver]:
        return LineTextOverlayStyleResolver

    def generate(self, data_item: MapDataItem, feature_group: folium.FeatureGroup) -> None:
        geometry = data_item.get_geometry()
        if not isinstance(geometry, LineString):
            return

        text = self.text_formatter(data_item)
        if not text:
            return

        style = self.style_resolver.resolve_style(data_item)
        coordinates = [(lat, lon) for lon, lat in geometry.coords]

        if style.reverse_path_direction:
            coordinates = coordinates[::-1]

        invisible_line = folium.PolyLine(
            locations=coordinates,
            color=None,
            opacity=0.0,
        )
        invisible_line.add_to(feature_group)

        PolyLineTextPath(
            invisible_line,
            text=text,
            repeat=style.text_repeat,
            center=style.text_center,
            below=style.text_below,
            orientation=style.text_orientation,
            offset=style.text_offset,
            attributes={
                'font-weight': style.font_weight,
                'font-size': str(style.font_size),
                'fill': style.font_color,
            }
        ).add_to(feature_group)
