from typing import List
from dataclasses import dataclass

from mescal.visualizations.folium_viz_system.system_base import StyleMapper, ResolvedStyle, StyleResolver


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


@dataclass
class ResolvedLineStyle(ResolvedStyle):
    """Specialized style container for line visualizations."""

    @property
    def line_color(self) -> str:
        return self.get('line_color')

    @property
    def line_width(self) -> float:
        return max(self.get('line_width'), 0.0)

    @property
    def line_opacity(self) -> float:
        return max(min(self.get('line_opacity'), 1.0), 0.0)

    @property
    def dash_pattern(self) -> List[int]:
        return self.get('dash_pattern')

    @property
    def reverse_path_direction(self) -> bool:
        return self.get('reverse_path_direction')

    @property
    def line_ant_path(self) -> bool:
        return self.get('line_ant_path')

    @property
    def line_ant_path_delay(self) -> int:
        return self.get('line_ant_path_delay')

    @property
    def line_ant_path_pulse_color(self) -> str:
        return self.get('line_ant_path_pulse_color')


class LineStyleResolver(StyleResolver[ResolvedLineStyle]):
    def __init__(
            self,
            line_color: StyleMapper | str = '#000000',
            line_width: StyleMapper | float = 3.0,
            line_opacity: StyleMapper | float = 1.0,
            dash_pattern: StyleMapper | List[int] = None,
            line_ant_path: StyleMapper | bool = False,
            line_ant_path_delay: StyleMapper | int = 1500,
            line_ant_path_pulse_color: StyleMapper | str = '#DBDBDB',
            reverse_path_direction: StyleMapper | bool = False,
            *style_mappers: StyleMapper,
    ):
        mappers = dict(
            line_color=line_color,
            line_width=line_width,
            line_opacity=line_opacity,
            dash_pattern=dash_pattern,
            line_ant_path=line_ant_path,
            line_ant_path_delay=line_ant_path_delay,
            line_ant_path_pulse_color=line_ant_path_pulse_color,
            reverse_path_direction=reverse_path_direction,
        )
        mappers = self._transform_static_values_to_style_mappers(mappers)
        self._validate_mapper_namings(mappers)
        super().__init__(list(mappers.values()) + list(style_mappers), style_type=ResolvedLineStyle)


@dataclass
class ResolvedLineTextStyle(ResolvedStyle):
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


class LineTextStyleResolver(StyleResolver[ResolvedLineTextStyle]):
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
            *style_mappers: StyleMapper,
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
        )
        mappers = self._transform_static_values_to_style_mappers(mappers)
        self._validate_mapper_namings(mappers)
        super().__init__(list(mappers.values()) + list(style_mappers), style_type=ResolvedLineTextStyle)


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
            *style_mappers: StyleMapper,
    ):
        mappers = dict(
            text_color=text_color,
            font_size=font_size,
            font_weight=font_weight,
            background_color=background_color,
            shadow_size=shadow_size,
            shadow_color=shadow_color,
        )
        mappers = self._transform_static_values_to_style_mappers(mappers)
        self._validate_mapper_namings(mappers)
        super().__init__(list(mappers.values()) + list(style_mappers), style_type=ResolvedTextOverlayStyle)
