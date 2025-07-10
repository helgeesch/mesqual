from typing import Union, Callable, Any, List, Dict, Generic, Type
from dataclasses import dataclass, field

from mescal.typevars import ResolvedStyleType
from mescal.visualizations.folium_viz_system.map_data_item import MapDataItem


@dataclass
class StyleMapper:
    """Maps a data column to a visual property using a mapping function."""
    property: str  # e.g., 'color', 'width', 'opacity', 'height', 'arrow_speed', 'shadow_color'
    column: str | None = None  # column to get value from data source, None for static values
    mapping: Union[Callable, Any] = None  # mapper function or static value
    return_type: type = str  # expected return type for validation


@dataclass
class ResolvedStyle:
    """Container for resolved style properties."""
    properties: dict = field(default_factory=dict)

    def get(self, property: str, default=None):
        return self.properties.get(property, default)

    def __getitem__(self, key):
        return self.properties[key]

    def __setitem__(self, key, value):
        self.properties[key] = value

    def __contains__(self, key):
        return key in self.properties


class StyleResolver(Generic[ResolvedStyleType]):
    """Resolves styling for map data items using flexible property mappings."""

    def __init__(self, style_mappers: List[StyleMapper] = None, style_type: Type[ResolvedStyleType] = ResolvedStyle):
        self.style_mappers = {mapper.property: mapper for mapper in style_mappers}
        self.style_type = style_type

    def resolve_style(self, data_item: MapDataItem) -> ResolvedStyleType:
        """Resolve styling for a data item."""
        resolved = self.style_type()

        for prop, mapper in self.style_mappers.items():
            if mapper.column:
                value = data_item.get_styling_value(mapper.column)
            else:
                value = None

            if callable(mapper.mapping):
                resolved[prop] = mapper.mapping(value)
            else:
                resolved[prop] = mapper.mapping

        return resolved

    @classmethod
    def _validate_mapper_namings(cls, mappers: Dict[str, StyleMapper]) -> None:
        for key, mapper in mappers.items():
            if mapper.property != key:
                raise ValueError(
                    f"StyleMapper property not set correctly; StyleMapper for {key} must have property set to '{key}'."
                )

    @classmethod
    def _transform_static_values_to_style_mappers(cls, mappers: Dict[str, Any]) -> Dict[str, StyleMapper]:
        for key, mapper in list(mappers.items()):
            if not isinstance(mapper, StyleMapper):
                mappers[key] = StyleMapper(key, None, mapper, type(mapper))
        return mappers


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
        return min(self.get('line_opacity'), 1.0)

    @property
    def dash_pattern(self) -> str:
        return self.get('dash_pattern')


class LineStyleResolver(StyleResolver[ResolvedLineStyle]):
    def __init__(
            self,
            line_color: StyleMapper | str = '#000000',
            line_width: StyleMapper | float = 3.0,
            line_opacity: StyleMapper | float = 1.0,
            dash_pattern: StyleMapper | str = None,
            *style_mappers: StyleMapper,
    ):
        mappers = dict(
            line_color=line_color,
            line_width=line_width,
            line_opacity=line_opacity,
            dash_pattern=dash_pattern,
        )
        mappers = self._transform_static_values_to_style_mappers(mappers)
        self._validate_mapper_namings(mappers)
        super().__init__(list(mappers.values()) + list(style_mappers), style_type=ResolvedLineStyle)


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
