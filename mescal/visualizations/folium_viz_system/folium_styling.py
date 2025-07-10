from dataclasses import dataclass, field
from typing import Union, Callable, Any, List

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

    # Convenience properties for common styles
    @property
    def color(self) -> str:
        return self.get('color', '#000000')

    @property
    def width(self) -> float:
        return self.get('width', 1.0)

    @property
    def opacity(self) -> float:
        return self.get('opacity', 1.0)


@dataclass
class ResolvedAreaStyle(ResolvedStyle):
    """Specialized style container for area visualizations."""

    @property
    def fill_color(self) -> str:
        return self.get('color', '#D9D9D9')

    @property
    def border_color(self) -> str:
        return self.get('border_color', 'white')

    @property
    def border_width(self) -> float:
        return self.get('border_width', 2.0)

    @property
    def fill_opacity(self) -> float:
        return self.get('opacity', 0.7)


@dataclass
class ResolvedLineStyle(ResolvedStyle):
    """Specialized style container for line visualizations."""

    @property
    def line_color(self) -> str:
        return self.get('color', '#000000')

    @property
    def line_width(self) -> float:
        return self.get('width', 3.0)

    @property
    def line_opacity(self) -> float:
        return self.get('opacity', 1.0)

    @property
    def dash_pattern(self) -> str:
        return self.get('dash_pattern', None)


@dataclass
class ResolvedCircleMarkerStyle(ResolvedStyle):
    """Specialized style container for circle marker visualizations."""

    @property
    def fill_color(self) -> str:
        return self.get('color', '#D9D9D9')

    @property
    def border_color(self) -> str:
        return self.get('border_color', 'white')

    @property
    def radius(self) -> float:
        return self.get('width', 8.0)

    @property
    def border_width(self) -> float:
        return self.get('border_width', 1.0)

    @property
    def fill_opacity(self) -> float:
        return self.get('opacity', 1.0)


@dataclass
class ResolvedTextOverlayStyle(ResolvedStyle):
    """Specialized style container for text overlay visualizations."""

    @property
    def text_color(self) -> str:
        return self.get('text_color', '#3A3A3A')

    @property
    def shadow_color(self) -> str:
        return self.get('shadow_color', '#F2F2F2')

    @property
    def font_size(self) -> str:
        return self.get('font_size', '10pt')

    @property
    def font_weight(self) -> str:
        return self.get('font_weight', 'bold')

    @property
    def background_color(self) -> str:
        return self.get('background_color', None)


class StyleResolver:
    """Resolves styling for map data items using flexible property mappings."""

    def __init__(self, style_mappers: List[StyleMapper], style_type: type = ResolvedStyle):
        self.style_mappers = {mapper.property: mapper for mapper in style_mappers}
        self.style_type = style_type

    def resolve_style(self, data_item: MapDataItem) -> ResolvedStyle:
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
    def create_simple(
            cls,
            style_type: type = ResolvedStyle,
            color: Union[Callable, str] = '#D9D9D9',
            width: Union[Callable, float] = 3.0,
            opacity: Union[Callable, float] = 1.0,
            color_column: str = None,
            width_column: str = None,
            opacity_column: str = None,
            **additional_mappings
    ) -> 'StyleResolver':
        """Create a StyleResolver with common mappings."""
        mappers = [
            StyleMapper('color', color_column, color, str),
            StyleMapper('width', width_column, width, float),
            StyleMapper('opacity', opacity_column, opacity, float),
        ]

        # Add additional mappings
        for prop, mapping_info in additional_mappings.items():
            if isinstance(mapping_info, dict):
                mappers.append(StyleMapper(
                    property=prop,
                    column=mapping_info.get('column'),
                    mapping=mapping_info.get('mapping'),
                    return_type=mapping_info.get('return_type', str)
                ))
            else:
                mappers.append(StyleMapper(prop, None, mapping_info, type(mapping_info)))

        return cls(mappers, style_type)
