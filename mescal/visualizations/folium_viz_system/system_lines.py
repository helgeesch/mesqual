import hashlib
from dataclasses import dataclass
from typing import List, Type

import folium
from folium.plugins import AntPath, PolyLineOffset
from shapely import LineString

from mescal.visualizations.folium_viz_system.simple_generators import TooltipGenerator, PopupGenerator, logger
from mescal.visualizations.folium_viz_system.map_data_item import MapDataItem

from mescal.visualizations.folium_viz_system.base_system import ResolvedStyle, StyleResolver, StyleMapper, \
    FoliumObjectGenerator


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
        effective_offset = self._get_line_offset(coordinates, feature_group)

        if style.reverse_path_direction:
            coordinates = coordinates[::-1]

        line_kwargs = dict(
            locations=coordinates,
            color=style.line_color,
            weight=style.line_width,
            opacity=style.line_opacity,
            offset=effective_offset,
            tooltip=tooltip,
            dash_array=style.dash_pattern or None,
            popup=folium.Popup(popup, max_width=300) if popup else None,
        )

        if style.line_ant_path:
            if effective_offset not in [None, 0, 0.0]:
                logger.warning(
                    f'Trying to set an offset with an animated LineAntPath is not possible. '
                    f'Offset will not be applied for {data_item.get_name()}.'
                )
            line_kwargs.update(dict(
                paused=False,
                reverse=False,
                hardware_acceleration=False,
                delay=style.line_ant_path_delay,
                pulse_color=style.line_ant_path_pulse_color,
            ))
            poly_line = AntPath(**line_kwargs)
        else:
            poly_line = PolyLineOffset(**line_kwargs)

        poly_line.add_to(feature_group)

    def _get_line_offset(self, coordinates: list[tuple[float, float]], feature_group: folium.FeatureGroup) -> float:
        line_hash = self._hash_coordinates(coordinates)
        registry = self._get_registry_for_group(feature_group)
        offset_index = registry.get(line_hash, 0)
        registry[line_hash] = offset_index + 1
        offset_px = offset_index * self.offset_increment
        offset_side = 1 if offset_index % 2 == 0 else -1
        effective_offset = offset_px * offset_side
        return effective_offset

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
