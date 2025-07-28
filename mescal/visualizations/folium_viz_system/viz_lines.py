import hashlib
from dataclasses import dataclass
from typing import List, Type, Any

import folium
from folium.plugins import AntPath, PolyLineOffset
from shapely import LineString, MultiLineString

from mescal.visualizations.folium_viz_system.map_data_item import MapDataItem

from mescal.visualizations.folium_viz_system.base_viz_system import (
    ResolvedFeature,
    FeatureResolver,
    PropertyMapper,
    FoliumObjectGenerator,
)
from mescal.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ResolvedLineFeature(ResolvedFeature):
    """Specialized style container for line visualizations."""

    @property
    def geometry(self) -> LineString:
        return self.get('geometry')

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


class LineFeatureResolver(FeatureResolver[ResolvedLineFeature]):
    def __init__(
            self,
            line_color: PropertyMapper | str = '#000000',
            line_width: PropertyMapper | float = 3.0,
            line_opacity: PropertyMapper | float = 1.0,
            dash_pattern: PropertyMapper | List[int] = None,
            line_ant_path: PropertyMapper | bool = False,
            line_ant_path_delay: PropertyMapper | int = 1500,
            line_ant_path_pulse_color: PropertyMapper | str = '#DBDBDB',
            reverse_path_direction: PropertyMapper | bool = False,
            tooltip: PropertyMapper | str | bool = True,
            popup: PropertyMapper | folium.Popup | bool = False,
            geometry: PropertyMapper | LineString = None,
            **property_mappers: PropertyMapper | Any,
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
            tooltip=tooltip,
            popup=popup,
            geometry=self._explicit_or_fallback(geometry, self._default_line_string_mapper()),
            **property_mappers
        )
        super().__init__(feature_type=ResolvedLineFeature, **mappers)


class LineGenerator(FoliumObjectGenerator[LineFeatureResolver]):
    """Generates folium PolyLine objects for line geometries with optional per-feature-group offset tracking."""

    def __init__(
            self,
            feature_resolver: LineFeatureResolver = None,
            per_feature_group_offset_registry: bool = True,
            offset_increment: int = 5,
    ):
        super().__init__(feature_resolver)
        self.offset_increment = offset_increment
        self.per_feature_group_registry = per_feature_group_offset_registry
        self._global_registry: dict[str, int] = {}
        self._group_registry: dict[int, dict[str, int]] = {}

    def reset_registry(self) -> None:
        self._global_registry.clear()
        self._group_registry.clear()

    def _feature_resolver_type(self) -> Type[LineFeatureResolver]:
        return LineFeatureResolver

    def generate(self, data_item: MapDataItem, feature_group: folium.FeatureGroup) -> None:
        style = self.feature_resolver.resolve_feature(data_item)
        geometry = style.geometry
        if not isinstance(geometry, (LineString, MultiLineString)):
            return

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
            tooltip=style.tooltip,
            dash_array=style.dash_pattern or None,
            popup=style.popup,
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


if __name__ == '__main__':
    import os
    import webbrowser
    import pandas as pd
    from shapely.geometry import Polygon, LineString
    import folium

    from mescal.visualizations.value_mapping_system import (
        SegmentedContinuousColorscale,
        SegmentedContinuousOpacityMapping,
    )

    line_df = pd.DataFrame({
        'geometry': [
            LineString([(7.0, 50.0), (7.5, 52.0)]),
            LineString([(8.0, 50.0), (8.5, 52.0)]),
            LineString([(9.0, 50.0), (9.5, 52.0)]),
            LineString([(9.0, 50.0), (9.5, 52.0)]),
        ],
        'flow': [10, 20, 30, 29]
    }, index=['line1', 'line2', 'line3.1', 'line3.2'])

    m = folium.Map(location=[50.25, 8.0], zoom_start=7, tiles='CartoDB Positron')

    color_map = SegmentedContinuousColorscale.single_segment_autoscale_factory_from_array(
        values=line_df['flow'].values,
        colorscale=['green', 'blue', 'red']
    )

    opacity_map = SegmentedContinuousOpacityMapping.single_segment_autoscale_factory_from_array(
        values=line_df['flow'].values,
        output_range=(0.4, 0.9)
    )

    line_generator = LineGenerator(
        feature_resolver=LineFeatureResolver(
            line_color=PropertyMapper.from_item_attr('flow', color_map),
            line_opacity=PropertyMapper.from_item_attr('flow', opacity_map),
            line_width=10,
            tooltip=False,
            popup=True,
        ),
        offset_increment=15,
    )

    fg = folium.FeatureGroup(name='Test Lines')
    line_generator.generate_objects_for_model_df(line_df, fg)
    fg.add_to(m)

    m.add_child(folium.LayerControl())
    m.save('_tmp/map.html')
    webbrowser.open('file://' + os.path.abspath('_tmp/map.html'))
