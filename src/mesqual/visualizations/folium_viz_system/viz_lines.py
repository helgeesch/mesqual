import logging
import hashlib
from dataclasses import dataclass
from typing import List, Type, Any

import folium
from folium.plugins import AntPath, PolyLineOffset
from shapely import LineString, MultiLineString

from mesqual.visualizations.folium_viz_system.visualizable_data_item import VisualizableDataItem

from mesqual.visualizations.folium_viz_system.base_viz_system import (
    ResolvedFeature,
    FeatureResolver,
    PropertyMapper,
    FoliumObjectGenerator,
)

logger = logging.getLogger(__name__)


@dataclass
class ResolvedLineFeature(ResolvedFeature):
    """
    Resolved visual properties for line/connection map elements.

    Container for all computed styling properties of line visualizations,
    including colors, widths, dash patterns, and animation effects.
    Used by LineGenerator to create folium PolyLine objects.
    """

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
    """
    Resolves visual properties for line/connection map elements.

    Specialized feature resolver for line visualizations that handles LineString
    geometries, styling options like colors and widths, and animated ant paths.
    Commonly used for visualizing transmission lines, interconnectors, or any
    linear connections between geographic points.

    Args:
        line_color: Line stroke color (static value or PropertyMapper)
        line_width: Line width in pixels (static value or PropertyMapper)
        line_opacity: Line transparency 0-1 (static value or PropertyMapper)
        dash_pattern: Dash pattern as list of integers (static value or PropertyMapper)
        line_ant_path: Enable animated "marching ants" effect (static value or PropertyMapper)
        line_ant_path_delay: Animation delay in milliseconds (static value or PropertyMapper)
        line_ant_path_pulse_color: Color of animation pulses (static value or PropertyMapper)
        reverse_path_direction: Reverse line direction (static value or PropertyMapper)
        tooltip: Tooltip content, see `FeatureResolver`.
        popup: Popup content, see `FeatureResolver`.
        geometry: LineString geometry (defaults to 'geometry' attribute)
        **property_mappers: Additional custom property mappings

    Examples:
        Static styling:

        >>> resolver = LineFeatureResolver(line_color='#FF0000', line_width=3.0, line_opacity=0.8)

        Color and width from the absolute flow, an ant path for large flows, and the
        geometry from the `geo_line_string` column of the border model:

        >>> from mesqual.visualizations.folium_viz_system import PropertyMapper
        >>> from mesqual.visualizations.value_mapping_system import (
        ...     SegmentedContinuousColorscale,
        ...     SegmentedContinuousLineWidthMapping,
        ... )
        >>> resolver = LineFeatureResolver(
        ...     line_color=PropertyMapper.from_kpi_value(
        ...         SegmentedContinuousColorscale(segments={(0, 1000): 'Blues'}), use_abs_kpi_value=True
        ...     ),
        ...     line_width=PropertyMapper.from_kpi_value(
        ...         SegmentedContinuousLineWidthMapping(segments={(0, 1000): [1.0, 6.0]}), use_abs_kpi_value=True
        ...     ),
        ...     line_ant_path=PropertyMapper.from_kpi_value(lambda v: abs(v) > 100),
        ...     geometry=PropertyMapper.from_item_attr('geo_line_string'),
        ... )
    """

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
            **property_mappers,
        )
        super().__init__(feature_type=ResolvedLineFeature, **mappers)


class LineGenerator(FoliumObjectGenerator[LineFeatureResolver]):
    """Generates folium PolyLine objects for lines, with offsets for parallel lines.

    Creates one line per data item with the styling of its
    `LineFeatureResolver`, with tooltip and popup. Lines with the same
    coordinates in the same order get an offset; ant paths get none. A line
    can be an animated ant path, for example to show the flow direction.
    Typical uses are transmission lines or borders with flows, capacities or
    utilization.

    Args:
        feature_resolver: `LineFeatureResolver` for the visual properties.
        per_feature_group_offset_registry: Track the offsets per feature group.
        offset_increment: Offset in pixels between lines with the same coordinates.
            Default: 5.

    Examples:
        >>> from mesqual.kpis import Aggregations, FlagAggKPIBuilder
        >>> from mesqual.visualizations.folium_viz_system import PropertyMapper
        >>> dataset.add_kpis_from_definitions(
        ...     FlagAggKPIBuilder().for_flag('CountryBorder.Results.flow_sent_net').with_aggregation(Aggregations.Mean).build()
        ... )
        >>> flows = dataset.kpi_collection.filter(flag='CountryBorder.Results.flow_sent_net')
        >>> import folium
        >>> from mesqual.visualizations.value_mapping_system import SegmentedContinuousLineWidthMapping
        >>> widths = SegmentedContinuousLineWidthMapping(segments={(0, 1000): [1.0, 6.0]})
        >>> generator = LineGenerator(
        ...     LineFeatureResolver(
        ...         line_width=PropertyMapper.from_kpi_value(widths, use_abs_kpi_value=True),
        ...         reverse_path_direction=PropertyMapper.from_kpi_value(lambda v: v < 0),
        ...         line_ant_path=True,
        ...         geometry=PropertyMapper.from_item_attr('geo_line_string'),
        ...     ),
        ...     offset_increment=10,
        ... )
        >>> fg = generator.generate_objects_for_kpi_collection(flows, folium.FeatureGroup('Flows'))
    """

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

    def generate(self, data_item: VisualizableDataItem, feature_group: folium.FeatureGroup) -> None:
        """
        Generate and add a folium PolyLine to the feature group with automatic offset.

        Args:
            data_item: Data item containing LineString geometry and associated data
            feature_group: Folium feature group to add the line to
        """
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
            line_kwargs.update(
                dict(
                    paused=False,
                    reverse=False,
                    hardware_acceleration=False,
                    delay=style.line_ant_path_delay,
                    pulse_color=style.line_ant_path_pulse_color,
                )
            )
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
        return hashlib.md5(str(rounded).encode('utf-8')).hexdigest()
