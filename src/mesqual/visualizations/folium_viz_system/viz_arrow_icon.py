from dataclasses import dataclass
from typing import Type, Any, TYPE_CHECKING, Union
import base64

from shapely import Point
import folium

from mesqual.visualizations.folium_viz_system.visualizable_data_item import VisualizableDataItem
from mesqual.visualizations.folium_viz_system.base_viz_system import (
    ResolvedFeature,
    FeatureResolver,
    PropertyMapper,
    FoliumObjectGenerator,
)

if TYPE_CHECKING:
    from captain_arro import ArrowTypeEnum


@dataclass
class ResolvedArrowIconFeature(ResolvedFeature):
    """Resolved visual properties of an animated arrow icon: location, angle, arrow type, size, color and animation.

    `ArrowIconGenerator` draws one marker with this icon per data item.
    """

    @property
    def location(self) -> Point:
        return self.get('location')

    @property
    def offset(self) -> tuple[float, float]:
        return self.get('offset')

    @property
    def azimuth_angle(self) -> float:
        return self.get('azimuth_angle')

    @property
    def arrow_type(self) -> 'ArrowTypeEnum':
        return self.get('arrow_type')

    @property
    def color(self) -> str:
        return self.get('color')

    @property
    def stroke_width(self) -> int:
        return self.get('stroke_width')

    @property
    def width(self) -> int:
        return self.get('width')

    @property
    def height(self) -> int:
        return self.get('height')

    @property
    def speed_in_px_per_second(self) -> float:
        return self.get('speed_in_px_per_second')

    @property
    def speed_in_duration_seconds(self) -> float:
        return self.get('speed_in_duration_seconds')

    @property
    def num_arrows(self) -> int:
        return self.get('num_arrows')

    @property
    def opacity(self) -> float:
        return self.get('opacity')

    @property
    def reverse_direction(self) -> bool:
        return self.get('reverse_direction')


class ArrowIconFeatureResolver(FeatureResolver[ResolvedArrowIconFeature]):
    """
    Resolves the visual properties of an animated arrow icon, for example a
    border flow drawn as an arrow in the direction of the flow.

    The arrow types and the animation come from the package `captain_arro`.

    Args:
        arrow_type: Type of arrow from ArrowTypeEnum (static value or PropertyMapper)
        color: Arrow color (static value or PropertyMapper)
        stroke_width: Arrow stroke width in pixels (static value or PropertyMapper)
        width: Arrow icon width in pixels (static value or PropertyMapper)
        height: Arrow icon height in pixels (static value or PropertyMapper)
        speed_in_px_per_second: Animation speed in pixels/second (static value or PropertyMapper).
            Set it to None to use `speed_in_duration_seconds`.
        speed_in_duration_seconds: Animation duration in seconds (static value or PropertyMapper).
            Needs `speed_in_px_per_second=None`.
        num_arrows: Number of animated arrows (static value or PropertyMapper)
        opacity: Icon opacity 0-1 (static value or PropertyMapper)
        reverse_direction: Reverse arrow direction (static value or PropertyMapper)
        tooltip: Tooltip content, see `FeatureResolver`.
        popup: Popup content, see `FeatureResolver`.
        location: Point of the icon. Default: the first `Point` of the item
            attributes `location`, `projection_point`, `centroid` and `midpoint`;
            else a point from the attributes `lat` and `lon` or `latitude` and
            `longitude` if both values are instances of Python `int` or `float`
            (`numpy.float64` is one, `numpy.int64` is not); else from the
            attribute `geometry`: the point, the
            representative point of a polygon, or the midpoint of a line. An
            item without location gets no icon.
        offset: Offset of the icon from its location as (x, y) (static value or PropertyMapper)
        azimuth_angle: Arrow rotation angle in degrees (static value or PropertyMapper).
            By default, it reads `azimuth_angle`, `rotation_angle` or `projection_angle`
            from the data item. For an item without an angle, `ArrowIconGenerator`
            raises a TypeError.
        **property_mappers: Additional custom property mappings

    Examples:
        Static styling:

        >>> from captain_arro import ArrowTypeEnum
        >>> resolver = ArrowIconFeatureResolver(
        ...     arrow_type=ArrowTypeEnum.MOVING_FLOW_ARROW,
        ...     color='#FF0000',
        ...     width=50,
        ...     height=30,
        ... )

        Direction from the sign of the flow. The location and the angle come from the
        `projection_point` and `azimuth_angle` of the border model:

        >>> from mesqual.kpis import Aggregations, FlagAggKPIBuilder
        >>> from mesqual.visualizations.folium_viz_system import KPIDataItem, PropertyMapper
        >>> dataset.add_kpis_from_definitions(
        ...     FlagAggKPIBuilder().for_flag('CountryBorder.Results.flow_sent_net').with_aggregation(Aggregations.Mean).build()
        ... )
        >>> flow = KPIDataItem(dataset.kpi_collection.filter(object_name='C1 - C2')[0])
        >>> resolver = ArrowIconFeatureResolver(
        ...     arrow_type=ArrowTypeEnum.MOVING_FLOW_ARROW,
        ...     reverse_direction=PropertyMapper.from_kpi_value(lambda v: v < 0),
        ... )
        >>> feature = resolver.resolve_feature(flow)
        >>> print(feature.location, feature.azimuth_angle, feature.reverse_direction)
        POINT (9 48) 90.0 True
    """

    def __init__(
        self,
        arrow_type: Union[PropertyMapper, 'ArrowTypeEnum'] = None,
        color: PropertyMapper | str = '#2563eb',
        stroke_width: PropertyMapper | int = 8,
        width: PropertyMapper | int = 60,
        height: PropertyMapper | int = 60,
        speed_in_px_per_second: PropertyMapper | float | None = 20.0,
        speed_in_duration_seconds: PropertyMapper | float | None = None,
        num_arrows: PropertyMapper | int = 3,
        opacity: PropertyMapper | float = 0.8,
        reverse_direction: PropertyMapper | bool = False,
        tooltip: PropertyMapper | str | bool = True,
        popup: PropertyMapper | folium.Popup | bool = False,
        location: PropertyMapper | Point = None,
        offset: PropertyMapper | tuple[float, float] = (0, 0),
        azimuth_angle: PropertyMapper | float = None,
        **property_mappers: PropertyMapper | Any,
    ):
        from captain_arro import ArrowTypeEnum

        mappers = dict(
            arrow_type=arrow_type or ArrowTypeEnum.MOVING_FLOW_ARROW,
            color=color,
            stroke_width=stroke_width,
            width=width,
            height=height,
            speed_in_px_per_second=speed_in_px_per_second,
            speed_in_duration_seconds=speed_in_duration_seconds,
            num_arrows=num_arrows,
            opacity=opacity,
            reverse_direction=reverse_direction,
            tooltip=tooltip,
            popup=popup,
            location=self._explicit_or_fallback(location, self._default_location_mapper()),
            offset=offset,
            azimuth_angle=self._explicit_or_fallback(azimuth_angle, self._default_rotation_angle_mapper()),
            **property_mappers,
        )
        super().__init__(feature_type=ResolvedArrowIconFeature, **mappers)

    @staticmethod
    def _default_rotation_angle_mapper() -> PropertyMapper:

        def get_rotation_angle(data_item: VisualizableDataItem) -> float | None:
            for k in ['azimuth_angle', 'rotation_angle', 'projection_angle']:
                if data_item.object_has_attribute(k):
                    angle = data_item.get_object_attribute(k)
                    return angle
            return None

        return PropertyMapper(get_rotation_angle)


class ArrowIconGenerator(FoliumObjectGenerator[ArrowIconFeatureResolver]):
    """Generates folium Marker objects with animated SVG arrow icons.

    Creates one marker per data item with an animated arrow icon of the
    captain_arro library, rotated to the azimuth angle of the item. Typical uses
    are border flows or price spreads, with the size or color from the value.

    Examples:
        >>> from mesqual.kpis import Aggregations, FlagAggKPIBuilder
        >>> from mesqual.visualizations.folium_viz_system import PropertyMapper
        >>> dataset.add_kpis_from_definitions(
        ...     FlagAggKPIBuilder().for_flag('CountryBorder.Results.flow_sent_net').with_aggregation(Aggregations.Mean).build()
        ... )
        >>> flows = dataset.kpi_collection.filter(flag='CountryBorder.Results.flow_sent_net')
        >>> import folium
        >>> from captain_arro import ArrowTypeEnum
        >>> from mesqual.visualizations.value_mapping_system import SegmentedContinuousInputToContinuousOutputMapping
        >>> sizes = SegmentedContinuousInputToContinuousOutputMapping(segments={(0, 1000): [20, 60]})
        >>> generator = ArrowIconGenerator(
        ...     ArrowIconFeatureResolver(
        ...         arrow_type=PropertyMapper.from_kpi_value(
        ...             lambda v: ArrowTypeEnum.MOVING_FLOW_ARROW if abs(v) > 50 else ArrowTypeEnum.BOUNCING_SPREAD_ARROW
        ...         ),
        ...         width=PropertyMapper.from_kpi_value(sizes, use_abs_kpi_value=True),
        ...         reverse_direction=PropertyMapper.from_kpi_value(lambda v: v < 0),
        ...         speed_in_px_per_second=None,
        ...         speed_in_duration_seconds=4,
        ...     )
        ... )
        >>> fg = generator.generate_objects_for_kpi_collection(flows, folium.FeatureGroup('Flow arrows'))
    """

    def _feature_resolver_type(self) -> Type[ArrowIconFeatureResolver]:
        return ArrowIconFeatureResolver

    def generate(self, data_item: VisualizableDataItem, feature_group: folium.FeatureGroup) -> None:
        """
        Generate and add a folium Marker with animated SVG arrow icon.

        Args:
            data_item: Data item containing point location and associated data
            feature_group: Folium feature group to add the arrow marker to
        """
        style = self.feature_resolver.resolve_feature(data_item)
        if style.location is None:
            return

        svg_content = self._generate_arrow_svg(style, data_item)
        encoded_svg = base64.b64encode(svg_content.encode()).decode()

        angle = (
            float(style.azimuth_angle) - 90
        )  # Folium counts clockwise from right-pointing direction; normal convention is CCW
        x_offset, y_offset = style.offset

        icon_html = f'''
            <div style="
                position: absolute;
                left: 50%;
                top: 50%;
                transform: translate(-50%, -50%) translate({x_offset}px, {-y_offset}px) rotate({angle}deg);
                opacity: {style.opacity};
            ">
                <img src="data:image/svg+xml;base64,{encoded_svg}" 
                     width="{style.width}" 
                     height="{style.height}">
                </img>
            </div>
        '''

        icon = folium.DivIcon(
            html=icon_html, icon_size=(style.width, style.height), icon_anchor=(style.width // 2, style.height // 2)
        )
        folium.Marker(
            location=(style.location.y, style.location.x), icon=icon, tooltip=style.tooltip, popup=style.popup
        ).add_to(feature_group)

    def _generate_arrow_svg(self, style: ResolvedArrowIconFeature, data_item: VisualizableDataItem) -> str:
        from inspect import signature
        from captain_arro import get_generator_for_arrow_type

        def safe_init(cls: type, kwargs: dict):
            init_params = signature(cls.__init__).parameters
            accepted_keys = {
                k
                for k in init_params
                if k != 'self'
                and init_params[k].kind in (init_params[k].POSITIONAL_OR_KEYWORD, init_params[k].KEYWORD_ONLY)
            }
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_keys}
            return cls(**filtered_kwargs)

        arrow_type = style.arrow_type
        generator_class = get_generator_for_arrow_type(arrow_type)
        arrow_style_kwargs = style.to_dict()
        if 'direction' not in arrow_style_kwargs:
            arrow_style_kwargs['direction'] = 'left' if style.reverse_direction else 'right'
        return safe_init(generator_class, arrow_style_kwargs).generate_svg(unique_id=True)
