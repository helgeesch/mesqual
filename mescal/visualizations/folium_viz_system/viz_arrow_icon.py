from dataclasses import dataclass
from typing import Type, Literal, Any
from enum import Enum
import base64

from shapely import Point
import folium

from mescal.visualizations.folium_viz_system.map_data_item import MapDataItem
from mescal.visualizations.folium_viz_system.base_viz_system import (
    ResolvedStyle,
    StyleResolver,
    StyleMapper, DataItemStyleMapper,
    FoliumObjectGenerator,
)
from captain_arro import get_generator_for_arrow_type, ArrowTypeEnum


@dataclass
class ResolvedArrowIconStyle(ResolvedStyle):
    @property
    def location(self) -> Point:
        return self.get('location')

    @property
    def rotation_angle(self) -> float:
        return self.get('rotation_angle')

    @property
    def arrow_type(self) -> ArrowTypeEnum:
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


class ArrowIconStyleResolver(StyleResolver[ResolvedArrowIconStyle]):
    def __init__(
            self,
            arrow_type: StyleMapper | ArrowTypeEnum = ArrowTypeEnum.MOVING_FLOW_ARROW,
            color: StyleMapper | str = '#2563eb',
            stroke_width: StyleMapper | int = 8,
            width: StyleMapper | int = 60,
            height: StyleMapper | int = 60,
            speed_in_px_per_second: StyleMapper | float | None = 20.0,
            speed_in_duration_seconds: StyleMapper | float | None = None,
            num_arrows: StyleMapper | int = 3,
            opacity: StyleMapper | float = 0.8,
            reverse_direction: StyleMapper | bool = False,
            location: StyleMapper | Point = None,
            rotation_angle: StyleMapper | float = None,
            **style_mappers: StyleMapper | Any,
    ):
        mappers = dict(
            arrow_type=arrow_type,
            color=color,
            stroke_width=stroke_width,
            width=width,
            height=height,
            speed_in_px_per_second=speed_in_px_per_second,
            speed_in_duration_seconds=speed_in_duration_seconds,
            num_arrows=num_arrows,
            opacity=opacity,
            reverse_direction=reverse_direction,
            location=self._explicit_or_fallback(location, self._default_location_mapper()),
            rotation_angle=self._explicit_or_fallback(rotation_angle, self._default_rotation_angle_mapper()),
            **style_mappers
        )
        super().__init__(style_type=ResolvedArrowIconStyle, **mappers)

    @staticmethod
    def _default_rotation_angle_mapper() -> DataItemStyleMapper:

        def get_rotation_angle(data_item: MapDataItem) -> float | None:
            for k in ['rotation_angle', 'projection_point']:
                if data_item.object_has_attribute(k):
                    angle = data_item.get_object_attribute(k)
                    return angle
            return None

        return DataItemStyleMapper(get_rotation_angle, float)


class ArrowIconGenerator(FoliumObjectGenerator[ArrowIconStyleResolver]):
    def _style_resolver_type(self) -> Type[ArrowIconStyleResolver]:
        return ArrowIconStyleResolver

    def generate(self, data_item: MapDataItem, feature_group: folium.FeatureGroup) -> None:
        style = self.style_resolver.resolve_style(data_item)
        if style.location is None:
            return

        tooltip = self.tooltip_generator.generate_tooltip(data_item)
        popup = self.popup_generator.generate_popup(data_item) if self.popup_generator else None

        svg_content = self._generate_arrow_svg(style, data_item)
        encoded_svg = base64.b64encode(svg_content.encode()).decode()

        # Folium counts clockwise from right-pointing direction; normal convention is CCW
        rotation_angle = - style.rotation_angle

        icon_html = f'''
            <div style="
                position: absolute;
                left: 50%;
                top: 50%;
                transform: translate(-50%, -50%) rotate({rotation_angle}deg);
                opacity: {style.opacity};
            ">
                <img src="data:image/svg+xml;base64,{encoded_svg}" 
                     width="{style.width}" 
                     height="{style.height}">
            </div>
        '''

        marker_kwargs = {
            'location': (style.location.y, style.location.x),
            'icon': folium.DivIcon(
                html=icon_html,
                icon_size=(style.width, style.height),
                icon_anchor=(style.width // 2, style.height // 2)
            ),
            'tooltip': tooltip
        }

        if popup:
            marker_kwargs['popup'] = folium.Popup(popup, max_width=300)

        folium.Marker(**marker_kwargs).add_to(feature_group)

    def _generate_arrow_svg(self, style: ResolvedArrowIconStyle, data_item: MapDataItem) -> str:
        from inspect import signature

        def safe_init(cls: type, kwargs: dict):
            init_params = signature(cls.__init__).parameters
            accepted_keys = {
                k for k in init_params
                if k != 'self' and init_params[k].kind in (
                init_params[k].POSITIONAL_OR_KEYWORD, init_params[k].KEYWORD_ONLY)
            }
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_keys}
            return cls(**filtered_kwargs)

        arrow_type = style.arrow_type
        generator_class = get_generator_for_arrow_type(arrow_type)
        arrow_style_kwargs = style.to_dict()
        if 'direction' not in arrow_style_kwargs:
            arrow_style_kwargs['direction'] = 'left' if style.reverse_direction else 'right'
        return safe_init(generator_class, arrow_style_kwargs).generate_svg(unique_id=True)


if __name__ == '__main__':
    import os
    import webbrowser
    import pandas as pd
    from shapely.geometry import Point
    from mescal.visualizations.value_mapping_system import (
        SegmentedContinuousColorscale,
        SegmentedContinuousOpacityMapping,
    )

    border_model_df = pd.DataFrame({
        'projection_point': [Point(7.45, 49.15), Point(6.94, 52.22), Point(6.34, 50.38), Point(12.31, 50.25)],
        'projection_angle': [-110.0, 170.0, 180.0, -15],
        'border_flow': [150.0, -75.0, 200.0, -100],
        'bing_bong': [5.0, -2.0, 8.0, -10],
    }, index=['DE-FR', 'DE-NL', 'DE-BE', 'DE-CZ'])

    m = folium.Map(location=[50.0, 9.0], zoom_start=6, tiles='Cartodb Positron')

    flow_color_mapping = SegmentedContinuousColorscale.single_segment_autoscale_factory_from_array(
        values=border_model_df['border_flow'].abs().values,
        colorscale=['red', 'blue'],
    )

    size_mapping = SegmentedContinuousOpacityMapping.single_segment_autoscale_factory_from_array(
        values=border_model_df['border_flow'].abs().values,
        output_range=(40, 80)
    )

    arrow_generator = ArrowIconGenerator(
        style_resolver=ArrowIconStyleResolver(
            arrow_type=StyleMapper.for_attribute('border_flow', lambda x: ArrowTypeEnum.SPOTLIGHT_FLOW_ARROW if x > 0 else ArrowTypeEnum.MOVING_FLOW_ARROW),
            color=StyleMapper.for_attribute('border_flow', lambda x: flow_color_mapping(abs(x))),
            stroke_width=4,
            rotation_angle=StyleMapper.for_attribute('projection_angle', lambda x: x),
            reverse_direction=StyleMapper.for_attribute('border_flow', lambda x: x < 0),
            speed_in_px_per_second=None,
            speed_in_duration_seconds=2,
            width=StyleMapper.for_attribute('border_flow', lambda x: size_mapping(abs(x))),
            height=StyleMapper.for_attribute('border_flow', lambda x: size_mapping(abs(x))/2),
        )
    )

    fg = folium.FeatureGroup(name='Border Flows')

    arrow_generator.generate_objects_for_model_df(border_model_df, fg)

    fg.add_to(m)
    m.add_child(folium.LayerControl())
    m.save('_tmp/map.html')
    webbrowser.open('file://' + os.path.abspath('_tmp/map.html'))
