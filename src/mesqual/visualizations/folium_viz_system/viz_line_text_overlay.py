from dataclasses import dataclass
from typing import Type, Any

import folium
from folium.plugins import PolyLineTextPath
from shapely import LineString

from mesqual.visualizations.folium_viz_system.visualizable_data_item import VisualizableDataItem
from mesqual.visualizations.folium_viz_system.base_viz_system import (
    ResolvedFeature,
    FeatureResolver,
    PropertyMapper,
    FoliumObjectGenerator,
)


@dataclass
class ResolvedLineTextOverlayFeature(ResolvedFeature):
    """Resolved visual properties of a text along a line.

    Used by `LineTextOverlayGenerator`. The properties give the resolved
    values with the defaults of `LineTextOverlayFeatureResolver`; without a
    value, `text_offset` is half the font size.
    """

    @property
    def geometry(self) -> LineString:
        return self.get('geometry')

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


class LineTextOverlayFeatureResolver(FeatureResolver[ResolvedLineTextOverlayFeature]):
    """Resolves the visual properties of a text that follows a line, for example a flow value along a line.

    Args:
        text_offset: Distance of the text from the line in pixels. Default: half the font size.
        text_orientation: Rotation of the text in degrees.
        text_repeat: Repeat the text along the whole line.
        text_center: Place the text at the middle of the line.
        text_below: Place the text below the line.
        font_weight: CSS font weight.
        font_size: Font size in pixels.
        font_color: Font color.
        reverse_path_direction: Run the text from the end to the start of the line.
        text_print_content: The text. `True` gives `get_text_representation()`
            of the item. A falsy value (`False`, an empty text, `0`) draws no text.
        tooltip: Tooltip content, see `FeatureResolver`.
        popup: Popup content, see `FeatureResolver`.
        geometry: LineString of the text path. Default: the first LineString or
            MultiLineString of the item attributes `geometry` and `line_string`.
            `LineTextOverlayGenerator` draws a LineString only.
        **property_mappers: Additional custom property mappings.

    Each argument takes a static value or a `PropertyMapper`.
    """

    def __init__(
        self,
        text_offset: PropertyMapper | float = None,
        text_orientation: PropertyMapper | int = 0,
        text_repeat: PropertyMapper | bool = False,
        text_center: PropertyMapper | bool = False,
        text_below: PropertyMapper | bool = False,
        font_weight: PropertyMapper | str = 'bold',
        font_size: PropertyMapper | int = 12,
        font_color: PropertyMapper | str = '#000000',
        reverse_path_direction: PropertyMapper | bool = False,
        text_print_content: PropertyMapper | str = True,
        tooltip: PropertyMapper | str | bool = True,
        popup: PropertyMapper | folium.Popup | bool = False,
        geometry: PropertyMapper | LineString = None,
        **property_mappers: PropertyMapper | Any,
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
            text_print_content=text_print_content,
            tooltip=tooltip,
            popup=popup,
            geometry=self._explicit_or_fallback(geometry, self._default_line_string_mapper()),
            **property_mappers,
        )
        super().__init__(feature_type=ResolvedLineTextOverlayFeature, **mappers)


class LineTextOverlayGenerator(FoliumObjectGenerator[LineTextOverlayFeatureResolver]):
    """Draws a text along the line of each item, with folium `PolyLineTextPath` on an invisible line.

    An item without LineString geometry or without text gets no text.

    Examples:
        A text along each border line:

        >>> import folium
        >>> generator = LineTextOverlayGenerator(
        ...     LineTextOverlayFeatureResolver(
        ...         geometry=PropertyMapper.from_item_attr('geo_line_string'),
        ...         text_print_content=PropertyMapper.from_item_attr('area_to', lambda area: f'to {area}'),
        ...     )
        ... )
        >>> folium_map = folium.Map(location=[48, 9], zoom_start=6)
        >>> fg = generator.generate_objects_for_model_df(dataset.fetch('CountryBorder.Model'), folium.FeatureGroup('Borders'))
        >>> fg = fg.add_to(folium_map)
        >>> html = folium_map.get_root().render()
        >>> html.count('.setText('), '"to C2"' in html
        (2, True)
    """

    def _feature_resolver_type(self) -> Type[LineTextOverlayFeatureResolver]:
        return LineTextOverlayFeatureResolver

    def generate(self, data_item: VisualizableDataItem, feature_group: folium.FeatureGroup) -> None:
        style = self.feature_resolver.resolve_feature(data_item)
        if not isinstance(style.geometry, LineString):
            return

        if not style.text_print_content:
            return

        coordinates = [(lat, lon) for lon, lat in style.geometry.coords]

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
            text=style.text_print_content,
            repeat=style.text_repeat,
            center=style.text_center,
            below=style.text_below,
            orientation=style.text_orientation,
            offset=style.text_offset,
            attributes={
                'font-weight': style.font_weight,
                'font-size': str(style.font_size),
                'fill': style.font_color,
            },
        ).add_to(feature_group)
