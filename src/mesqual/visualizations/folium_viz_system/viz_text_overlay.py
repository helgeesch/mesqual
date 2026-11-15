from dataclasses import dataclass
from typing import Type, Any

from shapely import Point
import folium

from mesqual.visualizations.folium_viz_system.visualizable_data_item import VisualizableDataItem

from mesqual.visualizations.folium_viz_system.base_viz_system import (
    ResolvedFeature,
    FeatureResolver,
    PropertyMapper,
    FoliumObjectGenerator,
    _escape_text,
)


@dataclass
class ResolvedTextOverlayFeature(ResolvedFeature):
    """
    Resolved visual properties for text overlay elements.

    Container for all computed styling properties of text overlay visualizations,
    including font styling, positioning, colors, and shadow effects.
    Used by TextOverlayGenerator to create folium markers with styled text content.
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

    @property
    def text_is_html(self) -> bool:
        return bool(self.get('text_is_html'))


class TextOverlayFeatureResolver(FeatureResolver[ResolvedTextOverlayFeature]):
    """
    Resolves visual properties for text overlay elements.

    Specialized feature resolver for text overlay visualizations that handles text
    content, font styling, positioning, and visual effects. Commonly used for
    adding data labels, value displays, or annotations to map elements.

    Args:
        text_color: Text color (static value or PropertyMapper)
        font_size: Font size with units like '10pt' (static value or PropertyMapper)
        font_weight: Font weight like 'bold', 'normal' (static value or PropertyMapper)
        background_color: Background color for text (static value or PropertyMapper)
        shadow_size: Text shadow size like '0.5px' (static value or PropertyMapper)
        shadow_color: Text shadow color (static value or PropertyMapper)
        text_print_content: Text content to display (True for auto-generated).
            The text is HTML-escaped, so `<`, `>` and `&` show as characters.
        text_is_html: True to write `text_print_content` into the label as
            HTML, without escaping, for example for a label with line breaks
            (`<br>`) or markup.
        tooltip: Tooltip content, see `FeatureResolver`.
        popup: Popup content, see `FeatureResolver`.
        location: Point of the label. Default: the first `Point` of the item
            attributes `location`, `projection_point`, `centroid` and `midpoint`;
            else a point from the attributes `lat` and `lon` or `latitude` and
            `longitude` if both values are instances of Python `int` or `float`
            (`numpy.float64` is one, `numpy.int64` is not); else from the
            attribute `geometry`: the point, the
            representative point of a polygon, or the midpoint of a line. An
            item without location gets no label.
        **property_mappers: Additional custom property mappings

    Examples:
        Static styling:

        >>> resolver = TextOverlayFeatureResolver(text_color='#000000', font_size='12pt', font_weight='bold')

        Color and text from the KPI value:

        >>> from mesqual.visualizations.folium_viz_system import PropertyMapper
        >>> resolver = TextOverlayFeatureResolver(
        ...     text_color=PropertyMapper.from_kpi_value(lambda v: '#FF0000' if v > 100 else '#000000'),
        ...     text_print_content=PropertyMapper.from_kpi_value(lambda v: f'{v:.0f} MW'),
        ... )

        Two-line labels with HTML markup:

        >>> resolver = TextOverlayFeatureResolver(
        ...     text_print_content=PropertyMapper.from_item_attr('name', lambda name: f'<b>{name}</b><br>zone'),
        ...     text_is_html=True,
        ... )
    """

    def __init__(
        self,
        text_color: PropertyMapper | str = '#3A3A3A',
        font_size: PropertyMapper | str = '10pt',
        font_weight: PropertyMapper | str = 'bold',
        background_color: PropertyMapper | str = None,
        shadow_size: PropertyMapper | str = '0.5px',
        shadow_color: PropertyMapper | str | None = '#F2F2F2',
        text_print_content: PropertyMapper | str | bool = True,
        text_is_html: PropertyMapper | bool = False,
        tooltip: PropertyMapper | str | bool = True,
        popup: PropertyMapper | folium.Popup | bool = False,
        location: PropertyMapper | Point = None,
        offset: PropertyMapper | tuple[float, float] = (0, 0),
        azimuth_angle: PropertyMapper | float = 90,
        **property_mappers: PropertyMapper | Any,
    ):
        mappers = dict(
            text_color=text_color,
            font_size=font_size,
            font_weight=font_weight,
            background_color=background_color,
            shadow_size=shadow_size,
            shadow_color=shadow_color,
            text_print_content=text_print_content,
            text_is_html=text_is_html,
            tooltip=tooltip,
            popup=popup,
            location=self._explicit_or_fallback(location, self._default_location_mapper()),
            offset=offset,
            azimuth_angle=azimuth_angle,
            **property_mappers,
        )
        super().__init__(feature_type=ResolvedTextOverlayFeature, **mappers)


class TextOverlayGenerator(FoliumObjectGenerator[TextOverlayFeatureResolver]):
    """Generates text labels for map data items.

    Creates one folium Marker per data item with a styled HTML text label at the
    location of the item. Typical uses are KPI values on areas or lines, data
    labels and categorical labels.

    Examples:
        >>> from mesqual.kpis import Aggregations, FlagAggKPIBuilder
        >>> from mesqual.visualizations.folium_viz_system import PropertyMapper
        >>> dataset.add_kpis_from_definitions(
        ...     FlagAggKPIBuilder().for_flag('Country.Results.price').with_aggregation(Aggregations.Mean).build()
        ... )
        >>> prices = dataset.kpi_collection.filter(flag='Country.Results.price')
        >>> import folium
        >>> from mesqual.units import Units
        >>> generator = TextOverlayGenerator(
        ...     TextOverlayFeatureResolver(
        ...         text_print_content=PropertyMapper(
        ...             lambda item: Units.get_pretty_text_for_quantity(item.kpi.quantity, decimals=0, include_unit=False)
        ...         ),
        ...         font_size='10pt',
        ...         font_weight='bold',
        ...     )
        ... )
        >>> fg = generator.generate_objects_for_kpi_collection(prices, folium.FeatureGroup('Price labels'))

        Areas and labels in one feature group:

        >>> from mesqual.visualizations.folium_viz_system import AreaGenerator
        >>> fg = folium.FeatureGroup('Countries')
        >>> fg = AreaGenerator().generate_objects_for_model_df(dataset.fetch('Country.Model'), fg)
        >>> fg = TextOverlayGenerator().generate_objects_for_model_df(dataset.fetch('Country.Model'), fg)
    """

    def _feature_resolver_type(self) -> Type[TextOverlayFeatureResolver]:
        return TextOverlayFeatureResolver

    def generate(self, data_item: VisualizableDataItem, feature_group: folium.FeatureGroup) -> None:
        """
        Generate and add a folium Marker with styled text overlay.

        Args:
            data_item: Data item containing point location and text content
            feature_group: Folium feature group to add the text marker to
        """
        style = self.feature_resolver.resolve_feature(data_item)
        if not isinstance(style.location, Point):
            return

        if not style.text_print_content:
            return

        text_content = style.text_print_content if style.text_is_html else _escape_text(style.text_print_content)
        text_color = style.text_color
        font_size = style.font_size
        font_weight = style.font_weight
        shadow_size = style.shadow_size
        shadow_color = style.shadow_color

        angle = (
            float(style.azimuth_angle) - 90
        )  # Folium counts clockwise from right-pointing direction; normal convention is CCW
        x_offset, y_offset = style.offset

        icon_html = f"""
            <div style="
                position: absolute;
                left: 50%;
                top: 50%;
                transform: translate(-50%, -50%) translate({x_offset}px, {-y_offset}px) rotate({angle}deg);
                text-align: center;
                font-size: {font_size};
                font-weight: {font_weight};
                color: {text_color};
                white-space: nowrap;
                text-shadow:
                   -{shadow_size} -{shadow_size} 0 {shadow_color},  
                    {shadow_size} -{shadow_size} 0 {shadow_color},
                   -{shadow_size}  {shadow_size} 0 {shadow_color},
                    {shadow_size}  {shadow_size} 0 {shadow_color};
            ">
                {text_content}
            </div>
        """

        folium.Marker(
            location=(style.location.y, style.location.x),
            icon=folium.DivIcon(html=icon_html),
            tooltip=style.tooltip,
            popup=style.popup,
        ).add_to(feature_group)

    def _get_contrasting_color(self, surface_color: str) -> str:
        """Get contrasting text color for a surface color."""
        if self._is_dark(surface_color):
            return '#F2F2F2'
        return '#3A3A3A'

    def _get_shadow_color(self, text_color: str) -> str:
        """Get shadow color for text."""
        if text_color == '#F2F2F2':
            return '#3A3A3A'
        return '#F2F2F2'

    @staticmethod
    def _is_dark(color: str) -> bool:
        """Check if a color is dark."""
        if not color.startswith('#'):
            return False
        try:
            r, g, b = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
            return (0.299 * r + 0.587 * g + 0.114 * b) < 160
        except (ValueError, IndexError):
            return False
