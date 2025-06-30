from typing import Callable, Any

import folium
import pandas as pd
from shapely import LineString, Point

from mescal.visualizations.folium_map.model_visualizer_base import StyledModelVisualizerBase
from mescal.visualizations.styling.icon_styling import BasicArrowIconMap


class AreaBorderModelVisualizer(StyledModelVisualizerBase):
    """Visualizer for area border model DataFrames."""

    PROJECTION_POINT_COLUMN = 'projection_point'
    PROJECTION_ANGLE_COLUMN = 'projection_angle'
    IS_PHYSICAL_COLUMN = 'is_physical'
    GEO_LINE_STRING_COLUMN = 'geo_line_string'
    NAME_IS_ALPHABETICALLY_SORTED_COLUMN = 'name_is_alphabetically_sorted'

    def __init__(
            self,
            projection_point_column: str | None = None,
            projection_angle_column: str | None = None,
            is_physical_column: str | None = None,
            geo_line_string_column: str | None = None,
            name_is_alphabetically_sorted_column: str | None = None,
            show_only_alphabetically_sorted: bool = True,
            show_connection_lines: bool = True,
            arrow_size: float = 15.0,
            color_column: str | None = None,
            colormap: Callable[[Any], str] | str = '#FF0000',
            width_column: str | None = None,
            widthmap: Callable[[Any], float] | float = 3.0,
            opacity_column: str | None = None,
            opacitymap: Callable[[Any], float] | float = 1.0
    ):
        super().__init__(color_column, colormap, width_column, widthmap, opacity_column, opacitymap)
        self.projection_point_column = projection_point_column or self.PROJECTION_POINT_COLUMN
        self.projection_angle_column = projection_angle_column or self.PROJECTION_ANGLE_COLUMN
        self.is_physical_column = is_physical_column or self.IS_PHYSICAL_COLUMN
        self.geo_line_string_column = geo_line_string_column or self.GEO_LINE_STRING_COLUMN
        self.show_connection_lines = show_connection_lines
        self.arrow_size = arrow_size

    def _add_model_object_to_feature_group(self, object_id: Any, object_data: pd.Series,
                                           feature_group: folium.FeatureGroup):
        # Add connection line for non-physical borders
        if self.show_connection_lines:
            self._add_connection_line_if_needed(object_data, feature_group)

        # Add directional arrow
        self._add_directional_arrow(object_id, object_data, feature_group)

    def _add_connection_line_if_needed(self, object_data: pd.Series, feature_group: folium.FeatureGroup):
        is_physical = object_data.get(self.is_physical_column, True)

        if (not is_physical and
                self.geo_line_string_column in object_data and
                pd.notna(object_data[self.geo_line_string_column])):

            line_geom = object_data[self.geo_line_string_column]
            if isinstance(line_geom, LineString):
                coordinates = [(lat, lon) for lon, lat in line_geom.coords]

                folium.PolyLine(
                    locations=coordinates,
                    color='#666666',
                    weight=1,
                    opacity=0.7,
                    dashArray='5, 5'
                ).add_to(feature_group)

    def _add_directional_arrow(self, object_id: Any, object_data: pd.Series, feature_group: folium.FeatureGroup):
        if (self.projection_point_column not in object_data or
                pd.isna(object_data[self.projection_point_column])):
            return

        projection_point = object_data[self.projection_point_column]
        if not isinstance(projection_point, Point):
            return

        coords = (projection_point.y, projection_point.x)
        angle = object_data.get(self.projection_angle_column, 0)
        tooltip = self._get_tooltip_html(object_id, object_data)

        # Create arrow using ArrowIconMap
        arrow_color = self._get_color(object_data)
        arrow_map = BasicArrowIconMap(size=self.arrow_size, color=arrow_color)
        div_icon = arrow_map(angle)

        folium.Marker(
            location=coords,
            icon=div_icon,
            tooltip=tooltip
        ).add_to(feature_group)
