from typing import Callable, Any

import folium
import pandas as pd
from shapely import LineString, Point

from mescal.visualizations.folium_modules_deprecated.model_visualizer_base import StyledModelVisualizerBase
from mescal.visualizations.styling.icon_styling import BasicArrowIconMap


class AreaBorderModelVisualizer(StyledModelVisualizerBase):
    """Visualizer for area border model DataFrames."""

    PROJECTION_POINT_COLUMN = 'projection_point'
    PROJECTION_ANGLE_COLUMN = 'projection_angle'
    QUERY_FOR_GEO_LINE_STRING_VISUALIZATION = 'is_physical == False'
    GEO_LINE_STRING_COLUMN = 'geo_line_string'
    NAME_IS_ALPHABETICALLY_SORTED_COLUMN = 'name_is_alphabetically_sorted'

    def __init__(
            self,
            projection_point_column: str | None = None,
            projection_angle_column: str | None = None,
            query_for_geo_line_string_visualization: str | None = None,
            geo_line_string_column: str | None = None,
            arrow_icon_map: Callable[[Any], folium.DivIcon] = None,
            line_color_column: str | None = None,
            line_colormap: Callable[[Any], str] | str = '#FF0000',
            line_width_column: str | None = None,
            line_widthmap: Callable[[Any], float] | float = 3.0,
            line_opacity_column: str | None = None,
            line_opacitymap: Callable[[Any], float] | float = 1.0
    ):
        super().__init__(line_color_column, line_colormap, line_width_column, line_widthmap, line_opacity_column, line_opacitymap)
        self.projection_point_column = projection_point_column or self.PROJECTION_POINT_COLUMN
        self.projection_angle_column = projection_angle_column or self.PROJECTION_ANGLE_COLUMN
        self.query_for_geo_line_string_visualization = query_for_geo_line_string_visualization or self.QUERY_FOR_GEO_LINE_STRING_VISUALIZATION
        self.geo_line_string_column = geo_line_string_column or self.GEO_LINE_STRING_COLUMN
        self.arrow_icon_map = arrow_icon_map

    def _add_model_object_to_feature_group(
            self,
            object_id: Any,
            object_data: pd.Series,
            feature_group: folium.FeatureGroup
    ):
        self._add_connection_line_if_needed(object_data, feature_group)
        self._add_directional_arrow_icon(object_id, object_data, feature_group)

    def _add_connection_line_if_needed(self, object_data: pd.Series, feature_group: folium.FeatureGroup):
        object_satisfies_visualization_query = pd.eval(
            self.query_for_geo_line_string_visualization,
            local_dict=object_data.to_dict(),
            engine='python'
        )

        if (
                object_satisfies_visualization_query and
                self.geo_line_string_column in object_data and
                pd.notna(object_data[self.geo_line_string_column])
        ):

            line_geom = object_data[self.geo_line_string_column]
            if isinstance(line_geom, LineString):
                coordinates = [(lat, lon) for lon, lat in line_geom.coords]

                folium.PolyLine(
                    locations=coordinates,
                    color=self.colormap(object_data[self.color_column] if self.color_column else None),
                    weight=self.widthmap(object_data[self.width_column] if self.width_column else None),
                    opacity=self.opacitymap(object_data[self.opacity_column] if self.opacity_column else None),
                    dashArray='5, 5'
                ).add_to(feature_group)

    def _add_directional_arrow_icon(self, object_id: Any, object_data: pd.Series, feature_group: folium.FeatureGroup):
        if (
                self.projection_point_column not in object_data or
                pd.isna(object_data[self.projection_point_column])
        ):
            return

        projection_point = object_data[self.projection_point_column]
        if not isinstance(projection_point, Point):
            return

        coords = (projection_point.y, projection_point.x)
        angle = object_data.get(self.projection_angle_column, 0)
        tooltip = self._get_tooltip_html(object_id, object_data)

        # Create arrow using ArrowIconMap
        div_icon = self.arrow_icon_map(angle)

        folium.Marker(
            location=coords,
            icon=div_icon,
            tooltip=tooltip
        ).add_to(feature_group)
