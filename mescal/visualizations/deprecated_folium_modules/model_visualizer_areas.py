from typing import Callable, Any

import folium
import pandas as pd
from shapely import Polygon, MultiPolygon, Point

from mescal.visualizations.deprecated_folium_modules.model_visualizer_base import StyledModelVisualizerBase


class AreaModelVisualizer(StyledModelVisualizerBase):
    """Visualizer for area model DataFrames."""

    GEOMETRY_COLUMN = 'geometry'
    PROJECTION_POINT_COLUMN = 'projection_point'

    def __init__(
            self,
            geometry_column: str | None = None,
            projection_point_column: str | None = None,
            show_area_names: bool = True,
            color_column: str | None = None,
            colormap: Callable[[Any], str] | str = '#D9D9D9',
            width_column: str | None = None,
            widthmap: Callable[[Any], float] | float = 2.0,
            opacity_column: str | None = None,
            opacitymap: Callable[[Any], float] | float = 0.7
    ):
        super().__init__(color_column, colormap, width_column, widthmap, opacity_column, opacitymap)
        self.geometry_column = geometry_column or self.GEOMETRY_COLUMN
        self.projection_point_column = projection_point_column or self.PROJECTION_POINT_COLUMN
        self.show_area_names = show_area_names

    def _add_model_object_to_feature_group(self, object_id: Any, object_data: pd.Series,
                                           feature_group: folium.FeatureGroup):
        if self.geometry_column not in object_data or pd.isna(object_data[self.geometry_column]):
            return

        geometry = object_data[self.geometry_column]
        if not isinstance(geometry, (Polygon, MultiPolygon)):
            return

        tooltip = self._get_tooltip_html(object_id, object_data)

        style = {
            'fillColor': self._get_color(object_data),
            'color': 'white',
            'weight': self._get_width(object_data),
            'fillOpacity': self._get_opacity(object_data)
        }

        highlight = style.copy()
        highlight['weight'] = style['weight'] * 1.5
        highlight['fillOpacity'] = min(style['fillOpacity'] * 1.5, 1.0)

        geojson_data = {
            "type": "Feature",
            "geometry": geometry.__geo_interface__,
            "properties": {"tooltip": tooltip}
        }

        folium.GeoJson(
            geojson_data,
            style_function=lambda x, s=style: s,
            highlight_function=lambda x, h=highlight: h,
            tooltip=folium.GeoJsonTooltip(fields=['tooltip'], aliases=[''], sticky=True)
        ).add_to(feature_group)

        if self.show_area_names:
            self._add_area_name_to_feature_group(object_id, object_data, feature_group)

    def _add_area_name_to_feature_group(self, object_id: Any, object_data: pd.Series,
                                        feature_group: folium.FeatureGroup):
        projection_point = None

        if (self.projection_point_column in object_data and
                pd.notna(object_data[self.projection_point_column])):
            projection_point = object_data[self.projection_point_column]
        elif (self.geometry_column in object_data and
              pd.notna(object_data[self.geometry_column])):
            geometry = object_data[self.geometry_column]
            if isinstance(geometry, (Polygon, MultiPolygon)):
                projection_point = geometry.representative_point()

        if projection_point and isinstance(projection_point, Point):
            coords = (projection_point.y, projection_point.x)

            html = f'''
                <div style="
                    position: absolute;
                    left: 50%;
                    top: 50%;
                    transform: translate(-50%, -50%);
                    text-align: center;
                    font-size: 10pt;
                    font-weight: bold;
                    color: #2C2C2C;
                    white-space: nowrap;
                    text-shadow:
                       -0.5px -0.5px 0 #F2F2F2,  
                        0.5px -0.5px 0 #F2F2F2,
                       -0.5px  0.5px 0 #F2F2F2,
                        0.5px  0.5px 0 #F2F2F2;
                ">
                    {object_id}
                </div>
            '''

            folium.Marker(
                location=coords,
                icon=folium.DivIcon(html=html),
            ).add_to(feature_group)
