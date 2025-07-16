from typing import Callable, Any

import folium
import pandas as pd
from shapely import LineString

from mescal.visualizations.deprecated_folium_modules.model_visualizer_base import StyledModelVisualizerBase


class LineModelVisualizer(StyledModelVisualizerBase):
    """Visualizer for line model DataFrames."""

    GEOMETRY_COLUMN = 'geometry'

    def __init__(
            self,
            geometry_column: str | None = None,
            color_column: str | None = None,
            colormap: Callable[[Any], str] | str = '#D9D9D9',
            width_column: str | None = None,
            widthmap: Callable[[Any], float] | float = 3.0,
            opacity_column: str | None = None,
            opacitymap: Callable[[Any], float] | float = 1.0
    ):
        super().__init__(color_column, colormap, width_column, widthmap, opacity_column, opacitymap)
        self.geometry_column = geometry_column or self.GEOMETRY_COLUMN

    def _add_model_object_to_feature_group(
            self,
            object_id: Any,
            object_data: pd.Series,
            feature_group: folium.FeatureGroup
    ):
        if self.geometry_column not in object_data or pd.isna(object_data[self.geometry_column]):
            return

        geometry = object_data[self.geometry_column]
        if not isinstance(geometry, LineString):
            return

        coordinates = [(lat, lon) for lon, lat in geometry.coords]
        tooltip = self._get_tooltip_html(object_id, object_data)

        folium.PolyLine(
            locations=coordinates,
            color=self._get_color(object_data),
            weight=self._get_width(object_data),
            opacity=self._get_opacity(object_data),
            tooltip=tooltip
        ).add_to(feature_group)
