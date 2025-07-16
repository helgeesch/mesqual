from typing import Callable, Any

import folium
import pandas as pd
from shapely import Point

from mescal.visualizations.deprecated_folium_modules.model_visualizer_base import StyledModelVisualizerBase
from mescal.visualizations.deprecated_styling.icon_styling import IconMap


class NodeModelVisualizer(StyledModelVisualizerBase):
    """Visualizer for node model DataFrames."""

    LOCATION_COLUMN = 'location'

    def __init__(
            self,
            location_column: str | None = None,
            iconmap: IconMap | None = None,
            color_column: str | None = None,
            colormap: Callable[[Any], str] | str = '#D9D9D9',
            width_column: str | None = None,
            widthmap: Callable[[Any], float] | float = 8.0,
            opacity_column: str | None = None,
            opacitymap: Callable[[Any], float] | float = 1.0
    ):
        super().__init__(color_column, colormap, width_column, widthmap, opacity_column, opacitymap)
        self.location_column = location_column or self.LOCATION_COLUMN
        self.iconmap = iconmap

    def _add_model_object_to_feature_group(self, object_id: Any, object_data: pd.Series,
                                           feature_group: folium.FeatureGroup):
        if self.location_column not in object_data or pd.isna(object_data[self.location_column]):
            return

        location = object_data[self.location_column]
        if isinstance(location, Point):
            coords = (location.y, location.x)
        else:
            return

        tooltip = self._get_tooltip_html(object_id, object_data)

        # Check if using IconMap
        if self.iconmap is not None:
            div_icon = self.iconmap()  # IconMap returns folium.DivIcon directly

            folium.Marker(
                location=coords,
                icon=div_icon,
                tooltip=tooltip
            ).add_to(feature_group)
        else:
            # Default circle marker
            folium.CircleMarker(
                location=coords,
                radius=self._get_width(object_data),
                color='white',
                fillColor=self._get_color(object_data),
                fillOpacity=self._get_opacity(object_data),
                weight=1,
                tooltip=tooltip
            ).add_to(feature_group)
