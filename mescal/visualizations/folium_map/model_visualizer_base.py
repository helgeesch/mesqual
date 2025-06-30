from abc import ABC, abstractmethod
from typing import Callable, Any
import pandas as pd
import geopandas as gpd
import folium
from tqdm import tqdm

from mescal.utils.logging import get_logger

logger = get_logger(__name__)


class ModelVisualizerBase(ABC):
    """Base class for visualizing model DataFrames on folium maps."""

    def add_model_vis_to_feature_group(
            self,
            feature_group: folium.FeatureGroup,
            model_df: pd.DataFrame | gpd.GeoDataFrame,
    ) -> folium.FeatureGroup:
        """Add all model objects to FeatureGroup."""

        for idx, row in tqdm(model_df.iterrows(), total=len(model_df), desc=f'Adding {feature_group.tile_name}'):
            try:
                self._add_model_object_to_feature_group(idx, row, feature_group)
            except Exception as e:
                logger.warning(f"Could not add {idx} to map: {e}")

        return feature_group

    @abstractmethod
    def _add_model_object_to_feature_group(
            self,
            object_id: Any,
            object_data: pd.Series,
            feature_group: folium.FeatureGroup
    ):
        """Add single model object to feature group."""
        pass

    def _get_tooltip_html(self, object_id: Any, object_data: pd.Series) -> str:
        """Generate HTML tooltip from object data."""
        html = '<table style="border-collapse: collapse;">\n'
        html += f'  <tr><td style="padding: 4px 8px;"><strong>ID</strong></td><td style="text-align: right; padding: 4px 8px;">{object_id}</td></tr>\n'

        for col, value in object_data.items():
            if pd.notna(value):
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."
                html += f'  <tr><td style="padding: 4px 8px;">{col}</td><td style="text-align: right; padding: 4px 8px;">{value_str}</td></tr>\n'

        html += '</table>'
        return html


class StyledModelVisualizerBase(ModelVisualizerBase, ABC):
    """Base class for model visualizers with styling capabilities."""

    def __init__(
            self,
            color_column: str | None = None,
            colormap: Callable[[Any], str] | str = '#D9D9D9',
            width_column: str | None = None,
            widthmap: Callable[[Any], float] | float = 3.0,
            opacity_column: str | None = None,
            opacitymap: Callable[[Any], float] | float = 1.0
    ):
        self.color_column = color_column
        self.colormap = colormap if callable(colormap) else lambda x: colormap
        self.width_column = width_column
        self.widthmap = widthmap if callable(widthmap) else lambda x: widthmap
        self.opacity_column = opacity_column
        self.opacitymap = opacitymap if callable(opacitymap) else lambda x: opacitymap

    def _get_color(self, object_data: pd.Series) -> str:
        """Get color for object based on styling configuration."""
        if self.color_column and self.color_column in object_data:
            return self.colormap(object_data[self.color_column])
        return self.colormap(None)

    def _get_width(self, object_data: pd.Series) -> float:
        """Get width for object based on styling configuration."""
        if self.width_column and self.width_column in object_data:
            return self.widthmap(object_data[self.width_column])
        return self.widthmap(None)

    def _get_opacity(self, object_data: pd.Series) -> float:
        """Get opacity for object based on styling configuration."""
        if self.opacity_column and self.opacity_column in object_data:
            return self.opacitymap(object_data[self.opacity_column])
        return self.opacitymap(None)
