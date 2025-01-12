from typing import List, Dict, Type

import pandas as pd
import folium
from folium import Map, LayerControl

from mescal.kpis.kpi_base import KPI
from mescal.enums import VisualizationTypeEnum
from mescal.visualizations.folium_map.legends import LegendFactory, MescalMapLegend
from mescal.visualizations.folium_map.features import (
    MescalKPIFeature,
    MescalMapKPIFeatureGroup,
)
from mescal.visualizations.folium_map.renderers import RendererFactory


class MescalMap:
    def __init__(
            self,
            center: List[float],
            zoom: int = 4,
            width: str = "100%",
            height: str = "100%"
    ):
        self.map = Map(
            location=center,
            zoom_start=zoom,
            width=width,
            height=height
        )
        self.feature_groups: Dict[str, MescalMapKPIFeatureGroup | folium.FeatureGroup] = {}
        self._legend_added = False

    def add_kpi(
            self,
            kpi: KPI,
            group_name: str = None,
            style_config: dict = None
    ):
        # Get visualization type from flag index
        visualization_type = kpi._data_set.flag_index.get_visualization_type(kpi._flag)

        # If no group name provided, use KPI name
        if group_name is None:
            group_name = f"{kpi._flag} {kpi._data_set.name}"

        # Get or create feature group
        feature_group = self.get_or_create_kpi_group(
            group_name,
            VisualizationTypeEnum(visualization_type)
        )

        # Create feature
        model_flag = kpi._data_set.flag_index.get_linked_model_flag(kpi._flag)
        model_data = kpi._data_set.fetch(model_flag)
        feature = MescalKPIFeature(
            kpi=kpi,
            model_data=model_data,
            style_config=style_config or {}
        )

        # Add to group
        feature_group.add_feature(feature)

    def add_kpis(self, kpis: List[KPI], group_name: str = None, style_config: dict = None):
        for kpi in kpis:
            self.add_kpi(kpi, group_name, style_config)

    def get_or_create_kpi_group(
            self,
            name: str,
            visualization_type: VisualizationTypeEnum
    ) -> MescalMapKPIFeatureGroup:
        if name not in self.feature_groups:
            self.feature_groups[name] = MescalMapKPIFeatureGroup(
                name=name,
                visualization_type=visualization_type
            )
        return self.feature_groups[name]

    def render(self):
        # Render all feature groups
        for group in self.feature_groups.values():
            group.add_to_map(self.map)

        # Add layer control
        LayerControl().add_to(self.map)

        # Add legends if not already added
        if not self._legend_added:
            self._add_legends()
            self._legend_added = True

    def _add_legends(self):
        for group in self.feature_groups.values():
            if isinstance(group, MescalMapKPIFeatureGroup):
                group.legend.create_folium_element().add_to(self.map)

    def display(self):
        self.render()
        return self.map
