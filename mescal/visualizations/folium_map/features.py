from dataclasses import dataclass, field
from typing import Optional

import folium
from folium import FeatureGroup
import pandas as pd

from mescal.kpis.kpi_base import KPI
from mescal.enums import VisualizationTypeEnum
from mescal.visualizations.folium_map.legends import LegendFactory, MescalMapLegend
from mescal.visualizations.folium_map.renderers import RendererFactory


@dataclass
class MescalKPIFeature:
    kpi: KPI
    style_config: dict = field(default_factory=dict)
    model_data: pd.DataFrame = None
    rendered_object: Optional[folium.Element] = None


class MescalMapKPIFeatureGroup:
    def __init__(self, name: str, visualization_type: VisualizationTypeEnum):
        self.name = name
        self.visualization_type = visualization_type
        self.features: list[MescalKPIFeature] = []
        self._legend = None
        self._renderer = RendererFactory.get_renderer(visualization_type)

    def apply_style_to_all(self, style_config: dict):
        for feature in self.features:
            feature.style_config.update(style_config)

    @property
    def values(self) -> list[float]:
        return [feature.kpi.value for feature in self.features]

    @property
    def legend(self) -> MescalMapLegend:
        if self._legend is None:
            self._legend = LegendFactory.create_legend(
                self.name,
                self.visualization_type,
                self.values,
                is_discrete=self._should_use_discrete_legend()
            )
        return self._legend

    def add_feature(self, feature: MescalKPIFeature):
        self.features.append(feature)
        self._legend = None  # Reset legend when new features are added

    def _should_use_discrete_legend(self) -> bool:
        unique_values = len(set(self.values))
        return unique_values <= 5 or self.visualization_type == VisualizationTypeEnum.Border

    def render_features(self):
        self.folium_group = FeatureGroup(name=self.name)

        for feature in self.features:
            style = self.legend.get_style_for_value(feature.kpi.value)
            style.update(feature.style_config)

            tooltip = self.get_tooltip_html(feature)

            rendered = self._renderer.render(
                feature.model_data,
                style,
                tooltip
            )
            rendered.add_to(self.folium_group)
            feature.rendered_object = rendered

    def get_tooltip_html(self, feature: MescalKPIFeature) -> str:
        from mescal.utils.pretty_scaling import get_pretty_order_of_mag

        magnitude = get_pretty_order_of_mag(self.values)
        scaled_value = feature.kpi.value / 10 ** magnitude
        unit = feature.kpi.unit

        return f"""
            <div style="font-family: Arial">
                <h4>{feature.kpi.name}</h4>
                <p>Value: {scaled_value:.2f} × 10<sup>{magnitude}</sup> {unit}</p>
                <p>Dataset: {feature.kpi._data_set.name}</p>
            </div>
        """

    def add_to_map(self, map_obj: folium.Map):
        self.render_features()
        self.folium_group.add_to(map_obj)
