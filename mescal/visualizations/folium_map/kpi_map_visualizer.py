import folium
import branca.colormap as cm

from mescal.study_manager import StudyManager
from mescal.kpis.kpi_base import KPI
from mescal.kpis.kpi_collection import KPICollection
from mescal.units import Units
from mescal.utils.logging import get_logger
from mescal.utils.dict_combinations import dict_combination_iterator

logger = get_logger(__name__)


class AreaKPIMapVisualizer:
    def __init__(self, study_manager: StudyManager):
        self.study_manager = study_manager

    def get_feature_groups(self, kpi_collection: KPICollection, colormap: cm.ColorMap) -> list[folium.FeatureGroup]:
        feature_groups = []
        for kpi_group in self._get_kpi_groups(kpi_collection):
            group_name = self._get_feature_group_name(kpi_group)
            fg = folium.FeatureGroup(name=group_name, overlay=False, show=False)
            for kpi in kpi_group:
                try:
                    self._add_kpi_to_feature_group(kpi, fg, colormap)
                except Exception as e:
                    logger.warning(f'Exception while trying to add KPI {kpi.name} to FeatureGroup {group_name}: {e}')
            feature_groups.append(fg)
        return feature_groups

    def _get_kpi_groups(self, kpi_collection: KPICollection) -> list[KPICollection]:
        attribute_sets = kpi_collection.get_all_kpi_attributes_and_value_sets()
        relevant_attribute_sets = {
            k: v
            for k, v in attribute_sets.items()
            if k not in ['kpi_name', 'object_name', 'column_subset']  # TODO: could be something to set in __init__
        }

        groups: list[KPICollection] = []
        for group_kwargs in dict_combination_iterator(relevant_attribute_sets):
            g = kpi_collection.get_filtered_kpi_collection_by_attributes(**group_kwargs)
            if not g.empty:
                groups.append(g)
        return groups

    def _get_feature_group_name(self, kpi_group: KPICollection) -> str:
        values = [v for v in kpi_group.get_in_common_kpi_attributes().values() if v is not None]
        # TODO: filter to data_set_name, flag, aggregation, value_operation, unit,
        return ' '.join(values)

    def _add_kpi_to_feature_group(self, kpi: KPI, feature_group: folium.FeatureGroup, colormap: cm.ColorMap):
        style = self._get_style_kwargs(kpi, colormap)
        highlight = self._get_highlight_kwargs(kpi, colormap)
        geojson = self._get_geojson(kpi)
        folium.GeoJson(
            geojson,
            style_function=lambda x, s=dict(style): s,
            highlight_function=lambda x, h=dict(highlight): h,
            tooltip=folium.GeoJsonTooltip(fields=['tooltip'], aliases=[''], sticky=True)
        ).add_to(feature_group)

        icon_text = self._get_icon_text(kpi)
        if icon_text:
            icon_loc = self._get_icon_projection_point(kpi)
            text_color = self._get_contrast_color(style['fillColor'])
            icon_html = f'<div style="color:{text_color};">{icon_text}</div>'
            folium.Marker(location=icon_loc, icon=folium.DivIcon(html=icon_html)).add_to(feature_group)

    def _get_style_kwargs(self, kpi: KPI, colormap: cm.ColorMap) -> dict:
        return {
            'fillColor': colormap(kpi.value),
            'color': 'white',
            'weight': 1,
            'fillOpacity': 1
        }

    def _get_highlight_kwargs(self, kpi: KPI, colormap: cm.ColorMap) -> dict:
        highlight = self._get_style_kwargs(kpi, colormap)
        highlight['weight'] = 3
        highlight['fillOpacity'] = 0.8
        return highlight

    def _get_tooltip(self, kpi: KPI) -> str:
        return str(kpi.quantity)

    def _get_geojson(self, kpi: KPI) -> dict:
        info = kpi.get_attributed_object_info_from_model()
        tooltip = self._get_tooltip(kpi)
        return {
            "type": "Feature",
            "geometry": info.geometry.__geo_interface__,
            "properties": {"tooltip": tooltip}
        }

    def _get_contrast_color(self, color: str) -> str:
        return '#194D6C' if self._is_dark(color) else '#ffffff'

    def _get_icon_text(self, kpi: KPI) -> str:
        icon_text = Units.get_pretty_text_for_quantity(kpi.quantity)
        return f'{icon_text}'

    def _get_icon_projection_point(self, kpi: KPI) -> tuple[float, float]:
        info = kpi.get_attributed_object_info_from_model()
        return info['projection_point'].coords[0][::-1]

    @staticmethod
    def _is_dark(color: str) -> bool:
        r, g, b = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
        return (0.299 * r + 0.587 * g + 0.114 * b) < 186
