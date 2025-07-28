from dataclasses import dataclass
from typing import Type, Any

import folium
from shapely import Polygon, MultiPolygon

from mescal.visualizations.folium_viz_system.map_data_item import MapDataItem
from mescal.visualizations.folium_viz_system.base_viz_system import (
    ResolvedFeature,
    FeatureResolver,
    PropertyMapper,
    FoliumObjectGenerator,
)


@dataclass
class ResolvedAreaFeature(ResolvedFeature):
    """Specialized style container for area visualizations."""

    @property
    def geometry(self) -> Polygon | MultiPolygon:
        return self.get('geometry')

    @property
    def fill_color(self) -> str:
        return self.get('fill_color')

    @property
    def border_color(self) -> str:
        return self.get('border_color')

    @property
    def border_width(self) -> float:
        return self.get('border_width')

    @property
    def fill_opacity(self) -> float:
        return min(self.get('fill_opacity'), 1.0)

    @property
    def highlight_border_width(self) -> float:
        highlight_border_width = self.get('highlight_border_width')
        if highlight_border_width is None:
            return self.border_width
        return highlight_border_width

    @property
    def highlight_fill_opacity(self) -> float:
        return min(self.get('highlight_fill_opacity'), 1.0)


class AreaFeatureResolver(FeatureResolver[ResolvedAreaFeature]):
    def __init__(
            self,
            fill_color: PropertyMapper | str = '#D2D2D2',
            border_color: PropertyMapper | str = 'white',
            border_width: PropertyMapper | float = 2.0,
            fill_opacity: PropertyMapper | float = 0.8,
            highlight_border_width: PropertyMapper | float = None,
            highlight_fill_opacity: PropertyMapper | float = 1.0,
            tooltip: PropertyMapper | str | bool = True,
            popup: PropertyMapper | folium.Popup | bool = False,
            geometry: PropertyMapper | Polygon = None,
            **property_mappers: PropertyMapper | Any,
    ):
        mappers = dict(
            fill_color=fill_color,
            border_color=border_color,
            border_width=border_width,
            fill_opacity=fill_opacity,
            highlight_border_width=highlight_border_width,
            highlight_fill_opacity=highlight_fill_opacity,
            tooltip=tooltip,
            popup=popup,
            geometry=self._explicit_or_fallback(geometry, self._default_geometry_mapper()),
            **property_mappers
        )
        super().__init__(feature_type=ResolvedAreaFeature, **mappers)


class AreaGenerator(FoliumObjectGenerator[AreaFeatureResolver]):
    """Generates folium GeoJson objects for area geometries."""

    def _feature_resolver_type(self) -> Type[AreaFeatureResolver]:
        return AreaFeatureResolver

    def generate(self, data_item: MapDataItem, feature_group: folium.FeatureGroup) -> None:
        style = self.feature_resolver.resolve_feature(data_item)

        geometry = style.geometry
        if not isinstance(geometry, (Polygon, MultiPolygon)):
            return

        style_dict = {
            'fillColor': style.fill_color,
            'color': style.border_color,
            'weight': style.border_width,
            'fillOpacity': style.fill_opacity
        }

        highlight_dict = style_dict.copy()
        highlight_dict['weight'] = style.highlight_border_width
        highlight_dict['fillOpacity'] = style.highlight_fill_opacity

        geojson_data = {
            "type": "Feature",
            "geometry": geometry.__geo_interface__,
            "properties": {"tooltip": style.tooltip}
        }

        folium.GeoJson(
            geojson_data,
            style_function=lambda x, s=style_dict: s,
            highlight_function=lambda x, h=highlight_dict: h,
            tooltip=folium.GeoJsonTooltip(fields=['tooltip'], aliases=[''], sticky=True) if style.tooltip else None,
            popup=style.popup
        ).add_to(feature_group)


if __name__ == '__main__':
    import os
    import webbrowser
    import pandas as pd
    from shapely.geometry import Polygon
    import folium

    from mescal.visualizations.value_mapping_system import (
        SegmentedContinuousColorscale,
        SegmentedContinuousOpacityMapping,
    )

    area_df = pd.DataFrame({
        'geometry': [
            Polygon([(7.0, 50.0), (7.5, 50.0), (7.5, 50.5), (7.0, 50.5)]),
            Polygon([(8.0, 50.0), (8.5, 50.0), (8.5, 50.5), (8.0, 50.5)]),
            Polygon([(9.0, 50.0), (9.5, 50.0), (9.5, 50.5), (9.0, 50.5)])
        ],
        'value': [10, 20, 30]
    }, index=['area1', 'area2', 'area3'])

    m = folium.Map(location=[50.25, 8.0], zoom_start=8, tiles='CartoDB Positron')

    color_map = SegmentedContinuousColorscale.single_segment_autoscale_factory_from_array(
        values=area_df['value'].values,
        colorscale=['green', 'blue', 'red']
    )

    opacity_map = SegmentedContinuousOpacityMapping.single_segment_autoscale_factory_from_array(
        values=area_df['value'].values,
        output_range=(0.4, 0.9)
    )

    area_generator = AreaGenerator(
        feature_resolver=AreaFeatureResolver(
            fill_color=PropertyMapper.from_item_attr('value', color_map),
            fill_opacity=PropertyMapper.from_item_attr('value', opacity_map),
            border_color='#ABABAB',
            border_width=10,
            tooltip=True,
        )
    )

    fg = folium.FeatureGroup(name='Test Areas')
    area_generator.generate_objects_for_model_df(area_df, fg)
    fg.add_to(m)

    m.add_child(folium.LayerControl())
    m.save('_tmp/map.html')
    webbrowser.open('file://' + os.path.abspath('_tmp/map.html'))
