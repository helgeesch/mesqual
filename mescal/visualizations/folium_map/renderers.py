from abc import ABC, abstractmethod
from typing import Any, Union, List, Dict, Tuple, Callable

import folium
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString

from mescal.enums import VisualizationTypeEnum


class MapFeatureRenderer(ABC):
    @abstractmethod
    def render(
            self,
            feature_data: Union[pd.DataFrame, gpd.GeoDataFrame],
            style: dict,
            tooltip: str = None
    ) -> folium.Element:
        pass

    def _ensure_geometry(
            self,
            data: Union[pd.DataFrame, gpd.GeoDataFrame],
            geometry_type: str
    ) -> gpd.GeoDataFrame:
        if isinstance(data, gpd.GeoDataFrame):
            return data

        if "geometry" in data.columns:
            return gpd.GeoDataFrame(data)

        return self._create_geometry(data, geometry_type)

    def _create_geometry(
            self,
            data: pd.DataFrame,
            geometry_type: str
    ) -> gpd.GeoDataFrame:
        if geometry_type == "point":
            geometry = [Point(xy) for xy in zip(data["lon"], data["lat"])]
        elif geometry_type == "line":
            geometry = [LineString(zip(row["lon"], row["lat"]))
                        for _, row in data.iterrows()]
        else:
            raise ValueError(f"Unsupported geometry type: {geometry_type}")

        return gpd.GeoDataFrame(data, geometry=geometry)

    def _style_function(self, style: dict) -> Callable[[Any], dict]:
        return lambda x: style


class AreaRenderer(MapFeatureRenderer):
    def render(
            self,
            feature_data: Union[pd.DataFrame, gpd.GeoDataFrame],
            style: dict,
            tooltip: str = None
    ) -> folium.Element:
        gdf = self._ensure_geometry(feature_data, "area")

        return folium.GeoJson(
            gdf.__geo_interface__,
            style_function=self._style_function(style),
            tooltip=tooltip,
            name=style.get("name", ""),
        )


class PointRenderer(MapFeatureRenderer):
    def render(
            self,
            feature_data: Union[pd.DataFrame, gpd.GeoDataFrame],
            style: dict,
            tooltip: str = None
    ) -> folium.Element:
        gdf = self._ensure_geometry(feature_data, "point")

        # Get first point (assuming single point feature)
        point = gdf.geometry.iloc[0]
        location = [point.y, point.x]  # lat, lon

        marker_style = {
            "radius": style.pop("radius", 8),
            "color": style.pop("color", "blue"),
            "fill": style.pop("fill", True),
            "fill_color": style.pop("fillColor", None),
            "fill_opacity": style.pop("fillOpacity", 0.6),
            "weight": style.pop("weight", 2),
            **style
        }

        return folium.CircleMarker(
            location=location,
            tooltip=tooltip,
            **marker_style
        )


class LineRenderer(MapFeatureRenderer):
    def render(
            self,
            feature_data: Union[pd.DataFrame, gpd.GeoDataFrame],
            style: dict,
            tooltip: str = None
    ) -> folium.Element:
        if isinstance(feature_data, pd.DataFrame) and all(isinstance(x, list) for x in feature_data["lat"]):
            # Handle pre-formatted coordinate lists
            coordinates = list(zip(feature_data["lat"].iloc[0], feature_data["lon"].iloc[0]))
        else:
            gdf = self._ensure_geometry(feature_data, "line")
            line = gdf.geometry.iloc[0]
            coordinates = [[y, x] for x, y in line.coords]  # Convert to lat, lon

        line_style = {
            "weight": style.pop("weight", 2),
            "color": style.pop("color", "blue"),
            "opacity": style.pop("opacity", 0.8),
            **style
        }

        return folium.PolyLine(
            locations=coordinates,
            tooltip=tooltip,
            **line_style
        )


class BorderRenderer(MapFeatureRenderer):
    def render(
            self,
            feature_data: Union[pd.DataFrame, gpd.GeoDataFrame],
            style: dict,
            tooltip: str = None
    ) -> folium.Element:
        gdf = self._ensure_geometry(feature_data, "point")
        point = gdf.geometry.iloc[0]
        location = [point.y, point.x]  # lat, lon

        icon_style = {
            "icon": style.pop("icon", "info"),
            "prefix": style.pop("prefix", "fa"),
            "color": style.pop("color", "white"),
            "icon_color": style.pop("iconColor", "black"),
            **style
        }

        return folium.Icon(
            location=location,
            tooltip=tooltip,
            **icon_style
        )


class RendererFactory:
    _renderers = {
        VisualizationTypeEnum.Area: AreaRenderer(),
        VisualizationTypeEnum.Point: PointRenderer(),
        VisualizationTypeEnum.Line: LineRenderer(),
        VisualizationTypeEnum.Border: BorderRenderer(),
    }

    @classmethod
    def get_renderer(cls, visualization_type: VisualizationTypeEnum) -> MapFeatureRenderer:
        if visualization_type not in cls._renderers:
            raise ValueError(f"No renderer found for visualization type: {visualization_type}")
        return cls._renderers[visualization_type]