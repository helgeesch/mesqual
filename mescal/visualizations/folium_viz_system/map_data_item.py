from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from shapely import Point, Polygon, MultiPolygon, LineString

from mescal.kpis import KPI, KPICollection


class MapDataItem(ABC):
    """Abstract interface for data items that can be visualized on maps."""

    @abstractmethod
    def get_geometry(self) -> Any:
        """Get the geometric representation of the data."""
        pass

    @abstractmethod
    def get_tooltip_data(self) -> dict:
        """Get data for tooltip display."""
        pass

    @abstractmethod
    def get_styling_value(self, column: str) -> Any:
        """Get value for styling from specified column."""
        pass

    @abstractmethod
    def get_location(self) -> tuple[float, float]:
        """Get lat/lon coordinates for point-based objects."""
        pass


class ModelDataItem(MapDataItem):
    """Map data item for model DataFrame rows."""

    GEOMETRY_COLUMN = 'geometry'
    LOCATION_COLUMN = 'location'

    def __init__(self, object_data: pd.Series):
        self.object_data = object_data
        self.object_id = object_data.name

    def get_geometry(self) -> Any:
        return self.object_data.get(self.GEOMETRY_COLUMN)

    def get_tooltip_data(self) -> dict:
        data = {'ID': self.object_id}
        for col, value in self.object_data.items():
            if pd.notna(value):
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."
                data[col] = value_str
        return data

    def get_styling_value(self, column: str) -> Any:
        return self.object_data.get(column)

    def get_location(self) -> tuple[float, float]:
        if self.LOCATION_COLUMN in self.object_data:
            location = self.object_data[self.LOCATION_COLUMN]
            if isinstance(location, Point):
                return location.y, location.x

        geometry = self.get_geometry()
        if isinstance(geometry, Point):
            return geometry.y, geometry.x
        elif isinstance(geometry, (Polygon, MultiPolygon)):
            point = geometry.representative_point()
            return point.y, point.x
        elif isinstance(geometry, LineString):
            point = geometry.interpolate(0.5, normalized=True)
            return point.y, point.x

        raise ValueError(f"Cannot determine location for {self.object_id}")


class KPIDataItem(MapDataItem):
    """Map data item for KPI objects - reuses ModelDataItem internally."""

    PROJECTION_POINT_COLUMN = 'projection_point'
    KPI_VALUE_COLUMN = 'kpi_value'

    def __init__(self, kpi: KPI, kpi_collection: KPICollection = None, study_manager=None):
        self.kpi = kpi
        self.kpi_collection = kpi_collection
        self.study_manager = study_manager
        self._object_info = kpi.get_attributed_object_info_from_model()
        self._model_item = ModelDataItem(self._object_info)

    def get_geometry(self) -> Any:
        return self._model_item.get_geometry()

    def get_tooltip_data(self) -> dict:
        kpi_data = {
            'KPI': self.kpi.get_kpi_name_with_dataset_name(),
            'Value': str(self.kpi.quantity),
        }
        model_data = self._model_item.get_tooltip_data()
        return {**kpi_data, **model_data}

    def get_styling_value(self, column: str) -> Any:
        if column == self.KPI_VALUE_COLUMN:
            return self.kpi.value
        return self._model_item.get_styling_value(column)

    def get_location(self) -> tuple[float, float]:
        if self.PROJECTION_POINT_COLUMN in self._object_info:
            point = self._object_info[self.PROJECTION_POINT_COLUMN]
            return point.y, point.x
        return self._model_item.get_location()
