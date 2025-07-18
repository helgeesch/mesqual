from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from shapely import Point, Polygon, MultiPolygon, LineString

from mescal.kpis import KPI, KPICollection


class MapDataItem(ABC):
    """Abstract interface for data items that can be visualized on maps."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def get_name(self) -> str:
        """Get a representative name for the object."""
        pass

    @abstractmethod
    def get_text_representation(self) -> str:
        """Get a representative text for the object."""
        pass

    @abstractmethod
    def get_tooltip_data(self) -> dict:
        """Get data for tooltip display."""
        pass

    @abstractmethod
    def get_object_attribute(self, attribute: str) -> Any:
        """Get value for styling from specified column."""
        pass

    @abstractmethod
    def object_has_attribute(self, attribute: str) -> bool:
        pass


class ModelDataItem(MapDataItem):
    """Map data item for model DataFrame rows."""

    def __init__(self, object_data: pd.Series, **kwargs):
        self.object_data = object_data
        self.object_id = object_data.name
        super().__init__(**kwargs)

    def get_name(self) -> str:
        return str(self.object_id)

    def get_text_representation(self) -> str:
        return self.get_name()

    def get_tooltip_data(self) -> dict:
        data = {'ID': self.object_id}
        for col, value in self.object_data.items():
            if pd.notna(value):
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."
                data[col] = value_str
        return data

    def get_object_attribute(self, attribute: str) -> Any:
        return self.object_data.get(attribute)

    def object_has_attribute(self, attribute: str) -> bool:
        return attribute in self.object_data


class KPIDataItem(MapDataItem):
    """Map data item for KPI objects - reuses ModelDataItem internally."""

    KPI_VALUE_COLUMN = 'kpi_value'

    def __init__(self, kpi: KPI, kpi_collection: KPICollection = None, **kwargs):
        self.kpi = kpi
        self.kpi_collection = kpi_collection
        self._object_info = kpi.get_attributed_object_info_from_model()
        self._model_item = ModelDataItem(self._object_info)
        super().__init__(**kwargs)

    def get_name(self) -> str:
        return str(self.kpi.name)

    def get_text_representation(self) -> str:
        return f"{self.kpi.value:.1f}"  # TODO: use pretty formatting and quantities etc.

    def get_tooltip_data(self) -> dict:
        kpi_data = {
            'KPI': self.kpi.get_kpi_name_with_dataset_name(),
            'Value': str(self.kpi.quantity),
        }
        model_data = self._model_item.get_tooltip_data()
        return {**kpi_data, **model_data}

    def get_object_attribute(self, attribute: str) -> Any:
        if attribute == self.KPI_VALUE_COLUMN:
            return self.kpi.value
        return self._model_item.get_object_attribute(attribute)

    def object_has_attribute(self, attribute: str) -> bool:
        if attribute == self.KPI_VALUE_COLUMN:
            return True
        return self._model_item.object_has_attribute(attribute)
