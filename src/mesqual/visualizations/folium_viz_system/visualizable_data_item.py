from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

import pandas as pd
from shapely.geometry.base import BaseGeometry

from mesqual.kpis import KPI, KPICollection
from mesqual.units import Units
from mesqual.visualizations.value_mapping_system.base import is_missing

if TYPE_CHECKING:
    from mesqual.units import QuantityToTextConverter


class VisualizableDataItem(ABC):
    """
    Abstract interface for data items that can be visualized on maps.

    Defines the contract for all data items that can be processed by the
    folium visualization system. Provides attribute access, tooltip data
    generation, and text representation for map elements.

    This abstraction enables polymorphic handling of different data sources
    (model DataFrames, KPI objects, custom data) within the visualization
    pipeline while maintaining consistent interface expectations.
    """

    def __init__(self, **kwargs: Any):
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
    def get_tooltip_data(self, **kwargs) -> dict:
        """Get data for tooltip display."""
        pass

    @abstractmethod
    def get_object_attribute(self, attribute: str) -> Any:
        """Get value for styling from specified column."""
        pass

    @abstractmethod
    def object_has_attribute(self, attribute: str) -> bool:
        pass


class ModelDataItem(VisualizableDataItem):
    """
    Visualizable data item for model DataFrame rows.

    Wraps pandas Series objects (DataFrame rows) to provide the VisualizableDataItem
    interface. Commonly used for visualizing static model data like network
    topology, geographic boundaries, or reference datasets.

    The attributes are the values of the row. The name is the row index.

    Args:
        object_data: Pandas Series representing a DataFrame row
        object_type: Optional type identifier for the object
        **kwargs: Additional attributes to set on the data item

    Examples:
        >>> item = ModelDataItem(dataset.fetch('Country.Model').loc['C1'], object_type='Country')
        >>> item.get_name()
        'C1'
        >>> item.get_tooltip_data()
        {'ID': 'C1', 'name': 'Country 1'}
        >>> item.get_object_attribute('geometry')
        <POLYGON ((9 46, 9 50, 5 50, 5 46, 9 46))>

        The generators wrap each row of a model DataFrame in a `ModelDataItem`.
    """

    OBJECT_NAME_COLUMNS = ['name', 'object_id', 'index', 'object_name']

    def __init__(self, object_data: pd.Series, object_type: str | Any = None, **kwargs: Any):
        self.object_data = object_data
        self.object_id = object_data.name
        self.object_type = object_type
        self.name_attributes = self.OBJECT_NAME_COLUMNS + [self.object_type]
        super().__init__(**kwargs)

    def get_name(self) -> str:
        return str(self.object_id)

    def get_text_representation(self) -> str:
        return self.get_name()

    def get_tooltip_data(self, **kwargs) -> dict:
        """Return the ID and the column values of the row, as text.

        Leaves out missing values and geometries. Cuts a value longer than 50
        characters to 47 characters and `...`.
        """
        data = {'ID': self.object_id}
        for col, value in self.object_data.items():
            if isinstance(value, BaseGeometry):
                continue
            if not is_missing(value):
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[:47] + '...'
                data[col] = value_str
        return data

    def get_object_attribute(self, attribute: str) -> Any:
        if (attribute in self.name_attributes) and (attribute not in self.object_data):
            return self.get_name()
        return self.object_data.get(attribute)

    def object_has_attribute(self, attribute: str) -> bool:
        return (attribute in self.object_data) or (attribute in self.name_attributes)


class KPIDataItem(VisualizableDataItem):
    """
    Visualizable data item for KPI objects.

    Wraps a KPI for the map. `get_object_attribute` reads the attributes of the
    model object of the KPI; `kpi_value`, `value` and `kpi` give the KPI value
    when the model object has no attribute of that name. The text is the KPI
    value in its pretty unit.

    Args:
        kpi: KPI with its value and attributes
        kpi_collection: Optional KPI collection for context
        **kwargs: Additional attributes to set on the data item

    Examples:
        >>> from mesqual.kpis import Aggregations, FlagAggKPIBuilder
        >>> dataset.add_kpis_from_definitions(
        ...     FlagAggKPIBuilder().for_flag('Country.Results.price').with_aggregation(Aggregations.Mean).build()
        ... )
        >>> item = KPIDataItem(dataset.kpi_collection.filter(object_name='C1')[0], dataset.kpi_collection)
        >>> print(round(item.get_object_attribute('kpi_value'), 2))
        42.1
        >>> item.get_object_attribute('name')
        'Country 1'
        >>> item.get_text_representation()
        '42.1 €/MWh'

        The generators wrap each KPI of a collection in a `KPIDataItem`.
    """

    KPI_VALUE_COLUMNS = ['kpi_value', 'value', 'kpi']

    def __init__(self, kpi: KPI, kpi_collection: KPICollection = None, **kwargs: Any):
        self.kpi = kpi
        self.kpi_collection = kpi_collection
        self._object_info = kpi.get_object_info_from_model()
        self._model_item = ModelDataItem(self._object_info)
        super().__init__(**kwargs)

    def get_name(self) -> str:
        return str(self.kpi.name)

    def get_text_representation(self, converter: QuantityToTextConverter = None) -> str:
        """
        Get formatted text representation of the KPI value.

        Args:
            converter: Optional QuantityToTextConverter for custom formatting

        Returns:
            Formatted string representation of the KPI value
        """

        if converter is not None:
            return converter.convert(self.kpi.quantity)

        q = Units.get_quantity_in_pretty_unit(self.kpi.quantity)
        text = Units.get_pretty_text_for_quantity(q)
        return text

    def get_tooltip_data(self, converter: QuantityToTextConverter = None) -> dict:
        kpi_data = {
            'KPI': self.kpi.get_kpi_name_with_dataset_name(),
            'Value': self.get_text_representation(converter),
        }
        _MAX_MODEL_DATA_LEN = 3
        model_data = self._model_item.get_tooltip_data()
        model_data = dict(list(model_data.items())[:_MAX_MODEL_DATA_LEN])
        if len(model_data) > _MAX_MODEL_DATA_LEN:
            model_data['...'] = '...'
        return {**kpi_data, **model_data}

    def get_object_attribute(self, attribute: str) -> Any:
        if (attribute in self.KPI_VALUE_COLUMNS) and (not self._model_item.object_has_attribute(attribute)):
            return self.kpi.value
        return self._model_item.get_object_attribute(attribute)

    def object_has_attribute(self, attribute: str) -> bool:
        if attribute in self.KPI_VALUE_COLUMNS:
            return True
        return self._model_item.object_has_attribute(attribute)
