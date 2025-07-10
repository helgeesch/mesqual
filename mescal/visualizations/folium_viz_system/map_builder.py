from typing import Union

import folium
import pandas as pd

from mescal.kpis import KPICollection, KPI
from mescal.visualizations.folium_viz_system.folium_generators import FoliumObjectGenerator
from mescal.visualizations.folium_viz_system.map_data_item import ModelDataItem, KPIDataItem


class FoliumMapBuilder:
    """High-level builder for creating folium visualizations."""

    def __init__(self):
        self.generators: dict[str, FoliumObjectGenerator] = {}

    def register_generator(self, name: str, generator: FoliumObjectGenerator):
        """Register a generator for a specific visualization type."""
        self.generators[name] = generator

    def add_model_data(
            self,
            generator: Union[str, FoliumObjectGenerator],
            model_df: pd.DataFrame,
            feature_group: folium.FeatureGroup
    ):
        """Add model DataFrame data to the map."""
        gen = self._resolve_generator(generator)

        for _, row in model_df.iterrows():
            data_item = ModelDataItem(row)
            gen.generate(data_item, feature_group)

    def add_kpi_data(
            self,
            generator: Union[str, FoliumObjectGenerator],
            kpi_collection: KPICollection,
            feature_group: folium.FeatureGroup,
            study_manager=None
    ):
        """Add KPI data to the map."""
        gen = self._resolve_generator(generator)

        for kpi in kpi_collection:
            data_item = KPIDataItem(kpi, kpi_collection, study_manager)
            gen.generate(data_item, feature_group)

    def add_single_kpi(
            self,
            generator: Union[str, FoliumObjectGenerator],
            kpi: KPI,
            feature_group: folium.FeatureGroup,
            kpi_collection: KPICollection = None,
            study_manager=None
    ):
        """Add a single KPI to the map with optional context."""
        gen = self._resolve_generator(generator)
        data_item = KPIDataItem(kpi, kpi_collection, study_manager)
        gen.generate(data_item, feature_group)

    def _resolve_generator(self, generator: Union[str, FoliumObjectGenerator]) -> FoliumObjectGenerator:
        """Resolve generator from name or return instance directly."""
        if isinstance(generator, str):
            if generator not in self.generators:
                raise ValueError(f"Generator '{generator}' not registered")
            return self.generators[generator]
        return generator
