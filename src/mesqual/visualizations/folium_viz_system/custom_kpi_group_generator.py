from abc import ABC, abstractmethod
from typing import List, Tuple, Literal, Union, Any, TYPE_CHECKING

import folium

from mesqual.kpis import KPICollection, KPI
from mesqual.visualizations.folium_viz_system.base_viz_system import FoliumObjectGenerator
from mesqual.visualizations.folium_viz_system.visualizable_data_item import KPIDataItem

if TYPE_CHECKING:
    from mesqual.study_manager import StudyManager
    from mesqual.datasets import DatasetCollection


class CustomKPIGroupGenerator(ABC):
    """Base class for custom KPI groups and generator routing.

    Use it when the attribute grouping of `KPICollectionMapVisualizer` does not
    fit. A subclass defines two things in one place:

    1. `create_kpi_groups_with_names`: the KPIs of each feature group, and its name.
    2. `get_generators_for_kpi`: the generators for each KPI.

    `generate_and_add_feature_groups_to_map` then builds the feature groups and
    adds them to the map.

    Typical uses:

    - Overlapping groups: the same KPI is in more than one feature group, for
      example flows with prices and flows with net positions.
    - Different styles per KPI type, for example areas for prices and arrows for
      flows.
    - Feature group names that are not a composition of KPI attributes.

    Examples:
        One feature group for each of the scenarios `base` and `high_demand`,
        with areas for the prices and arrows for the border flows:

        >>> import folium
        >>> from mesqual.kpis import Aggregations, FlagAggKPIBuilder
        >>> from mesqual.visualizations.folium_viz_system import AreaGenerator, ArrowIconGenerator
        >>> definitions = (
        ...     FlagAggKPIBuilder()
        ...     .for_flags(['Country.Results.price', 'CountryBorder.Results.flow_sent_net'])
        ...     .with_aggregation(Aggregations.Mean)
        ...     .build()
        ... )
        >>> study.scen.add_kpis_from_definitions_to_all_child_datasets(definitions)
        >>> class PricesAndFlows(CustomKPIGroupGenerator):
        ...     def __init__(self):
        ...         self.areas = AreaGenerator()
        ...         self.arrows = ArrowIconGenerator()
        ...
        ...     def create_kpi_groups_with_names(self, source):
        ...         kpis = source.get_merged_kpi_collection()
        ...         return [(name, kpis.filter(dataset_name=name)) for name in ['base', 'high_demand']]
        ...
        ...     def get_generators_for_kpi(self, kpi):
        ...         if kpi.attributes.flag == 'Country.Results.price':
        ...             return [self.areas]
        ...         return [self.arrows]
        >>> m = folium.Map(location=[48, 9], zoom_start=6)
        >>> feature_groups = PricesAndFlows().generate_and_add_feature_groups_to_map(study.scen, m)
        >>> [fg.layer_name for fg in feature_groups]
        ['base', 'high_demand']
    """

    @abstractmethod
    def create_kpi_groups_with_names(
        self, source: Union['KPICollection', 'StudyManager', 'DatasetCollection', Any]
    ) -> List[Tuple[str, KPICollection]]:
        """Return the feature groups as (name, KPIs) pairs.

        The source is what the caller passes to
        `generate_and_add_feature_groups_to_map`, for example a `KPICollection`, a
        `StudyManager` or a `DatasetCollection`. A KPI can be in more than one
        group; it is then drawn once per group. The order of the list is the order
        in the layer control. Return an empty list if there are no groups.

        Args:
            source: Source of the KPIs.

        Returns:
            List of `(group_name, kpi_group)` tuples. `group_name` is the name in
                the layer control and must be unique; `kpi_group` is a `KPICollection`.

        Examples:
            One group per dataset of a `DatasetCollection`:

            ```python
            def create_kpi_groups_with_names(self, source):
                return [(dataset.name, dataset.kpi_collection) for dataset in source.dataset_iterator]
            ```
        """
        pass

    @abstractmethod
    def get_generators_for_kpi(self, kpi: KPI) -> List[FoliumObjectGenerator]:
        """Return the generators for one KPI.

        The method is called for each KPI of each group. Route by the KPI
        attributes, such as `flag`, `dataset_name`, `aggregation` or `object_name`.
        The generators draw in the order of the list, so a line after an area
        is drawn over it. Markers, such as text overlays and arrow icons, are
        always above areas and lines. An empty list skips the KPI.

        Args:
            kpi: The KPI.

        Returns:
            List of `FoliumObjectGenerator` instances for this KPI.
        """
        pass

    def generate_and_add_feature_groups_to_map(
        self,
        source: Union['KPICollection', 'StudyManager', 'DatasetCollection', Any],
        map_obj: folium.Map,
        show: Literal['first', 'last', 'all', 'none'] = 'first',
        overlay: bool = False,
        **data_item_kwargs: Any,
    ) -> List[folium.FeatureGroup]:
        """Generate the feature groups and add them to the map.

        For each group of `create_kpi_groups_with_names`, the method creates a
        folium `FeatureGroup`, wraps each KPI in a `KPIDataItem`, applies the
        generators of `get_generators_for_kpi` and adds the group to the map. A
        generator that raises an error for a KPI is skipped for that KPI. The
        example is in `CustomKPIGroupGenerator`.

        Args:
            source: Source of the KPIs, passed to `create_kpi_groups_with_names`.
            map_obj: Folium map to add the feature groups to.
            show: Feature groups visible at the start: `'first'` (default),
                `'last'`, `'all'` or `'none'`.
            overlay: Whether the feature groups are overlays in the layer control.
            **data_item_kwargs: Arguments for each `KPIDataItem`.

        Returns:
            The feature groups added to the map.
        """
        # Get custom groups from subclass
        groups_with_names = self.create_kpi_groups_with_names(source)

        feature_groups = []
        total_groups = len(groups_with_names)

        for idx, (group_name, kpi_group) in enumerate(groups_with_names):
            # Determine if this group should be shown initially
            show_fg = self._should_show_group(idx, total_groups, show)

            # Create feature group
            fg = folium.FeatureGroup(name=group_name, overlay=overlay, show=show_fg)

            # Process each KPI in the group
            for kpi in kpi_group:
                # Get generators for this specific KPI from subclass
                generators = self.get_generators_for_kpi(kpi)

                # Create data item for this KPI
                data_item = KPIDataItem(kpi, kpi_group, **data_item_kwargs)

                # Apply each generator
                for generator in generators:
                    try:
                        generator.generate(data_item, fg)
                    except Exception as e:
                        # Log or handle generation errors without stopping the loop
                        # This matches the behavior in KPICollectionMapVisualizer
                        continue

            feature_groups.append(fg)
            map_obj.add_child(fg)

        return feature_groups

    @staticmethod
    def _should_show_group(idx: int, total: int, show: str) -> bool:
        """
        Determine if a feature group should be shown initially.

        Args:
            idx: Index of the feature group (0-based)
            total: Total number of feature groups
            show: Show strategy ('first', 'last', 'all', 'none')

        Returns:
            True if the group should be visible initially
        """
        if show == 'first':
            return idx == 0
        elif show == 'last':
            return idx == total - 1
        elif show == 'all':
            return True
        else:  # 'none'
            return False
