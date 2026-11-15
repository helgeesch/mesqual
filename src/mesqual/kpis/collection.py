from __future__ import annotations

from typing import Any, Callable, Literal, Iterable
from collections import defaultdict
import pandas as pd

from mesqual.kpis.kpi import KPI
from mesqual.units import Units


class KPICollection:
    """
    Collection of KPIs with filtering, grouping, and export capabilities.

    Provides rich query interface for finding related KPIs and organizing
    KPIs for visualization and reporting.
    """

    def __init__(self, kpis: Iterable[KPI] | None = None):
        """
        Initialize KPI collection.

        Args:
            kpis: Optional list of KPIs to initialize with
        """
        self._kpis: list[KPI] = list(kpis) if kpis is not None else []

    def add(self, kpi: KPI) -> None:
        """Add single KPI to collection."""
        self._kpis.append(kpi)

    def extend(self, kpis: list[KPI]) -> None:
        """Add multiple KPIs to collection."""
        self._kpis.extend(kpis)

    def clear(self) -> None:
        """Remove all KPIs from collection."""
        self._kpis.clear()

    def filter(self, **attribute_filters: Any) -> KPICollection:
        """
        Filter KPIs by exact attribute matches.

        Args:
            **attribute_filters: Attribute name-value pairs to filter by

        Returns:
            New KPICollection with filtered KPIs

        Examples:
            >>> from mesqual.kpis import Aggregations, FlagAggKPIBuilder
            >>> dataset.add_kpis_from_definitions(
            ...     FlagAggKPIBuilder().for_flag('Country.Results.price').with_aggregations([Aggregations.Mean, Aggregations.Max]).build()
            ... )
            >>> [kpi.name for kpi in dataset.kpi_collection.filter(object_name='C1', aggregation=Aggregations.Mean)]
            ['base Country.Results.price Mean C1']
        """
        filtered = []
        for kpi in self._kpis:
            primitive = kpi.to_dict(primitive_values=True)
            non_primitive = kpi.to_dict(primitive_values=False)
            match = all(
                primitive.get(attr, None) == value or non_primitive.get(attr, None) == value
                for attr, value in attribute_filters.items()
            )
            if match:
                filtered.append(kpi)
        return KPICollection(filtered)

    def filter_by_model_properties(
        self,
        properties: dict[str, Any] | None = None,
        query_expr: str | None = None,
        filter_funcs: dict[str, Callable[[Any], bool]] | None = None,
    ) -> KPICollection:
        """
        Filter KPIs by properties from their model objects.

        Three modes of operation:

        1. Property filters: Exact match or list membership
        2. Query expression: Pandas query string
        3. Filter functions: Custom functions applied to properties

        All conditions across all modes are combined with AND logic.

        Args:
            properties: Dict of property names to values. Scalars for exact match,
                lists for membership checks.
            query_expr: Pandas query expression (uses engine="python"). Local variables
                (`@name`) are not supported.
            filter_funcs: Dict of property names to filter functions. A KPI that misses the
                property or for which the function raises does not match.

        Returns:
            New KPICollection with filtered KPIs

        Raises:
            Exception: The error of the first KPI if `query_expr` fails for every KPI,
                for example `SyntaxError` for a syntax error or
                `pandas.errors.UndefinedVariableError` for a name that no KPI has.
                A KPI for which the query fails does not match.

        Examples:
            The generators have the model properties `node`, `fuel` and `capacity`:

            >>> from mesqual.kpis import Aggregations, FlagAggKPIBuilder
            >>> dataset.add_kpis_from_definitions(
            ...     FlagAggKPIBuilder().for_flag('Generator.Results.generation').with_aggregation(Aggregations.Mean).build()
            ... )
            >>> kpis = dataset.kpi_collection
            >>> [kpi.attributes.object_name for kpi in kpis.filter_by_model_properties(properties={'node': 'N1', 'fuel': ['nuclear', 'wind']})]
            ['G1']
            >>> [kpi.attributes.object_name for kpi in kpis.filter_by_model_properties(query_expr='node == "N2" and capacity > 1050')]
            ['G4']
            >>> [kpi.attributes.object_name for kpi in kpis.filter_by_model_properties(filter_funcs={'capacity': lambda x: x > 1050})]
            ['G4', 'G5']
        """
        kpis = [kpi for kpi in self._kpis if kpi.attributes.object_name is not None]
        data = [kpi.get_object_info_from_model() for kpi in kpis]
        return self._filter_by_series(kpis, data, properties, query_expr, filter_funcs)

    def filter_by_kpi_attributes(
        self,
        attributes: dict[str, Any] | None = None,
        query_expr: str | None = None,
        filter_funcs: dict[str, Callable[[Any], bool]] | None = None,
    ) -> KPICollection:
        """
        Filter KPIs by their attributes.

        Three modes of operation:

        1. Property filters: Exact match or list membership
        2. Query expression: Pandas query string
        3. Filter functions: Custom functions applied to attributes

        All conditions across all modes are combined with AND logic.

        Args:
            attributes: Dict of attribute names to values. Scalars for exact match,
                lists for membership checks.
            query_expr: Pandas query expression (uses engine="python"). Local variables
                (`@name`) are not supported.
            filter_funcs: Dict of attribute names to filter functions. A KPI that misses the
                attribute or for which the function raises does not match.

        Returns:
            New KPICollection with filtered KPIs

        Raises:
            Exception: The error of the first KPI if `query_expr` fails for every KPI,
                for example `SyntaxError` for a syntax error or
                `pandas.errors.UndefinedVariableError` for a name that no KPI has.
                A KPI for which the query fails does not match.

        Examples:
            >>> from mesqual.kpis import Aggregations, FlagAggKPIBuilder
            >>> dataset.add_kpis_from_definitions(
            ...     FlagAggKPIBuilder().for_flag('Country.Results.price').with_aggregations([Aggregations.Mean, Aggregations.Max]).build()
            ... )
            >>> kpis = dataset.kpi_collection
            >>> len(kpis.filter_by_kpi_attributes(attributes={'object_name': ['C1', 'C2']}))
            4
            >>> [kpi.name for kpi in kpis.filter_by_kpi_attributes(query_expr='aggregation == "Max" and value > 65')]
            ['base Country.Results.price Max C2']
            >>> len(kpis.filter_by_kpi_attributes(filter_funcs={'value': lambda x: x > 50}))
            2
        """
        data = [pd.Series(kpi.to_dict(primitive_values=True)) for kpi in self._kpis]
        return self._filter_by_series(self._kpis, data, attributes, query_expr, filter_funcs)

    @classmethod
    def _filter_by_series(
        cls,
        kpis: list[KPI],
        data: list[pd.Series],
        properties: dict[str, Any] | None,
        query_expr: str | None,
        filter_funcs: dict[str, Callable[[Any], bool]] | None,
    ) -> KPICollection:
        """Keep the KPIs whose data passes the filters.

        The query runs first for every KPI. A query that fails for every KPI raises
        the first error; a KPI for which the query fails does not match.
        """
        filtered = []
        errors = []
        for kpi, series in zip(kpis, data):
            try:
                passes = cls._series_passes_property_filters(series, properties, query_expr, filter_funcs)
            except Exception as error:
                errors.append(error)
                continue
            if passes:
                filtered.append(kpi)
        if errors and len(errors) == len(kpis):
            raise errors[0]
        return KPICollection(filtered)

    @classmethod
    def _series_passes_property_filters(
        cls,
        data: pd.Series,
        properties: dict[str, Any] | None = None,
        query_expr: str | None = None,
        filter_funcs: dict[str, Callable[[Any], bool]] | None = None,
    ) -> bool:
        """
        Apply property filters to a data Series.

        Private method used by both filter_by_model_properties and filter_by_kpi_attributes.

        Args:
            data: Series containing property/attribute data
            properties: Dict of property names to values for exact match or list membership
            query_expr: Pandas query expression
            filter_funcs: Dict of property names to filter functions

        Returns:
            True if data passes all filters, False otherwise
        """
        if query_expr is not None and not cls._passes_query(data, query_expr):
            return False

        if properties is not None:
            for prop_name, expected_value in properties.items():
                if prop_name not in data.index:
                    return False

                actual_value = data[prop_name]

                if isinstance(expected_value, list):
                    if actual_value not in expected_value:
                        return False
                else:
                    if actual_value != expected_value:
                        return False

        if filter_funcs is not None:
            for prop_name, filter_function in filter_funcs.items():
                if prop_name not in data.index:
                    return False

                try:
                    if not filter_function(data[prop_name]):
                        return False
                except Exception:
                    return False

        return True

    @staticmethod
    def _passes_query(data: pd.Series, query_expr: str) -> bool:
        temp_df = pd.DataFrame({k: [v] for k, v in data.items()})
        return not temp_df.query(query_expr, engine='python').empty

    def group_by(self, *attributes: str) -> dict[tuple, KPICollection]:
        """
        Group KPIs by attribute values.

        Args:
            *attributes: Attribute names to group by

        Returns:
            Dictionary mapping attribute value tuples to KPICollections

        Examples:
            >>> from mesqual.kpis import Aggregations, FlagAggKPIBuilder
            >>> dataset.add_kpis_from_definitions(
            ...     FlagAggKPIBuilder().for_flag('Country.Results.price').with_aggregations([Aggregations.Mean, Aggregations.Max]).build()
            ... )
            >>> groups = dataset.kpi_collection.group_by('object_name')
            >>> {key: len(group) for key, group in groups.items()}
            {('C1',): 2, ('C2',): 2}
        """
        groups = defaultdict(list)

        for kpi in self._kpis:
            key = tuple(self._get_attribute_value(kpi, attr) for attr in attributes)
            groups[key].append(kpi)

        return {k: KPICollection(v) for k, v in groups.items()}

    @staticmethod
    def _get_attribute_value(kpi: KPI, attribute: str) -> Any:
        value = getattr(kpi.attributes, attribute, None)
        if value is None:
            value = kpi.attributes.dataset_attributes.get(attribute)
        return value

    def get_related(
        self, reference_kpi: KPI, vary_attributes: list[str], exclude_attributes: list[str] | None = None
    ) -> 'KPICollection':
        """
        Find KPIs related to reference, varying only specified attributes.

        Args:
            reference_kpi: The KPI to find relatives for
            vary_attributes: Attributes that can differ (e.g., ['aggregation'])
            exclude_attributes: Attributes to ignore in comparison (e.g., ['name_prefix'])

        Returns:
            New KPICollection with related KPIs

        Examples:
            >>> from mesqual.kpis import Aggregations, FlagAggKPIBuilder
            >>> dataset.add_kpis_from_definitions(
            ...     FlagAggKPIBuilder().for_flag('Country.Results.price').with_aggregations([Aggregations.Mean, Aggregations.Max]).build()
            ... )
            >>> kpis = dataset.kpi_collection
            >>> mean_c1 = kpis.filter(object_name='C1', aggregation=Aggregations.Mean)[0]
            >>> [kpi.name for kpi in kpis.get_related(mean_c1, vary_attributes=['aggregation'])]
            ['base Country.Results.price Max C1']
            >>> [kpi.name for kpi in kpis.get_related(mean_c1, vary_attributes=['object_name'])]
            ['base Country.Results.price Mean C2']
        """
        exclude_attributes = exclude_attributes or ['name_prefix', 'name_suffix', 'custom_name']
        ref_attrs = reference_kpi.attributes.as_dict()
        related = []

        for kpi in self._kpis:
            if kpi is reference_kpi:
                continue

            kpi_attrs = kpi.attributes.as_dict()
            match = True

            for attr_name, ref_value in ref_attrs.items():
                # Skip attributes we want to vary or exclude
                if attr_name in vary_attributes or attr_name in exclude_attributes:
                    continue

                if kpi_attrs.get(attr_name) != ref_value:
                    match = False
                    break

            if match:
                related.append(kpi)

        return KPICollection(related)

    def get_all_attribute_values(self, attribute: str) -> set:
        """
        Get all unique values for a specific attribute across collection.

        Args:
            attribute: Attribute name to get values for

        Returns:
            Set of unique values for the attribute
        """
        values = set()
        for kpi in self._kpis:
            val = self._get_attribute_value(kpi, attribute)
            if val is not None:
                values.add(val)
        return values

    def get_all_kpi_attributes_and_value_sets(self, primitive_values: bool = False) -> dict[str, set]:
        """
        Get all attribute names and their unique value sets.

        Used by KPIGroupingManager to group KPIs.

        Args:
            primitive_values: If True, convert values to primitives

        Returns:
            Dictionary mapping attribute names to sets of values
        """
        attribute_sets = defaultdict(set)

        for kpi in self._kpis:
            attrs = kpi.attributes.as_dict(primitive_values=primitive_values)
            for attr_name, attr_value in attrs.items():
                if attr_value is not None:
                    attribute_sets[attr_name].add(attr_value)

        return dict(attribute_sets)

    def get_in_common_kpi_attributes(self, primitive_values: bool = False) -> dict:
        """
        Get attributes that have same value across all KPIs in collection.

        Args:
            primitive_values: If True, convert values to primitives

        Returns:
            Dictionary of common attributes
        """
        if self.empty:
            return {}

        # Start with first KPI's attributes
        common = self._kpis[0].attributes.as_dict(primitive_values=primitive_values)

        # Remove any that differ in subsequent KPIs
        for kpi in self._kpis[1:]:
            attrs = kpi.attributes.as_dict(primitive_values=primitive_values)
            common = {k: v for k, v in common.items() if attrs.get(k) == v}

        return common

    def to_dataframe(
        self,
        unit_handling: Literal['original', 'auto_convert', 'target', 'custom'] = 'original',
        target_unit: Units.Unit | None = None,
        target_units_by_group: dict[tuple, Units.Unit] | None = None,
        group_by_attributes: list[str] | None = None,
        normalize_to_collection: bool = False,
    ) -> pd.DataFrame:
        """
        Export KPIs as DataFrame, with the units chosen by `unit_handling`.

        Args:
            unit_handling: Strategy for unit conversion:

                - 'original': Keep original units
                - 'auto_convert': Convert to pretty units per KPI
                - 'target': Use single target_unit for all
                - 'custom': Use target_units_by_group mapping
            target_unit: Single target unit (when unit_handling='target')
            target_units_by_group: Dict mapping group key → target unit
            group_by_attributes: Attributes to group by for 'custom' mode
            normalize_to_collection: If True, find common "pretty" unit across entire collection

        Returns:
            DataFrame with KPI data

        Examples:
            >>> from mesqual.kpis import Aggregations, FlagAggKPIBuilder
            >>> from mesqual.units import Units
            >>> dataset.add_kpis_from_definitions(
            ...     FlagAggKPIBuilder().for_flag('Generator.Results.generation').with_aggregation(Aggregations.Mean).build()
            ... )
            >>> kpis = dataset.kpi_collection
            >>> kpis.to_dataframe()[['object_name', 'value', 'unit']].round(1).head(2)
              object_name  value unit
            0          G1  263.3  MWh
            1          G2  191.7  MWh
            >>> kpis.to_dataframe(unit_handling='target', target_unit=Units.GWh)[['object_name', 'value', 'unit']].round(3).head(2)
              object_name  value unit
            0          G1  0.263  GWh
            1          G2  0.192  GWh
        """
        data = []

        # Normalize to collection: find common pretty unit
        common_unit = None
        if normalize_to_collection:
            quantities = [kpi.quantity for kpi in self._kpis]
            try:
                common_unit = Units.get_common_pretty_unit_for_quantities(quantities)
            except ValueError:
                # Quantities have different dimensionalities, fall back to auto_convert
                pass

        for kpi in self._kpis:
            quantity = kpi.quantity

            # Apply unit conversion based on strategy
            if normalize_to_collection and common_unit:
                quantity = quantity.to(common_unit)

            elif unit_handling == 'auto_convert':
                quantity = Units.get_quantity_in_pretty_unit(quantity)

            elif unit_handling == 'target' and target_unit:
                quantity = quantity.to(target_unit)

            elif unit_handling == 'custom' and target_units_by_group and group_by_attributes:
                # Build group key
                kpi_attributes = kpi.attributes.as_dict(primitive_values=True)
                group_key = tuple(
                    kpi_attributes[attr] if attr in kpi_attributes else None for attr in group_by_attributes
                )
                if group_key in target_units_by_group:
                    quantity = quantity.to(target_units_by_group[group_key])

            row = {
                'name': kpi.name,
                **kpi.attributes.as_dict(primitive_values=True),
                'value': quantity.magnitude,
                'quantity': quantity,
                'unit': str(quantity.units),  # Set after attributes to avoid being overwritten
            }
            data.append(row)

        return pd.DataFrame(data)

    @property
    def kpis(self) -> dict[str, KPI]:
        return {kpi.name: kpi for kpi in self._kpis}

    def __iter__(self):
        """Iterate over KPIs in collection."""
        return iter(self._kpis)

    def __len__(self):
        """Number of KPIs in collection."""
        return len(self._kpis)

    def __getitem__(self, index):
        """Get KPI by index."""
        return self._kpis[index]

    def __contains__(self, item):
        kpi_name = item.name
        return any(kpi.name == kpi_name for kpi in self._kpis)

    @property
    def empty(self) -> bool:
        """Check if collection is empty."""
        return len(self._kpis) == 0

    @property
    def size(self) -> int:
        """Get number of KPIs in collection."""
        return len(self._kpis)

    def __repr__(self) -> str:
        """String representation of collection."""
        return f'KPICollection({self.size} KPIs)'

    def __add__(self, other: KPICollection) -> KPICollection:
        return KPICollection(self._kpis + other._kpis)
