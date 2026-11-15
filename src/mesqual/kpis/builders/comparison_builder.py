from __future__ import annotations

from mesqual.kpis.definitions.base import KPIDefinition
from mesqual.kpis.definitions.comparison import ComparisonKPIDefinition
from mesqual.kpis.aggregations import ValueComparison
from mesqual.kpis.builders.base import KPIBuilder


class ComparisonKPIBuilder(KPIBuilder[ComparisonKPIDefinition]):
    """
    Builder for creating comparison KPI definitions from scenario KPIs.

    Takes existing KPI definitions (typically for scenarios) and creates
    comparison definitions that compute differences, percentage changes,
    or other comparisons between datasets.

    Examples:
        >>> from mesqual.kpis import Aggregations, FlagAggKPIBuilder, ValueComparisons
        >>> base_definitions = FlagAggKPIBuilder().for_flag('Country.Results.price').with_aggregation(Aggregations.Mean).build()
        >>> comparison_definitions = (
        ...     ComparisonKPIBuilder(base_definitions)
        ...     .with_comparisons([ValueComparisons.Increase, ValueComparisons.PercentageIncrease])
        ...     .build()
        ... )
        >>> len(comparison_definitions)  # 1 base definition, 2 comparisons
        2
        >>> study.comp.add_kpis_from_definitions_to_all_child_datasets(comparison_definitions)
    """

    def __init__(self, base_definitions: list[KPIDefinition]):
        """
        Initialize builder with base KPI definitions.

        Args:
            base_definitions: List of KPI definitions to create comparisons from
        """
        super().__init__()
        self._base_definitions = base_definitions
        self._comparisons: list[ValueComparison] = []

    def with_comparison(self, comp: ValueComparison) -> ComparisonKPIBuilder:
        """
        Set a single comparison operation.

        Args:
            comp: Comparison operation (e.g., ValueComparisons.Increase)

        Returns:
            Self for chaining
        """
        self._comparisons = [comp]
        return self

    def with_comparisons(self, comps: list[ValueComparison]) -> ComparisonKPIBuilder:
        """
        Set multiple comparison operations.

        Args:
            comps: List of comparison operations

        Returns:
            Self for chaining
        """
        self._comparisons = comps
        return self

    def build(self) -> list[ComparisonKPIDefinition]:
        """
        Generate all comparison KPI definitions.

        Creates the Cartesian product of base_definitions × comparisons.

        Returns:
            List of ComparisonKPIDefinition instances
        """
        definitions = []

        for base_def in self._base_definitions:
            for comp in self._comparisons:
                comp_def = ComparisonKPIDefinition(
                    base_definition=base_def,
                    comparison=comp,
                    name_prefix=self._name_prefix,
                    name_suffix=self._name_suffix,
                    custom_name=self._custom_name,
                    extra_attributes=self._extra_attributes,
                )
                definitions.append(comp_def)

        return definitions
