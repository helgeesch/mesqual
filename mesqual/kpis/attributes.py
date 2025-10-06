from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable

from mesqual.flag import FlagTypeProtocol
from mesqual.kpis.aggregations import Aggregation, ValueComparison, ArithmeticValueOperation
from mesqual.units import Units


@dataclass
class KPIAttributes:
    """
    Rich metadata container for KPI instances.

    Stores all context needed for filtering, grouping, visualization,
    and unit conversion of KPI values.

    Attributes:
        flag: Variable flag (e.g., 'BZ.Results.market_price')
        model_flag: Associated model flag (e.g., 'BZ.Model')
        object_name: Specific object identifier (e.g., 'DE-LU')
        aggregation: Aggregation applied (e.g., Aggregations.Mean)
        dataset_name: Name of the dataset
        dataset_type: Type of dataset ('scenario', 'comparison', etc.)
        value_comparison: Comparison operation for comparison KPIs
        arithmetic_operation: Arithmetic operation for derived KPIs
        reference_dataset_name: Reference dataset for comparisons
        variation_dataset_name: Variation dataset for comparisons
        name_prefix: Custom prefix for KPI name
        name_suffix: Custom suffix for KPI name
        custom_name: Complete custom name override
        unit: Physical unit of the KPI value
        target_unit: Target unit for conversion
        dataset_attributes: Additional attributes from dataset (e.g., scenario attributes)
        extra_attributes: Extra attributes set by user (e.g. for filtering / grouping purposes)
    """

    # Core identifiers
    flag: FlagTypeProtocol
    model_flag: FlagTypeProtocol | None = None
    object_name: Hashable | None = None
    aggregation: Aggregation | None = None

    # Dataset context
    dataset_name: str = ''
    dataset_type: str = ''

    # Comparison-specific
    value_comparison: ValueComparison | None = None
    arithmetic_operation: ArithmeticValueOperation | None = None
    reference_dataset_name: str | None = None
    variation_dataset_name: str | None = None

    # Naming
    name_prefix: str = ''
    name_suffix: str = ''
    custom_name: str | None = None

    # Unit handling
    unit: Units.Unit | None = None
    target_unit: Units.Unit | None = None

    # Additional attributes
    dataset_attributes: dict[str, Any] = field(default_factory=dict)
    extra_attributes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self, primitive_values: bool = True) -> dict[str, Any]:
        """
        Export attributes as dictionary for filtering.

        Args:
            primitive_values: If True, convert objects to strings for serialization

        Returns:
            Dictionary representation of attributes
        """
        if not primitive_values:
            raise NotImplementedError # TODO: refactor to complex and primitive types
        d = {
            'flag': str(self.flag),
            'model_flag': str(self.model_flag),
            'object_name': self.object_name,
            'aggregation': str(self.aggregation) if primitive_values and self.aggregation else self.aggregation,
            'dataset_name': self.dataset_name,
            'dataset_type': self.dataset_type,
            'value_comparison': str(self.value_comparison) if primitive_values and self.value_comparison else self.value_comparison,
            'arithmetic_operation': str(self.arithmetic_operation) if primitive_values and self.arithmetic_operation else self.arithmetic_operation,
            'reference_dataset_name': self.reference_dataset_name,
            'variation_dataset_name': self.variation_dataset_name,
            'name_prefix': self.name_prefix,
            'name_suffix': self.name_suffix,
            'custom_name': self.custom_name,
            'unit': str(self.unit) if primitive_values and self.unit else self.unit,
            'target_unit': str(self.target_unit) if primitive_values and self.target_unit else self.target_unit,
            **self.dataset_attributes,
            **self.extra_attributes
        }
        return d

    def get(self, key: str, default: Any = None) -> Any:
        """
        Dict-like get interface for compatibility.

        Args:
            key: Attribute key to retrieve
            default: Default value if key not found

        Returns:
            Attribute value or default
        """
        return self.as_dict().get(key, default)
