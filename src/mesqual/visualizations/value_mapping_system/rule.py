from typing import Any, Callable

from mesqual.visualizations.value_mapping_system.base import BaseMapping


class RuleBasedMapping(BaseMapping):
    """Maps values with a function.

    A call returns `mapping_rule(value)`. The rule handles missing values
    itself, for example with `is_missing`.

    Args:
        mapping_rule: Function that returns the output for one value.

    Examples:
        Red for negative values, green for others, gray for missing values:

        >>> from mesqual.visualizations.value_mapping_system import is_missing
        >>> def sign_color(value: float) -> str:
        ...     if is_missing(value):
        ...         return '#cccccc'
        ...     return '#d62728' if value < 0 else '#2ca02c'
        >>> colors = RuleBasedMapping(sign_color)
        >>> colors(-3), colors(None)
        ('#d62728', '#cccccc')
    """

    def __init__(self, mapping_rule: Callable[[Any], Any]):
        self.rule = mapping_rule

    def __call__(self, value: Any) -> Any:
        return self.rule(value)
