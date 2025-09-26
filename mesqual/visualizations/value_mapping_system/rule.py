from typing import Callable, Any

from mesqual.visualizations.value_mapping_system.base import BaseMapping


class RuleBasedMapping(BaseMapping):

    def __init__(self, mapping_rule: Callable[[Any], Any]):
        self.rule = mapping_rule

    def __call__(self, value):
        return self.rule(value)
