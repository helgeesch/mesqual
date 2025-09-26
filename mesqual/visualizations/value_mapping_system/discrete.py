from typing import Any, Literal

from mesqual.visualizations.value_mapping_system.base import BaseMapping

MAPPING_MODES = Literal['fallback', 'auto_assign']


class DiscreteInputMapping(BaseMapping):
    DEFAULT_TARGET_VALUES: list = []

    def __init__(self, mapping: dict | None = None, mode: MAPPING_MODES = "fallback", default_output: Any = None):
        self._mode = mode
        self._mapping = mapping or {}
        self._used_outputs = set(self._mapping.values())
        self._default_output = default_output
        self._available_outputs = [v for v in self.DEFAULT_TARGET_VALUES if v not in self._used_outputs]

    @property
    def mapping(self) -> dict:
        return self._mapping

    @property
    def default_output(self) -> Any:
        return self._default_output

    def __call__(self, value):
        if value in self._mapping:
            return self._mapping[value]
        return self._handle_missing(value)

    def _handle_missing(self, value):
        if self._mode == 'auto_assign':
            return self._auto_assign(value)
        return self._default_output

    def _auto_assign(self, value):
        if not self._available_outputs:
            return self._default_output
        new_output = self._available_outputs.pop(0)
        self._mapping[value] = new_output
        self._used_outputs.add(new_output)
        return new_output


class DiscreteColorMapping(DiscreteInputMapping):
    DEFAULT_TARGET_VALUES = [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf',
    ]
    DEFAULT_OUTPUT = '#cccccc'


class DiscreteLineDashPatternMapping(DiscreteInputMapping):
    DEFAULT_TARGET_VALUES = [
        '',
        '5,5',
        '5,5,1,5',
        '10,5',
        '5,10',
        '15,5,5,5',
        '10,5,5,5,5,5',
    ]
    DEFAULT_OUTPUT = '2,2'


class DiscreteLineWidthMapping(DiscreteInputMapping):
    DEFAULT_TARGET_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    DEFAULT_OUTPUT = 1.0


class DiscreteIconMapping(DiscreteInputMapping):
    pass
