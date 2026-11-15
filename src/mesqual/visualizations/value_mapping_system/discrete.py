from collections.abc import Iterable, Mapping
from typing import Any

from mesqual.visualizations.value_mapping_system.base import BaseMapping, is_missing


class DiscreteInputMapping(BaseMapping):
    """Maps discrete input values to outputs, such as colors or line widths.

    A call returns the output for one value:

    1. A missing value (`None`, `NaN`, `pd.NA`, `pd.NaT`; see `is_missing`) gets
       `default_output`.
    2. A value in `mapping` gets its output.
    3. Another value gets the next palette output, if a palette is given. The
       mapping stores this assignment, so the value keeps its output for the life
       of the mapping object.
    4. Without a palette, another value gets `default_output`.

    A missing value gets `default_output` only when the mapping is called with it;
    `Series.map(mapping, na_action="ignore")`, for example, does not call it.

    The palette leaves out the outputs of the fixed entries, so an assigned output
    does not repeat a fixed one; if that leaves no outputs, the palette uses all
    given outputs. The palette starts again from its first output when it runs out.

    Args:
        mapping: Fixed output per input value. The mapping copies it and does
            not change it.
        palette: Outputs for values without a fixed output, in assignment order.
            `None` or no outputs: these values get `default_output`.
        default_output: Output for missing values and for values without an
            output when there is no palette.

    Raises:
        ValueError: If a key of `mapping` is a missing value. Use
            `default_output` for missing values.
    """

    def __init__(
        self,
        mapping: Mapping[Any, Any] | None = None,
        palette: Iterable[Any] | None = None,
        default_output: Any = None,
    ):
        self._mapping = {} if mapping is None else dict(mapping)
        self._check_no_missing_keys()
        self._palette = self._working_palette(palette, list(self._mapping.values()))
        self._default_output = default_output
        self._n_assigned = 0
        self._default_used = False

    @property
    def mapping(self) -> dict:
        """Output per input value: the fixed entries first, also the ones never
        mapped, then the palette assignments in the order of first use."""
        return self._mapping

    @property
    def default_output(self) -> Any:
        """Output for missing values and for values without an output."""
        return self._default_output

    @property
    def default_used(self) -> bool:
        """True once a call fell back to `default_output`: a missing value, or a
        value without an output when there is no palette."""
        return self._default_used

    def __call__(self, value: Any) -> Any:
        if is_missing(value):
            return self._use_default()
        if value in self._mapping:
            return self._mapping[value]
        if self._palette:
            return self._assign(value)
        return self._use_default()

    def _check_no_missing_keys(self) -> None:
        missing = [key for key in self._mapping if is_missing(key)]
        if missing:
            raise ValueError(
                f'Mapping keys {missing} are missing values and are never looked up. '
                'Use default_output for missing values.'
            )

    @staticmethod
    def _working_palette(palette: Iterable[Any] | None, fixed_outputs: list) -> list:
        given = [] if palette is None else list(palette)
        free = [output for output in given if output not in fixed_outputs]
        return free or given

    def _use_default(self) -> Any:
        self._default_used = True
        return self._default_output

    def _assign(self, value: Any) -> Any:
        output = self._palette[self._n_assigned % len(self._palette)]
        self._n_assigned += 1
        self._mapping[value] = output
        return output


class DiscreteColorMapping(DiscreteInputMapping):
    """Maps discrete input values to CSS colors.

    Examples:
        Fixed colors for two fuels, palette colors for the others. The mapping keeps
        its assignments, so one mapping gives the same colors in many frames, and a
        legend shows all colors that the frames used:

        >>> import folium
        >>> from mesqual.visualizations.folium_viz_system.legends import DiscreteColorLegend
        >>> colors = DiscreteColorMapping(
        ...     mapping={'solar': '#ffcc00', 'wind': '#3399ff'},
        ...     palette=['#1f77b4', '#ff7f0e', '#2ca02c'],
        ...     default_output='#cccccc',
        ... )
        >>> pd.Series(['solar', 'gas', None]).map(colors).tolist()
        ['#ffcc00', '#1f77b4', '#cccccc']
        >>> pd.Series(['coal', 'gas']).map(colors).tolist()
        ['#ff7f0e', '#1f77b4']
        >>> list(colors.mapping)
        ['solar', 'wind', 'gas', 'coal']
        >>> legend = DiscreteColorLegend(colors, title='Fuel').add_to(folium.Map())
    """


class DiscreteLineDashPatternMapping(DiscreteInputMapping):
    """Maps discrete input values to SVG dash patterns, such as `"5,5"`."""


class DiscreteLineWidthMapping(DiscreteInputMapping):
    """Maps discrete input values to line widths in px."""


class DiscreteIconMapping(DiscreteInputMapping):
    """Maps discrete input values to icons."""
