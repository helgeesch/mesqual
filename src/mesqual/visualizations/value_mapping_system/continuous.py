from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import colors

from mesqual.visualizations.value_mapping_system.base import BaseMapping, is_missing


class SegmentedContinuousInputMappingBase(BaseMapping):
    """Base class of the segmented continuous mappings, with the call rules.

    Use a subclass, such as `SegmentedContinuousColorscale` or
    `SegmentedContinuousLineWidthMapping`, or subclass it for a new output type.
    Each segment is a closed input range `(start, end)` with an output:

    - One output, such as `5.0` or `"#ff0000"`: every value in the segment
      gets it.
    - A list of outputs: the outputs sit at equal steps from `start` to `end`,
      and a value between two of them gets the linear interpolation.

    A call returns the output for one value:

    1. A missing value (see `is_missing`) gets `default_output`.
    2. A value that `float()` cannot convert gets `default_output`.
    3. A value below the first segment or above the last segment gets the output
       at that edge.
    4. A value in a gap between two segments gets `default_output`.
    5. A numeric output is clipped to `min_output_value` and `max_output_value`,
       also a numeric `default_output`: with `default_output=0.5` and
       `min_output_value=2`, a missing value gives 2.

    Args:
        segments: Output per input range `(start, end)`. The mapping sorts the
            segments by range.
        default_output: Output for missing values and for values without an output.
        min_output_value: Lower bound of numeric outputs. `None`: no bound.
        max_output_value: Upper bound of numeric outputs. `None`: no bound.

    Raises:
        ValueError: If `segments` is empty, two segments overlap, or a segment
            has `end <= start`.
    """

    def __init__(
        self,
        segments: dict[tuple[float | int, float | int], Any],
        default_output: Any = None,
        min_output_value: float | int = None,
        max_output_value: float | int = None,
    ):
        self._default_output = default_output
        self._min_output_value = min_output_value
        self._max_output_value = max_output_value
        self._segments = dict(sorted(segments.items()))
        self._validate_segments()
        ranges = list(self._segments)
        self._input_value_range = (float(ranges[0][0]), float(ranges[-1][1]))

    @property
    def default_output(self) -> Any:
        """Output for missing values and for values without an output."""
        return self._default_output

    def __call__(self, value: float):
        if is_missing(value):
            result = self._default_output
        else:
            try:
                result = self._compute_output(float(value))
            except (ValueError, TypeError):
                result = self._default_output

        # in case of numeric output
        if isinstance(result, (float, int)):
            if self._min_output_value:
                result = max(result, self._min_output_value)
            if self._max_output_value:
                result = min(result, self._max_output_value)
        return result

    @staticmethod
    def _get_low_high_from_values(values, trim_percentile) -> tuple[float, float]:
        arr = np.asarray(values)
        arr = arr[~pd.isna(arr)].astype(float)
        if arr.size == 0:
            raise ValueError('Cannot autoscale: the values are empty or all missing.')
        if trim_percentile:
            # A percentile next to an infinite value is NaN; _widened_finite_range raises for it.
            with np.errstate(invalid='ignore'):
                low, high = (float(v) for v in np.percentile(arr, trim_percentile))
        else:
            low, high = float(np.min(arr)), float(np.max(arr))
        return low, high

    @staticmethod
    def _widened_finite_range(low: float, high: float) -> tuple[float, float]:
        """Return `(low, high)`, widened if it is one value. Raise `ValueError` if a bound or the width is not finite."""
        if low == high:
            half_width = abs(low) * 0.1 or 1.0
            low, high = low - half_width, high + half_width
        if not np.isfinite([low, high, high - low]).all():
            raise ValueError(
                f'Cannot autoscale: the range ({low}, {high}) or its width is not finite. '
                'An infinite or very large value in the range causes this.'
            )
        return low, high

    @property
    def segments(self) -> dict[tuple[float | int, float | int], Any]:
        """Output per input range `(start, end)`, sorted by range."""
        return self._segments

    def _validate_segments(self):
        if not self._segments:
            raise ValueError('A continuous mapping needs at least one segment.')
        prev_end = -float('inf')
        for (start, end), _ in self._segments.items():
            if start < prev_end:
                raise ValueError(f'Overlapping segments: ({start}, {end})')
            if end <= start:
                raise ValueError(f'Invalid segment: ({start}, {end})')
            prev_end = end

    def _compute_output(self, value: float):
        clipped_value = np.clip(value, self.min_input_value, self.max_input_value)
        for (start, end), output in self._segments.items():
            if start <= clipped_value <= end:
                if isinstance(output, list):  # continuous output --> interpolate
                    return self._interpolate(clipped_value, start, end, output)
                return output
        return self._default_output

    def _interpolate(self, value: float, start: float, end: float, outputs: list[float]) -> float:
        if len(outputs) == 1:
            return outputs[0]
        pos = (value - start) / (end - start)
        idx = pos * (len(outputs) - 1)
        idx_low, idx_high = int(np.floor(idx)), int(np.ceil(idx))
        if idx_low == idx_high:
            return outputs[idx_low]
        frac = idx - idx_low
        return outputs[idx_low] + frac * (outputs[idx_high] - outputs[idx_low])

    @property
    def min_input_value(self) -> float:
        """Start of the first segment."""
        return self._input_value_range[0]

    @property
    def max_input_value(self) -> float:
        """End of the last segment."""
        return self._input_value_range[1]

    @property
    def input_value_range(self) -> tuple[float, float]:
        """Start of the first segment and end of the last segment."""
        return self._input_value_range


class SegmentedContinuousColorscale(SegmentedContinuousInputMappingBase):
    """Maps numbers to CSS colors, segment by segment.

    The output of a segment is a list of colors, one color string, or the name of
    a matplotlib colormap, such as `"viridis"`. A string that is a colormap name
    is a colormap, also when it is a color too, such as `"gray"`; any other
    string is one color. A colormap name gives 100 colors of the colormap. A call
    returns a hex color, interpolated in RGBA within the segment, or
    `default_output`. The rules of a call are in
    `SegmentedContinuousInputMappingBase`.

    Args:
        segments: Colors per input range `(start, end)`. The mapping does not
            change the given dict.
        default_output: Color for missing values and for values without a color.

    Raises:
        ValueError: If a string is neither a colormap name nor a color,
            `segments` is empty, two segments overlap, or a segment has
            `end <= start`.

    Examples:
        A diverging scale for prices, gray for missing prices. A value outside all
        segments gets the edge color, here the last color of YlOrRd:

        >>> from mesqual.visualizations.value_mapping_system import SegmentedContinuousColorscale
        >>> prices = SegmentedContinuousColorscale(
        ...     segments={(-50, 0): ['#2166ac', '#f7f7f7'], (0, 200): 'YlOrRd'},
        ...     default_output='#cccccc',
        ... )
        >>> prices(-50), prices(500), prices(None)
        ('#2166ac', '#800026', '#cccccc')
    """

    def __init__(
        self,
        segments: dict[tuple[float | int, float | int], Any],
        default_output: Any = None,
    ):

        super().__init__(
            segments=self._convert_recognized_colorscale_strings_to_list_of_colors(segments),
            default_output=default_output,
        )

    @staticmethod
    def _convert_recognized_colorscale_strings_to_list_of_colors(segments: dict) -> dict:
        return {
            key: SegmentedContinuousColorscale._string_to_colors(val) if isinstance(val, str) else val
            for key, val in segments.items()
        }

    @staticmethod
    def _string_to_colors(name: str) -> list[str]:
        if name in matplotlib.colormaps:
            cmap = matplotlib.colormaps[name]
            return [colors.to_hex(cmap(i)) for i in np.linspace(0, 1, 100)]
        if colors.is_color_like(name):
            return [name]
        raise ValueError(f"'{name}' is neither a valid colormap nor a valid color string.")

    @classmethod
    def single_segment_autoscale_factory_from_array(
        cls,
        values: np.ndarray | list,
        colorscale: list[str] | str = 'viridis',
        trim_percentile: tuple[float, float] = None,
        continuous_midpoint: int | float = None,
        default_output: Any = None,
    ) -> 'SegmentedContinuousColorscale':
        """Return a colorscale with one segment from the range of `values`.

        Args:
            values: Values that set the range. Missing values are ignored. If
                the range is one value after the midpoint step, such as for a
                constant array without a midpoint or equal to the midpoint, it
                grows by 10 % of the value on each side, or by 1 for the value 0.
            colorscale: Colors, one color, or a matplotlib colormap name.
            trim_percentile: Lower and upper percentile of `values` for the range,
                such as `(5, 95)`, so outliers do not stretch the scale. `None`:
                the minimum and maximum.
            continuous_midpoint: Value at the middle of the scale. The range grows
                to the same distance on both sides of it, for a diverging
                colorscale. `None`: no midpoint.
            default_output: Color for missing values and for values without a color.

        Returns:
            Colorscale with one segment.

        Raises:
            ValueError: If `values` is empty, all values are missing, or the
                range or its width is not finite, from an infinite value in
                the range or a value near the float limit.
        """
        low, high = cls._get_low_high_from_values(values, trim_percentile)

        if continuous_midpoint is not None:
            dist_low = abs(low - continuous_midpoint)
            dist_high = abs(high - continuous_midpoint)
            max_dist = max(dist_low, dist_high)
            low = continuous_midpoint - max_dist
            high = continuous_midpoint + max_dist

        low, high = cls._widened_finite_range(low, high)
        segments = {(low, high): colorscale}
        return cls(segments=segments, default_output=default_output)

    def _interpolate(self, value: float, start: float, end: float, outputs: list[str]) -> str:
        rgba_colors = list(map(colors.to_rgba, outputs))
        positions = np.linspace(0, 1, len(rgba_colors))

        def color_fn(v: float) -> str:
            t = (np.clip(v, start, end) - start) / (end - start)
            r, g, b, a = np.transpose([np.interp(t, positions, channel) for channel in np.transpose(rgba_colors)])
            return colors.to_hex((r, g, b, a))

        return color_fn(value)

    def to_normalized_colorscale(self, num_reference_points_per_segment: int = 10) -> List[Tuple[float, str]]:
        """Return the colorscale as a plotly colorscale.

        The positions run from 0 at `min_input_value` to 1 at `max_input_value`.
        Each segment gets `num_reference_points_per_segment` points. The end of
        a segment moves by 1e-9 before the next segment, so a color step at a
        segment boundary stays sharp.

        Args:
            num_reference_points_per_segment: Number of points per segment,
                with its start and end.

        Returns:
            `(position, color)` pairs sorted by position.

        Raises:
            ValueError: If `num_reference_points_per_segment` is less than 2.
        """
        if num_reference_points_per_segment < 2:
            raise ValueError('num_reference_points_per_segment must be >= 2')
        total_range = self.max_input_value - self.min_input_value
        segment_items = list(self.segments.items())
        raw_scale: List[Tuple[float, str]] = []
        for idx, ((start, end), _) in enumerate(segment_items):
            norm_start = (start - self.min_input_value) / total_range
            norm_end = (end - self.min_input_value) / total_range
            if idx < len(segment_items) - 1:
                norm_end -= 1e-9
            positions = np.linspace(norm_start, norm_end, num_reference_points_per_segment)
            for pos in positions:
                value = self.min_input_value + pos * total_range
                raw_scale.append((round(pos, 10), self(value)))
        final_pos = (self.max_input_value - self.min_input_value) / total_range
        raw_scale.append((round(final_pos, 10), self(self.max_input_value)))
        seen = set()
        normalized_scale: List[Tuple[float, str]] = []
        for pos, color in sorted(raw_scale, key=lambda x: x[0]):
            if pos in seen:
                continue
            seen.add(pos)
            normalized_scale.append((pos, color))
        return normalized_scale


class SegmentedContinuousInputToContinuousOutputMapping(SegmentedContinuousInputMappingBase):
    """Maps numbers to numbers, segment by segment.

    The output of a segment is one number, or a list of numbers to interpolate.
    The rules of a call and the arguments are in
    `SegmentedContinuousInputMappingBase`.

    Examples:
        Widths from 1 to 5 for values from 0 to 100, width 0.5 for missing values:

        >>> widths = SegmentedContinuousInputToContinuousOutputMapping(segments={(0, 100): [1.0, 5.0]}, default_output=0.5)
        >>> print(widths(50), widths(250), widths(None))
        3.0 5.0 0.5
    """

    @classmethod
    def single_segment_autoscale_factory_from_array(
        cls,
        values: np.ndarray | list,
        output_range: tuple[int | float, int | float],
        trim_percentile: tuple[float, float] = None,
        default_output: Any = None,
    ) -> 'SegmentedContinuousInputToContinuousOutputMapping':
        """Return a mapping with one segment from the range of `values`.

        Args:
            values: Values that set the range. Missing values are ignored. If
                the range is one value, such as for a constant array, it grows
                by 10 % of the value on each side, or by 1 for the value 0.
            output_range: Output at the start and at the end of the range.
            trim_percentile: Lower and upper percentile of `values` for the range,
                such as `(5, 95)`, so outliers do not stretch the scale. `None`:
                the minimum and maximum.
            default_output: Output for missing values and for values without an output.

        Returns:
            Mapping with one segment.

        Raises:
            ValueError: If `values` is empty, all values are missing, or the
                range or its width is not finite, from an infinite value in
                the range or a value near the float limit.
        """
        low, high = cls._widened_finite_range(*cls._get_low_high_from_values(values, trim_percentile))
        segments = {(low, high): [output_range[0], output_range[1]]}
        return cls(segments=segments, default_output=default_output)


class SegmentedContinuousLineWidthMapping(SegmentedContinuousInputToContinuousOutputMapping):
    """Maps numbers to line widths in px.

    Examples:
        Width 1 for small flows, 2 to 6 for large flows:

        >>> widths = SegmentedContinuousLineWidthMapping(
        ...     segments={(0, 500): 1.0, (500, 2000): [2.0, 6.0]},
        ...     default_output=0.5,
        ... )
        >>> print(widths(250), widths(1250), widths(5000), widths(None))
        1.0 4.0 6.0 0.5
    """


class SegmentedContinuousOpacityMapping(SegmentedContinuousInputToContinuousOutputMapping):
    """Maps numbers to opacities from 0 to 1.

    Examples:
        Opacity from 0.2 to 1 for values from 0 to 100, hidden for missing values:

        >>> opacity = SegmentedContinuousOpacityMapping(segments={(0, 100): [0.2, 1.0]}, default_output=0.0)
        >>> print(opacity(0), opacity(250), opacity(float('nan')))
        0.2 1.0 0.0
    """
