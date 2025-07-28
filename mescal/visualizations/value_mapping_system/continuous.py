from abc import abstractmethod, ABC
from typing import Any, Iterable

import numpy as np
import pandas as pd
from matplotlib import cm, colors

from mescal.visualizations.value_mapping_system.base import BaseMapping


class _ContinuousInputMapping(BaseMapping):
    """Base class for handling a continuous input. Can handle continuous or discrete outputs"""
    def __init__(
            self,
            nan_fallback: Any = None,
            min_output_value: float | int = None,  # only available for continuous outputs
            max_output_value: float | int = None,  # only available for continuous outputs
    ):
        self._nan_fallback = nan_fallback
        self._min_output_value = min_output_value
        self._max_output_value = max_output_value

    @property
    def nan_fallback(self) -> Any:
        return self._nan_fallback

    @abstractmethod
    def _compute_output(self, value: float):
        pass

    def __call__(self, value: float):
        if pd.isna(value) or value is None:
            result = self._nan_fallback
        else:
            try:
                result = self._compute_output(float(value))
            except (ValueError, TypeError):
                result = self._nan_fallback

        # in case of numeric output
        if isinstance(result, (float, int)):
            if self._min_output_value:
                result = max(result, self._min_output_value)
            if self._max_output_value:
                result = min(result, self._max_output_value)
        return result

    @staticmethod
    def _get_low_high_from_values(values, trim_percentile):
        arr = np.asarray(values)
        arr = arr[~pd.isna(arr)]
        if trim_percentile:
            low, high = np.percentile(arr, trim_percentile)
        else:
            low, high = float(np.min(arr)), float(np.max(arr))
        return low, high


class SegmentedContinuousInputMappingBase(_ContinuousInputMapping, ABC):
    """Continuous Input --> Continuous or discrete Output"""
    def __init__(
            self,
            segments: dict[tuple[float | int, float | int], Any],
            nan_fallback: Any = None,
            min_output_value: float | int = None,  # only available for continuous outputs
            max_output_value: float | int = None,  # only available for continuous outputs
    ):
        super().__init__(
            nan_fallback=nan_fallback,
            min_output_value=min_output_value,
            max_output_value=max_output_value
        )
        self._segments = dict(sorted(segments.items()))
        self._min_input_value = None
        self._max_input_value = None
        self._validate_segments()

    @property
    def segments(self) -> dict[tuple[float | int, float | int], Any]:
        return self._segments

    def _validate_segments(self):
        prev_end = -float('inf')
        for (start, end), _ in self._segments.items():
            if start < prev_end:
                raise ValueError(f"Overlapping segments: ({start}, {end})")
            if end <= start:
                raise ValueError(f"Invalid segment: ({start}, {end})")
            prev_end = end

    def _compute_output(self, value: float):
        clipped_value = np.clip(value, self.min_input_value, self.max_input_value)
        for (start, end), output in self._segments.items():
            if start <= clipped_value <= end:
                if isinstance(output, list):  # continuous output --> interpolate
                    return self._interpolate(clipped_value, start, end, output)
                return output
        return self._nan_fallback

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
        if self._min_input_value is None:
            self._min_input_value = float(np.min(self._get_all_values()))
        return self._min_input_value

    @property
    def max_input_value(self) -> float:
        if self._max_input_value is None:
            self._max_input_value = float(np.max(self._get_all_values()))
        return self._max_input_value

    def _get_all_values(self) -> list[float | int]:
        values = []
        for val in self._segments.keys():
            if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
                values.extend(val)
            else:
                values.append(val)
        return values

    @classmethod
    @abstractmethod
    def single_segment_autoscale_factory_from_array(cls, **kwargs) -> 'SegmentedContinuousInputMappingBase':
        pass


class SegmentedContinuousColorscale(SegmentedContinuousInputMappingBase):
    def __init__(
            self,
            segments: dict[tuple[float | int, float | int], Any],
            nan_fallback: Any = None,
    ):

        super().__init__(
            segments=self._convert_recognized_colorscale_strings_to_list_of_colors(segments),
            nan_fallback=nan_fallback,
        )

    @staticmethod
    def _convert_recognized_colorscale_strings_to_list_of_colors(segments: dict) -> dict:
        for key, val in list(segments.items()):
            if isinstance(val, str):
                try:
                    cmap = cm.get_cmap(val)
                    segments[key] = [colors.to_hex(cmap(i)) for i in np.linspace(0, 1, 100)]
                except ValueError:
                    try:
                        colors.to_rgba(val)
                        segments[key] = [val]
                    except ValueError:
                        raise ValueError(f"'{val}' is neither a valid colormap nor a valid color string.")
        return segments

    @classmethod
    def single_segment_autoscale_factory_from_array(
            cls,
            values: np.ndarray | list,
            colorscale: list[str] | str = 'viridis',
            trim_percentile: tuple[float, float] = None,
            continuous_midpoint: int | float = None,
            nan_fallback: Any = None,
            **kwargs
    ) -> 'SegmentedContinuousColorscale':
        low, high = cls._get_low_high_from_values(values, trim_percentile)

        if continuous_midpoint is not None:
            dist_low = abs(low - continuous_midpoint)
            dist_high = abs(high - continuous_midpoint)
            max_dist = max(dist_low, dist_high)
            low = continuous_midpoint - max_dist
            high = continuous_midpoint + max_dist

        segments = {(low, high): colorscale}
        return cls(segments=segments, nan_fallback=nan_fallback)

    def _interpolate(self, value: float, start: float, end: float, outputs: list[str]) -> str:
        from matplotlib import colors
        rgba_colors = list(map(colors.to_rgba, outputs))
        positions = np.linspace(0, 1, len(rgba_colors))

        def color_fn(v: float) -> str:
            t = (np.clip(v, start, end) - start) / (end - start)
            r, g, b, a = np.transpose([np.interp(t, positions, channel) for channel in np.transpose(rgba_colors)])
            return colors.to_hex((r, g, b, a))

        return color_fn(value)


class SegmentedContinuousInputToContinuousOutputMapping(SegmentedContinuousInputMappingBase):
    @classmethod
    def single_segment_autoscale_factory_from_array(
            cls,
            values: np.ndarray | list,
            output_range: tuple[int | float, int | float],
            trim_percentile: tuple[float, float] = None,
            nan_fallback: Any = None,
            **kwargs
    ) -> 'SegmentedContinuousInputToContinuousOutputMapping':
        low, high = cls._get_low_high_from_values(values, trim_percentile)

        segments = {(low, high): [output_range[0], output_range[1]]}
        return cls(segments=segments, nan_fallback=nan_fallback)


class SegmentedContinuousLineWidthMapping(SegmentedContinuousInputToContinuousOutputMapping):
    pass


class SegmentedContinuousOpacityMapping(SegmentedContinuousInputToContinuousOutputMapping):
    pass
