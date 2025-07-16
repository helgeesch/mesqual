import numpy as np
from collections.abc import Iterable


class SegmentedValueInterpolator:
    def __init__(self, segments: dict[tuple[float, float], float | list[float]], nan_value: float = np.nan) -> None:
        self.segments = dict(sorted(segments.items()))
        self.nan_value = nan_value
        self._min_input_value = None
        self._max_input_value = None
        self._validate_segments()

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
        for val in self.segments.values():
            if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
                values.extend(val)
            else:
                values.append(val)
        return values

    def _validate_segments(self) -> None:
        prev_end = -float('inf')
        for key, val in list(self.segments.items()):
            start, end = key
            if start < prev_end:
                raise ValueError(
                    f"Overlapping segments detected: ({start}, {end}) "
                    f"overlaps with previous segment ending at {prev_end}"
                )
            prev_end = end

            if end <= start:
                raise ValueError(f"Invalid segment ({start}, {end})")

            if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
                val = list(val)
                if len(val) == 1:
                    self.segments[key] = val[0]  # treat as constant
                elif len(val) == 2:
                    self.segments[key] = val     # keep as interpolation
                else:
                    raise ValueError(f"Segment {key} has invalid list length: {val}")

    def __call__(self, value: float | int | np.ndarray | list[float | int]) -> float:
        if isinstance(value, (list, np.ndarray)):
            return self._transform_array(value)
        return self._transform_value(value)

    def _transform_array(self, values: np.ndarray | list[float | int]) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        result = np.full_like(values, self.nan_value, dtype=np.float64)
        for i, v in enumerate(values):
            result[i] = self._transform_value(v)
        return result

    def _transform_value(self, value: float | int) -> float:
        if np.isnan(value):
            return self.nan_value

        value = np.clip(value, a_min=self.min_input_value, a_max=self.max_input_value)

        for (start, end), val in self.segments.items():
            if start <= value < end or (value == end and end == max(k[1] for k in self.segments)):
                if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
                    return float(val[0] + (value - start) * (val[1] - val[0]) / (end - start))
                return float(val)
        return self.nan_value

    @classmethod
    def from_points(cls, x: list[float], y: list[float], nan_value: float = np.nan) -> 'SegmentedValueInterpolator':
        if len(x) != len(y):
            raise ValueError("x and y must be of equal length")
        if len(x) < 2:
            raise ValueError("At least two points are required to form segments")

        segments: dict[tuple[float, float], float | list[float]] = {}

        for i in range(len(x) - 1):
            start, end = x[i], x[i + 1]
            y0, y1 = y[i], y[i + 1]
            if y0 == y1:
                segments[(start, end)] = y0
            else:
                segments[(start, end)] = [y0, y1]

        return cls(segments, nan_value=nan_value)


if __name__ == '__main__':
    test_segments = {
        (0, 10): 1.0,
        (10, 20): 3.0,
        (20, 30): [3.0, 5.0],
        (30, 50): [8.0],
    }

    interp = SegmentedValueInterpolator(test_segments, nan_value=-1.0)

    print(interp(25))  # → 4.0 (interpolated)
    print(interp(10))  # → 3.0
    print(interp(100))  # → -1.0 (outside)
    print(interp(np.nan))  # → -1.0

    print(interp(np.array([5, 15, 25, 35, 100, np.nan])))
    # → [1. 3. 4. 8. -1. -1.]
