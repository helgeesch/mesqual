"""Values of two frames compared with a tolerance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, eq=False)
class Comparison:
    """Values of two frames compared where both have a value.

    Attributes:
        compared: Number of compared values.
        violations: One row per value off by more than the tolerance: the index
            levels and column levels of the frames as columns (see `long`), then
            `actual`, `expected` and `difference`.

    Examples:
        >>> actual = pd.DataFrame({'N1': [10.0, 20.0], 'N2': [30.0, 41.0]})
        >>> expected = pd.DataFrame({'N1': [10.0, 20.0], 'N2': [30.0, 40.0]})
        >>> comparison = Comparison.of(actual, expected, tol=0.5)
        >>> comparison.compared, len(comparison.violations)
        (4, 1)
    """

    compared: int
    violations: pd.DataFrame

    _VALUE_COLUMNS = ('actual', 'expected', 'difference')

    @classmethod
    def of(
        cls,
        actual: pd.DataFrame | pd.Series,
        expected: pd.DataFrame | pd.Series,
        tol: float = 0.0,
        rel_tol: float = 0.0,
        empty_is_off: bool = False,
    ) -> Comparison:
        """Compare `actual` with `expected` on the index and columns they share.

        A value fits when `|actual - expected| <= tol + rel_tol * max(|actual|, |expected|)`,
        as in `math.isclose`. A cell with NaN on either side is not compared,
        unless `empty_is_off`: then an empty actual cell with an expected value
        is a violation.

        The index levels and column levels of both sides are matched by name
        when all levels on both sides have unique names; `expected` takes the
        level order of `actual`. Else they are matched by position.

        Args:
            actual: The values to test.
            expected: The values they must equal.
            tol: Absolute tolerance.
            rel_tol: Relative tolerance, a share of the larger absolute value.
            empty_is_off: Count an empty actual cell with an expected value as a violation.

        Raises:
            ValueError: The level names of both sides differ (all levels named),
                or the number of levels differs (matched by position); a level
                is named `actual`, `expected` or `difference`; a side has a
                duplicate index entry.
        """
        index_levels = actual.index.nlevels
        actual, expected = cls._stacked(actual), cls._stacked(expected)
        expected = cls._aligned(actual, expected)
        actual = cls._named(actual, index_levels)
        expected = expected.set_axis(expected.index.set_names(actual.index.names))
        cls._check_levels(actual, 'actual')
        cls._check_levels(expected, 'expected')
        both = pd.concat({'actual': actual, 'expected': expected}, axis=1, join='inner')
        both = both[both['expected'].notna()] if empty_is_off else both.dropna()
        difference = both['actual'] - both['expected']
        scale = both[['actual', 'expected']].abs().max(axis=1)
        off = (difference.abs() > tol + rel_tol * scale) | both['actual'].isna()
        violations = both[off].assign(difference=difference[off]).reset_index()
        return cls(len(both), violations)

    @classmethod
    def combine(cls, parts: Mapping[str, Comparison], label: str) -> Comparison:
        """Join comparisons into one.

        Args:
            parts: The comparisons by key.
            label: Name of the first column of the violations, which holds the
                key of their part.

        Raises:
            ValueError: `label` is a column of the violations of a part.
        """
        if any(label in part.violations.columns for part in parts.values()):
            raise ValueError(f'label {label!r} is a column of the violations of a part.')
        frames = [part.violations.assign(**{label: key}) for key, part in parts.items() if len(part.violations)]
        violations = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if len(violations):
            violations = violations[[label, *violations.columns.drop(label)]]
        return cls(sum(part.compared for part in parts.values()), violations)

    @classmethod
    def long(cls, values: pd.DataFrame | pd.Series) -> pd.Series:
        """Return the values as one float series, with all column levels of a frame moved into the index.

        An unnamed index level gets the name `index` (`index_<i>` for level `i`
        of several), an unnamed column level `column` (`column_<i>`).
        """
        return cls._named(cls._stacked(values), values.index.nlevels)

    @staticmethod
    def _stacked(values: pd.DataFrame | pd.Series) -> pd.Series:
        if isinstance(values, pd.DataFrame):
            values = values.stack(list(range(values.columns.nlevels)), future_stack=True)
        return values.astype('float64')

    @staticmethod
    def _named(values: pd.Series, index_levels: int) -> pd.Series:
        """Name the unnamed levels: the first `index_levels` as index levels, the others as column levels."""
        index_names = list(values.index.names)
        groups = [('index', index_names[:index_levels]), ('column', index_names[index_levels:])]
        names = [
            name if name is not None else (stem if len(group) == 1 else f'{stem}_{i}')
            for stem, group in groups
            for i, name in enumerate(group)
        ]
        return values.set_axis(values.index.set_names(names))

    @staticmethod
    def _aligned(actual: pd.Series, expected: pd.Series) -> pd.Series:
        names_a, names_e = list(actual.index.names), list(expected.index.names)
        by_name = all(None not in names and len(set(names)) == len(names) for names in (names_a, names_e))
        if by_name:
            if set(names_a) != set(names_e):
                raise ValueError(f'The level names differ: actual {names_a}, expected {names_e}.')
            return expected.reorder_levels(names_a) if names_a != names_e else expected
        if len(names_a) != len(names_e):
            raise ValueError(f'actual has {len(names_a)} index and column levels, expected {len(names_e)}.')
        return expected

    @classmethod
    def _check_levels(cls, values: pd.Series, side: str) -> None:
        reserved = set(cls._VALUE_COLUMNS) & set(values.index.names)
        if reserved:
            raise ValueError(f'{side} has levels named {sorted(reserved)}; rename them.')
        if values.index.has_duplicates:
            raise ValueError(f'{side} has duplicate index entries.')
