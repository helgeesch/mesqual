"""A check of bounds on the values of one flag."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mesqual.datasets import Dataset
from mesqual.typevars import FlagType
from mesqual.validation.check import DatasetCheck
from mesqual.validation.comparison import Comparison
from mesqual.validation.result import CheckResult, CheckStatus


class ConstraintCheck(DatasetCheck):
    """Checks that the values of one flag are within bounds.

    The check fetches the flag and tests each value: it must be at least
    `min_value` and at most `max_value`. For an exact value, use `equal`.
    An empty value (NaN) is a violation unless `isna_ok`; it is not tested
    against the bounds. Without values to test, the result is `SKIP`.

    A bound is a number, a `Series` or a `DataFrame`:

    - A number applies to every value.
    - A `DataFrame` aligns to the flag data by index and columns.
    - A `Series` on a frame flag applies per object when it matches the
      columns, per timestep when it matches the index. It has all levels of
      that axis; a bound on one level of a `MultiIndex` axis is not supported.
      The level names decide first: the bound index names equal the column
      level names (per object) or the index names (per timestep). Else the
      labels decide: the bound labels overlap the columns and not the index
      (per object), or the other way round (per timestep). Extra labels are
      allowed.
    - A `Series` on a series flag aligns to its index.

    A value without a bound (its label is not in the bound, or the bound is
    NaN) is not tested against that bound. A `Series` or `DataFrame` bound that
    covers no value of the data, for example a tz-naive bound on a tz-aware
    index, raises `ValueError` in the run. So do a `Series` bound that matches
    neither axis or both, an empty or a duplicate bound label, a `DataFrame`
    bound on a series flag, a data level named `value`, `min_value` or
    `max_value`, and an index level and a column level with the same name. The
    validator reports these as `ERROR`.

    `FAIL` lists one row per violation: the index levels and column levels of
    the flag data, `value`, then `min_value` and `max_value` for the bounds
    that are set.

    Examples:
        >>> loading = ConstraintCheck('Line.Results.loading', min_value=0, max_value=1, isna_ok=False)
        >>> generation = ConstraintCheck(  # per object: a Series by generator id
        ...     'Generator.Results.generation', min_value=0, max_value=dataset.fetch('Generator.Model')['capacity'],
        ... )
        >>> flow = ConstraintCheck(  # per value: a DataFrame like the flag data
        ...     'Line.Results.flow_sent_up', max_value=dataset.fetch('Line.Constraints.capacity_up'),
        ... )
        >>> [check.run(dataset).status.name for check in (loading, generation, flow)]
        ['PASS', 'PASS', 'PASS']
        >>> generation.run(dataset).summary
        '1660 values of Generator.Results.generation meet the rule >= 0, <= max_value per object, empty allowed.'

    Args:
        flag: The flag to check.
        min_value: Lowest allowed value, or `None`.
        max_value: Highest allowed value, or `None`.
        isna_ok: An empty value is not a violation.
        object_subset: Columns of a frame flag to check, for example line ids.
            `None` checks all columns. A column that the data does not have
            raises `KeyError` in the run, and a flag that gives a `Series`
            raises `ValueError`; the validator reports both as `ERROR`.
        name: The check name. Default: `ConstraintCheck[<flag>]`.

    Raises:
        ValueError: `object_subset` is empty, or a bound is NaN or a bool.

    Attributes:
        flag: The flag to check.
        min_value: Lowest allowed value, or `None`.
        max_value: Highest allowed value, or `None`.
        isna_ok: An empty value is not a violation.
        object_subset: Columns of the flag data to check, or `None` for all.
    """

    def __init__(
        self,
        flag: FlagType,
        min_value: float | pd.Series | pd.DataFrame | None = None,
        max_value: float | pd.Series | pd.DataFrame | None = None,
        isna_ok: bool = True,
        object_subset: list[int | str] | None = None,
        name: str | None = None,
    ):
        if object_subset is not None and not len(object_subset):
            raise ValueError('object_subset is empty; use None to check all columns.')
        for label, bound in (('min_value', min_value), ('max_value', max_value)):
            self._check_scalar(label, bound)
        self.flag = flag
        self.min_value = min_value
        self.max_value = max_value
        self.isna_ok = isna_ok
        self.object_subset = object_subset
        self._name = name

    @classmethod
    def equal(
        cls,
        flag: FlagType,
        value: float | pd.Series | pd.DataFrame,
        *,
        isna_ok: bool = True,
        object_subset: list[int | str] | None = None,
        name: str | None = None,
    ) -> ConstraintCheck:
        """Return a check that the values of `flag` equal `value`: `min_value` and `max_value` are both `value`.

        `value` is a number, a `Series` or a `DataFrame`, aligned as a bound
        (see the class docstring). The other arguments are those of the
        constructor. Example: `ConstraintCheck.equal('Line.Results.flow_sent_up', capacities, isna_ok=False)`.
        """
        return cls(flag, value, value, isna_ok=isna_ok, object_subset=object_subset, name=name)

    @property
    def name(self) -> str:
        """The check name: the name given, else `ConstraintCheck[<flag>]`."""
        return self._name or f'{type(self).__name__}[{self.flag}]'

    @property
    def flags(self) -> tuple[FlagType, ...]:
        """The flags the check reads: `flag`."""
        return (self.flag,)

    def _run(self, dataset: Dataset) -> CheckResult:
        data = dataset.fetch(self.flag)
        if self.object_subset is not None:
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f'{self.flag} gives a {type(data).__name__}; object_subset selects frame columns.')
            data = data[self.object_subset]
        values = Comparison.long(data)
        self._check_levels(values)
        what = f'values of {self.flag}{self._subset_text()}'
        if values.empty:
            return self.skip(f'no {what}.')
        bounds = {label: self._aligned(label, bound, data) for label, bound in self._bounds().items()}
        broken = self._broken(values, {label: series for label, (_, series) in bounds.items()})
        rule = self._rule_text({label: form for label, (form, _) in bounds.items()})
        if broken.any():
            details = pd.concat(
                {'value': values[broken], **{label: series[broken] for label, (_, series) in bounds.items()}},
                axis=1,
            ).reset_index()
            summary = f'{int(broken.sum())} of {len(values)} {what} break the rule {rule}.'
            return self.result(CheckStatus.FAIL, summary, details)
        return self.result(CheckStatus.PASS, f'{len(values)} {what} meet the rule {rule}.')

    def _broken(self, values: pd.Series, bounds: dict[str, pd.Series]) -> pd.Series:
        empty = values.isna()
        off = pd.Series(False, index=values.index)
        if 'min_value' in bounds:
            off |= values < bounds['min_value']
        if 'max_value' in bounds:
            off |= values > bounds['max_value']
        return (off & ~empty) | (empty & (not self.isna_ok))

    def _bounds(self) -> dict[str, float | pd.Series | pd.DataFrame]:
        bounds = {'min_value': self.min_value, 'max_value': self.max_value}
        return {label: bound for label, bound in bounds.items() if bound is not None}

    @staticmethod
    def _check_scalar(label: str, bound: float | pd.Series | pd.DataFrame | None) -> None:
        if bound is None or isinstance(bound, (pd.Series, pd.DataFrame)):
            return
        if isinstance(bound, (bool, np.bool_)):
            raise ValueError(f'{label} is a bool; give a number.')
        if pd.isna(bound):
            raise ValueError(f'{label} is NaN; give a number or None.')

    @staticmethod
    def _check_levels(values: pd.Series) -> None:
        names = list(values.index.names)
        reserved = {'value', 'min_value', 'max_value'} & set(names)
        if reserved:
            raise ValueError(f'The flag data has levels named {sorted(reserved)}; rename them.')
        if len(set(names)) != len(names):
            raise ValueError(f'The flag data has index and column levels with the same name: {names}.')

    def _aligned(
        self,
        label: str,
        bound: float | pd.Series | pd.DataFrame,
        data: pd.DataFrame | pd.Series,
    ) -> tuple[str, pd.Series]:
        """Return the form of the bound and the bound as a long series aligned to the values of the data."""
        form = self._form(bound, data)
        shaped = Comparison.long(self._shaped(bound, data, form))
        if form != 'scalar' and shaped.isna().all():
            raise ValueError(f'{label} covers no value of {self.flag}; check its labels, level names and time zone.')
        return form, shaped

    @staticmethod
    def _shaped(
        bound: float | pd.Series | pd.DataFrame,
        data: pd.DataFrame | pd.Series,
        form: str,
    ) -> pd.DataFrame | pd.Series:
        """Return the bound with the index and columns of the data; NaN where it has no value."""
        if form == 'scalar':
            if isinstance(data, pd.DataFrame):
                return pd.DataFrame(bound, index=data.index, columns=data.columns)
            return pd.Series(bound, index=data.index)
        if form == 'per value':
            if isinstance(bound, pd.DataFrame):
                return bound.reindex(index=data.index, columns=data.columns)
            return bound.reindex(data.index)
        if form == 'per object':
            row = bound.reindex(data.columns).to_numpy(dtype='float64')
            return pd.DataFrame(np.broadcast_to(row, data.shape), index=data.index, columns=data.columns)
        column = bound.reindex(data.index).to_numpy(dtype='float64')
        return pd.DataFrame(np.broadcast_to(column[:, None], data.shape), index=data.index, columns=data.columns)

    @classmethod
    def _form(cls, bound: float | pd.Series | pd.DataFrame, data: pd.DataFrame | pd.Series) -> str:
        """`'scalar'`, `'per value'`, `'per object'` or `'per timestep'`.

        Raises:
            ValueError: The bound does not fit the data (see the class docstring).
        """
        if not isinstance(bound, (pd.Series, pd.DataFrame)):
            return 'scalar'
        if bound.empty:
            raise ValueError('A Series or DataFrame bound is empty.')
        axes = [bound.index] + ([bound.columns] if isinstance(bound, pd.DataFrame) else [])
        if any(axis.has_duplicates for axis in axes):
            raise ValueError('A bound has duplicate labels.')
        if isinstance(bound, pd.DataFrame):
            if not isinstance(data, pd.DataFrame):
                raise ValueError('A DataFrame bound needs a flag that gives a DataFrame.')
            return 'per value'
        if isinstance(data, pd.Series):
            return 'per value'
        return cls._series_form(bound, data)

    @classmethod
    def _series_form(cls, bound: pd.Series, data: pd.DataFrame) -> str:
        axes = {'per object': data.columns, 'per timestep': data.index}
        names = list(bound.index.names)
        by_name = [form for form, axis in axes.items() if None not in names and names == list(axis.names)]
        if len(by_name) == 1:
            return by_name[0]
        overlaps = {
            form: axis.nlevels == bound.index.nlevels and bool(bound.index.isin(axis).any())
            for form, axis in axes.items()
        }
        by_label = [form for form, overlap in overlaps.items() if overlap]
        if len(by_label) == 1:
            return by_label[0]
        on_one_level = any(cls._on_some_levels(bound.index, axis) for axis in axes.values())
        if on_one_level or bound.index.nlevels not in {axis.nlevels for axis in axes.values()}:
            raise ValueError(
                f'A Series bound has {bound.index.nlevels} index levels; it needs all levels of the columns '
                f'({data.columns.nlevels}) or of the index ({data.index.nlevels}). '
                'A bound on one level of a MultiIndex axis is not supported.'
            )
        raise ValueError(
            'A Series bound must match the columns (per object) or the index (per timestep) of the flag data, '
            'by level names or by labels, and not both.'
        )

    @staticmethod
    def _on_some_levels(labels: pd.Index, axis: pd.Index) -> bool:
        """`True` if `labels` has fewer levels than `axis` and matches some of its levels by name or by labels."""
        if labels.nlevels >= axis.nlevels:
            return False
        if set(labels.names) & set(axis.names) - {None}:
            return True
        return labels.nlevels == 1 and any(
            labels.isin(axis.get_level_values(level)).any() for level in range(axis.nlevels)
        )

    def _rule_text(self, forms: dict[str, str]) -> str:
        texts = {
            label: str(getattr(self, label)) if form == 'scalar' else f'{label} {form}' for label, form in forms.items()
        }
        if forms.get('min_value') == forms.get('max_value') == 'scalar' and self.min_value == self.max_value:
            rules = [f'= {self.min_value}']
        elif self.min_value is not None and self.min_value is self.max_value:
            rules = [f'= value {forms["min_value"]}']
        else:
            rules = [
                f'{op} {texts[label]}' for label, op in (('min_value', '>='), ('max_value', '<=')) if label in texts
            ]
        rules.append('empty allowed' if self.isna_ok else 'not empty')
        return ', '.join(rules)

    def _subset_text(self) -> str:
        return f' for objects {self.object_subset}' if self.object_subset is not None else ''
