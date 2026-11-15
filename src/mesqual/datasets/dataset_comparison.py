from __future__ import annotations

import dataclasses
from typing import Any, Generic
from enum import Enum, auto

import numpy as np
import pandas as pd

from mesqual.enums import ComparisonTypeEnum, ItemTypeEnum
from mesqual.flag.dimension import Dimension
from mesqual.flag.flag_spec import FlagSpec
from mesqual.flag.schema import FrameSchema, Level
from mesqual.datasets.dataset import Dataset
from mesqual.datasets.describe import Availability
from mesqual.datasets.fetch_context import UNSET, Unset
from mesqual.datasets.dataset_collection import DatasetCollection, DatasetConcatCollection
from mesqual.typevars import FlagType, DatasetConfigType, FlagIndexType, DatasetType
from mesqual.utils.pandas_utils.is_numeric import pd_is_numeric
from mesqual.utils.pandas_utils.move_level import move_top_level_last


class ComparisonAttributesSourceEnum(Enum):
    """Source of the attributes of a `DatasetComparison`, set by its class attribute `COMPARISON_ATTRIBUTES_SOURCE`.

    The comparison adds the names of its variation and reference dataset in all cases.

    Attributes:
        USE_VARIATION_ATTS: The attributes of the variation dataset (default).
            Attributes set on the comparison do not count.
        USE_REFERENCE_ATTS: The attributes of the reference dataset.
            Attributes set on the comparison do not count.
        USE_INTERSECTION_ATTS: The attributes with the same value in both
            datasets, and the attributes set on the comparison.
    """

    USE_VARIATION_ATTS = auto()
    USE_REFERENCE_ATTS = auto()
    USE_INTERSECTION_ATTS = auto()


class DatasetComparison(
    Generic[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    DatasetCollection[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
):
    """
    The difference between a variation dataset and a reference dataset, with the `Dataset` interface.

    `fetch` fetches the flag from both datasets and compares the results.
    Numeric and non-numeric data, frames and series are handled.

    Comparison Types:
        - DELTA: Variation - Reference (default). For non-numeric data, a
          changed cell becomes `"<var> (was <ref>)"`. An entry (row or
          column) that only the variation has becomes `"<var> (new)"`; one
          that only the reference has becomes `"DELETED: <ref>"`. A NaN
          value in an entry of one side only stays NaN and gets no marker.
        - VARIATION: Returns variation data with optional NaN for unchanged values
        - BOTH: Side-by-side variation and reference data, with a new last
          column level `compared` (`BOTH_LEVEL_NAME`) that holds the names
          of the two datasets.

    All comparison types work for frames and series. For two series,
    `DELTA` and `VARIATION` return a series and `BOTH` a frame with the
    `compared` columns.

    Attributes:
        variation_dataset: The dataset representing the scenario being compared
        reference_dataset: The dataset representing the baseline for comparison

    Dataset parameters (`fetch_params`):

    - `comparison_type`: `DELTA` (default), `VARIATION` or `BOTH`.
    - `replace_unchanged_values_by_nan`: `True` replaces values that are
      equal in both datasets by NaN. Default `False`. Only for flags
      that are not time series, for example model flags, so the
      comparison is empty where nothing changed. For a time series
      flag, `True` raises `ValueError`, also inside a collection. Mask
      the final result instead: for a numeric `DELTA`, for example
      `delta.mask(delta == 0)` after any conversion; for `VARIATION`
      or `BOTH`, compare with the reference.
    - `fill_value`: Value for missing entries before the subtraction of
      `DELTA`. Default `None`: the `fill_value` of the flag spec.
      `float('nan')`: no fill, so an entry missing on one side gives
      NaN, also if the spec has a `fill_value`.

    Each parameter resolves like a flag parameter (`Dataset.fetch`).
    Flag parameters of the call go to both children. A dataset parameter that the call or
    `param_defaults` gave also goes to a child that accepts it, for
    example `comparison_type` to the children of a comparison of
    comparisons.

    Examples:
        The model change of a scenario without node N4:

        >>> base = study.scen.get_dataset('base')
        >>> without_n4 = DatasetComparison(study.scen.get_dataset('without_N4'), base)
        >>> without_n4.fetch('Node.Model', replace_unchanged_values_by_nan=True)['country']
        Node
        N1            NaN
        N2            NaN
        N3            NaN
        N4    DELETED: C2
        Name: country, dtype: object

        `comparison_type` from the call, `fill_value` from `param_defaults`
        and `replace_unchanged_values_by_nan` from its default:

        >>> from mesqual.datasets import DatasetConfig
        >>> from mesqual.enums import ComparisonTypeEnum
        >>> high_demand = DatasetComparison(
        ...     study.scen.get_dataset('high_demand'), base, config=DatasetConfig(param_defaults={'fill_value': 0}),
        ... )
        >>> both = high_demand.fetch('Node.Results.price', comparison_type=ComparisonTypeEnum.BOTH)
        >>> list(both.columns.names)
        ['Node', 'compared']
        >>> high_demand.get_fetch_record('Node.Results.price', comparison_type=ComparisonTypeEnum.BOTH).params
        {'comparison_type': <ComparisonTypeEnum.BOTH: 'both'>, 'replace_unchanged_values_by_nan': False, 'fill_value': 0}
    """

    COMPARISON_ATTRIBUTES_SOURCE = ComparisonAttributesSourceEnum.USE_VARIATION_ATTS
    COMPARISON_NAME_JOIN = ' vs '
    BOTH_LEVEL_NAME = 'compared'
    """Name of the last column level that `BOTH` adds. Its values are the names of the variation and the reference."""
    fetch_params = (
        Dimension(
            'comparison_type',
            'How to compare the variation with the reference.',
            {
                ComparisonTypeEnum.DELTA: 'variation minus reference; for non-numeric values the changes as text',
                ComparisonTypeEnum.VARIATION: 'the variation data',
                ComparisonTypeEnum.BOTH: 'variation and reference side by side',
            },
            default=ComparisonTypeEnum.DELTA,
        ),
        Dimension(
            'replace_unchanged_values_by_nan',
            'Replace values that are equal in both datasets by NaN. Not for time series flags.',
            {
                False: 'keep unchanged values',
                True: 'replace unchanged values by NaN',
            },
            default=False,
        ),
        Dimension(
            'fill_value',
            'Value for missing entries before the subtraction of a delta. None: the fill_value of the spec. '
            "float('nan'): no fill.",
            None,
            default=None,
        ),
    )

    VARIATION_DS_ATT_KEY = 'variation_dataset'
    REFERENCE_DS_ATT_KEY = 'reference_dataset'

    def __init__(
        self,
        variation_dataset: Dataset,
        reference_dataset: Dataset,
        name: str = None,
        attributes: dict = None,
        config: DatasetConfigType = None,
    ):
        name = name or self._get_auto_generated_name(variation_dataset, reference_dataset)

        super().__init__([reference_dataset, variation_dataset], name=name, attributes=attributes, config=config)

        self.variation_dataset = variation_dataset
        self.reference_dataset = reference_dataset

    def _get_auto_generated_name(self, variation_dataset: Dataset, reference_dataset: Dataset) -> str:
        return variation_dataset.name + self.COMPARISON_NAME_JOIN + reference_dataset.name

    @property
    def attributes(self) -> dict:
        match self.COMPARISON_ATTRIBUTES_SOURCE:
            case ComparisonAttributesSourceEnum.USE_VARIATION_ATTS:
                atts = self.variation_dataset.attributes.copy()
            case ComparisonAttributesSourceEnum.USE_REFERENCE_ATTS:
                atts = self.reference_dataset.attributes.copy()
            case _:
                atts = super().attributes
        atts[self.VARIATION_DS_ATT_KEY] = self.variation_dataset.name
        atts[self.REFERENCE_DS_ATT_KEY] = self.reference_dataset.name
        return atts

    def fetch(
        self,
        flag: FlagType,
        config: dict | DatasetConfigType = None,
        *,
        comparison_type: ComparisonTypeEnum | Unset = UNSET,
        replace_unchanged_values_by_nan: bool | Unset = UNSET,
        fill_value: float | None | Unset = UNSET,
        **kwargs: Any,
    ) -> pd.Series | pd.DataFrame:
        """Fetch a flag from both datasets and compare them, as `Dataset.fetch`.

        A dataset parameter left out of the call resolves from
        `param_defaults` of the effective config, else from the default
        given below.

        Args:
            flag: The flag.
            config: Config overrides for this call, as in `Dataset.fetch`.
            comparison_type: `DELTA` (default), `VARIATION` or `BOTH`.
            replace_unchanged_values_by_nan: `True` replaces values that are
                equal in both datasets by NaN. Default `False`. Only for flags
                that are not time series; a time series flag raises
                `ValueError`. Mask the final result instead: for a numeric
                `DELTA`, for example `delta.mask(delta == 0)` after any
                conversion; for `VARIATION` or `BOTH`, compare with the
                reference.
            fill_value: Value for missing entries before the subtraction of
                `DELTA`. Default `None`: the `fill_value` of the flag spec.
                `float('nan')`: no fill.
            **kwargs: Flag parameters, which go to both datasets, and the `granularity` argument of `Dataset.fetch`.

        Raises:
            ValueError: As `Dataset.fetch`, and for
                `replace_unchanged_values_by_nan=True` on a time series flag.
        """
        given = self._given(
            comparison_type=comparison_type,
            replace_unchanged_values_by_nan=replace_unchanged_values_by_nan,
            fill_value=fill_value,
        )
        return super().fetch(flag, config, **given, **kwargs)

    def transform_schema(self, schema: FrameSchema) -> FrameSchema:
        """Add `Level(BOTH_LEVEL_NAME, 'dataset')` as the last column level if the default `comparison_type` is `BOTH`.

        The default comes from the `param_defaults` of the instance config,
        else from the dimension.
        """
        comparison_type = (self.instance_config.param_defaults or {}).get('comparison_type')
        if comparison_type is None:
            comparison_type = next(d.default for d in self.fetch_params if d.name == 'comparison_type')
        if comparison_type is not ComparisonTypeEnum.BOTH:
            return schema
        return dataclasses.replace(schema, columns=(*schema.columns, Level(self.BOTH_LEVEL_NAME, 'dataset')))

    def _check_dataset_values(self, spec: FlagSpec, flag: FlagType, params: dict) -> None:
        """Also raise `ValueError` for `replace_unchanged_values_by_nan=True` on a time series flag."""
        super()._check_dataset_values(spec, flag, params)
        if params['replace_unchanged_values_by_nan'] and spec.item_type is ItemTypeEnum.TimeSeries:
            raise ValueError(
                f'Comparison {self.name}, flag {flag}: replace_unchanged_values_by_nan=True is only for flags that '
                f'are not time series. Mask the final result instead: for a numeric DELTA, for example '
                f'delta.mask(delta == 0) after any conversion; for VARIATION or BOTH, compare with the reference.'
            )

    def _fetch(
        self,
        flag: FlagType,
        effective_config: DatasetConfigType,
        *,
        comparison_type: ComparisonTypeEnum,
        replace_unchanged_values_by_nan: bool,
        fill_value: float | None,
        **kwargs,
    ) -> pd.Series | pd.DataFrame:
        df_var = self.variation_dataset.fetch(
            flag, effective_config, **self._kwargs_for_child(self.variation_dataset, flag)
        )
        df_ref = self.reference_dataset.fetch(
            flag, effective_config, **self._kwargs_for_child(self.reference_dataset, flag)
        )
        if fill_value is None:
            fill_value = self.flag_index.get_fill_value(flag)
        var, ref = self._as_frame(df_var), self._as_frame(df_ref)
        match comparison_type:
            case ComparisonTypeEnum.VARIATION:
                result = self._get_variation_comparison(var, ref, replace_unchanged_values_by_nan)
            case ComparisonTypeEnum.BOTH:
                result = self._get_both_comparison(var, ref, replace_unchanged_values_by_nan)
            case ComparisonTypeEnum.DELTA:
                result = self._get_delta_comparison(var, ref, replace_unchanged_values_by_nan, fill_value)
            case _:
                raise ValueError(f'Unsupported comparison_type: {comparison_type}')
        if isinstance(df_var, pd.Series) and isinstance(df_ref, pd.Series):
            return self._from_frame(result, df_var.name)
        return result

    _combine_availability = staticmethod(Availability.intersection)
    """Combine the availability of both children: the intersection (unavailable if one is unavailable)."""

    _SERIES_COLUMN = '__series__'

    @classmethod
    def _as_frame(cls, data: pd.Series | pd.DataFrame) -> pd.DataFrame:
        """A `Series` as a frame with one column, so all comparison types work for series."""
        return data.to_frame(cls._SERIES_COLUMN) if isinstance(data, pd.Series) else data

    @classmethod
    def _from_frame(cls, result: pd.DataFrame, name) -> pd.Series | pd.DataFrame:
        """The result for two `Series`: a `Series`, or for `BOTH` a frame with the `BOTH_LEVEL_NAME` columns."""
        if result.columns.nlevels > 1:
            return result.droplevel(0, axis=1)
        return result[cls._SERIES_COLUMN].rename(name)

    @staticmethod
    def _unchanged(df: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
        """`True` where `df` and `other` have the same value, or both NaN, in rows and columns that both have."""
        aligned = other.reindex(index=df.index, columns=df.columns)
        equal = df.eq(aligned) | (df.isna() & aligned.isna())
        in_both = np.outer(df.index.isin(other.index), df.columns.isin(other.columns))
        return equal & in_both

    def _get_variation_comparison(
        self, df_var: pd.DataFrame, df_ref: pd.DataFrame, replace_unchanged_values_by_nan: bool
    ) -> pd.DataFrame:
        if replace_unchanged_values_by_nan:
            return df_var.mask(self._unchanged(df_var, df_ref))
        return df_var.copy()

    def _get_both_comparison(
        self, df_var: pd.DataFrame, df_ref: pd.DataFrame, replace_unchanged_values_by_nan: bool
    ) -> pd.DataFrame:
        """Variation and reference side by side: a new last column level `BOTH_LEVEL_NAME` with the dataset names.

        Each column of the variation is followed by the same column of the
        reference. Columns of one side only appear once.
        """
        var_name, ref_name = self.variation_dataset.name, self.reference_dataset.name
        if replace_unchanged_values_by_nan:
            df_var, df_ref = df_var.mask(self._unchanged(df_var, df_ref)), df_ref.mask(self._unchanged(df_ref, df_var))
        result = move_top_level_last(
            pd.concat({var_name: df_var, ref_name: df_ref}, axis=1, names=[self.BOTH_LEVEL_NAME]),
            axis=1,
        )
        order = [
            (*self._as_tuple(column), side)
            for column in df_var.columns.union(df_ref.columns, sort=False)
            for side, frame in ((var_name, df_var), (ref_name, df_ref))
            if column in frame.columns
        ]
        return result[order]

    @staticmethod
    def _as_tuple(column) -> tuple:
        return column if isinstance(column, tuple) else (column,)

    def _get_delta_comparison(
        self,
        df_var: pd.DataFrame,
        df_ref: pd.DataFrame,
        replace_unchanged_values_by_nan: bool,
        fill_value: float | int | None,
    ) -> pd.DataFrame:
        if pd_is_numeric(df_var) and pd_is_numeric(df_ref):
            result = df_var.subtract(df_ref, fill_value=fill_value)
            return result.mask(result == 0) if replace_unchanged_values_by_nan else result

        index = df_var.index.union(df_ref.index, sort=False)
        columns = df_var.columns.union(df_ref.columns, sort=False)
        result = pd.DataFrame(index=index, columns=columns, dtype=object)
        empty = pd.Series(index=index[:0], dtype=object)
        for column in columns:
            result[column] = self._column_delta(
                df_var.get(column, empty),
                df_ref.get(column, empty),
                replace_unchanged_values_by_nan,
                fill_value,
            )
        return result

    def _column_delta(
        self,
        var: pd.Series,
        ref: pd.Series,
        replace_unchanged_values_by_nan: bool,
        fill_value: float | int | None,
    ) -> pd.Series:
        """Delta of one column: a difference for numeric data, else the changes as text.

        For non-numeric data, an entry of one side only gets a marker; a NaN
        value there stays NaN.
        """
        if pd_is_numeric(var) and pd_is_numeric(ref):
            delta = var.subtract(ref, fill_value=fill_value)
            return delta.mask(delta == 0) if replace_unchanged_values_by_nan else delta
        common = var.index.intersection(ref.index, sort=False)
        var_common, ref_common = var.loc[common], ref.loc[common]
        unchanged = (var_common.eq(ref_common) | (var_common.isna() & ref_common.isna())).to_numpy()
        delta = pd.Series(index=var.index.union(ref.index, sort=False), dtype=object)
        delta.loc[common[~unchanged]] = [
            f'{v} (was {r})' for v, r in zip(var_common[~unchanged], ref_common[~unchanged])
        ]
        if not replace_unchanged_values_by_nan:
            delta.loc[common[unchanged]] = var_common[unchanged].to_numpy()
        var_only = var.loc[var.index.difference(ref.index, sort=False)].dropna()
        delta.loc[var_only.index] = [f'{value} (new)' for value in var_only]
        ref_only = ref.loc[ref.index.difference(var.index, sort=False)].dropna()
        delta.loc[ref_only.index] = [f'DELETED: {value}' for value in ref_only]
        return delta


class DatasetConcatCollectionOfComparisons(
    Generic[DatasetConfigType, FlagType, FlagIndexType],
    DatasetConcatCollection[DatasetComparison, DatasetConfigType, FlagType, FlagIndexType],
):
    """Concat collection of comparisons.

    Constructor as `DatasetConcatCollection`, including `concat_axis`.

    Child parameters:
        It passes each child the dataset parameters that the child accepts
        and that the call or `param_defaults` gave, for example
        `comparison_type` of comparisons.

    A comparison parameter that the call or the `param_defaults` of the
    collection does not give reaches no comparison. Each comparison then resolves it with its own config, so a
    comparison gives the same numbers inside the collection as alone.

    Examples:
        Each comparison gets `comparison_type=BOTH` as keyword argument and
        resolves `replace_unchanged_values_by_nan` and `fill_value` itself:

        >>> from mesqual.enums import ComparisonTypeEnum
        >>> both = study.comp.fetch('Node.Results.price', comparison_type=ComparisonTypeEnum.BOTH)
        >>> study.comp.get_fetch_record('Node.Results.price', comparison_type=ComparisonTypeEnum.BOTH).params
        {'comparison_type': <ComparisonTypeEnum.BOTH: 'both'>}
    """

    def fetch(
        self,
        flag: FlagType,
        config: dict | DatasetConfigType = None,
        *,
        comparison_type: ComparisonTypeEnum | Unset = UNSET,
        replace_unchanged_values_by_nan: bool | Unset = UNSET,
        fill_value: float | None | Unset = UNSET,
        **kwargs: Any,
    ) -> pd.Series | pd.DataFrame:
        """Fetch a flag from each comparison and concat the results, as `Dataset.fetch`.

        A dataset parameter left out of the call resolves from
        `param_defaults` of the effective config, else from the default
        given below.

        Args:
            flag: The flag.
            config: Config overrides for this call, as in `Dataset.fetch`.
            comparison_type: `DELTA` (default), `VARIATION` or `BOTH`.
            replace_unchanged_values_by_nan: `True` replaces values that are
                equal in both datasets by NaN. Default `False`. Only for flags
                that are not time series; a time series flag raises
                `ValueError`. Mask the final result instead: for a numeric
                `DELTA`, for example `delta.mask(delta == 0)` after any
                conversion; for `VARIATION` or `BOTH`, compare with the
                reference.
            fill_value: Value for missing entries before the subtraction of
                `DELTA`. Default `None`: the `fill_value` of the flag spec.
                `float('nan')`: no fill.
            **kwargs: Flag parameters, which go to all comparisons, and the `granularity` argument of `Dataset.fetch`.
        """
        given = self._given(
            comparison_type=comparison_type,
            replace_unchanged_values_by_nan=replace_unchanged_values_by_nan,
            fill_value=fill_value,
        )
        return super().fetch(flag, config, **given, **kwargs)

    @classmethod
    def get_child_dataset_type(cls) -> type[DatasetType]:
        return DatasetComparison
