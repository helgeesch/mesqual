from __future__ import annotations

from typing import Generic

import pandas as pd

from mescal.data_sets.data_set import DataSet
from mescal.data_sets.data_set_collection import DataSetCollection, DataSetConcatCollection, DataSetType
from mescal.typevars import Flagtype, DataSetConfigType
from mescal.utils.pandas_utils.is_numeric import pd_is_numeric


class DataSetComparison(Generic[DataSetType, DataSetConfigType], DataSetCollection[DataSetType, DataSetConfigType]):
    """
    Takes two DataSets (variation and reference) and fetch method will return the delta between the two (var-ref).
    """
    def __init__(
            self,
            variation_data_set: DataSet,
            reference_data_set: DataSet,
            name: str = None,
            attributes: dict = None,
            config: DataSetConfigType = None,
    ):
        if name is None:
            name = variation_data_set.name + ' vs ' + reference_data_set.name

        super().__init__(
            [reference_data_set, variation_data_set],
            name=name,
            attributes=attributes,
            config=config
        )

        self.variation_data_set = variation_data_set
        self.reference_data_set = reference_data_set

    def _fetch(self, flag: Flagtype, config: dict | DataSetConfigType = None, fill_value: float | int | None = 0, **kwargs) -> pd.Series | pd.DataFrame:
        df_var: pd.DataFrame = self.variation_data_set.fetch(flag, config, **kwargs)
        df_ref: pd.DataFrame = self.reference_data_set.fetch(flag, config, **kwargs)

        if pd_is_numeric(df_var) and pd_is_numeric(df_ref):
            return df_var.subtract(df_ref, fill_value=fill_value)

        # TODO: implement other kinds of comparisons (can be an enum)

        raise NotImplementedError


class DataSetConcatCollectionOfComparisons(
    Generic[DataSetType, DataSetConfigType],
    DataSetConcatCollection[DataSetComparison, DataSet]
):
    data_set_type = DataSetComparison
