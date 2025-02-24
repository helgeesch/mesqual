from abc import ABC, abstractmethod
from typing import Generic

import pandas as pd

from mescal.typevars import DataSetType, FlagType, DataSetConfigType


class DataBase(Generic[DataSetType, DataSetConfigType], ABC):
    @abstractmethod
    def get(
            self,
            data_set: DataSetType,
            flag: FlagType,
            config: DataSetConfigType,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        pass
    
    @abstractmethod
    def set(
            self,
            data_set: DataSetType,
            flag: FlagType,
            config: DataSetConfigType,
            value,
            **kwargs
    ):
        pass
    
    @abstractmethod
    def key_is_up_to_date(
            self,
            data_set: DataSetType,
            flag: FlagType,
            config: DataSetConfigType,
            **kwargs
    ):
        pass
