from abc import ABC, abstractmethod

import pandas as pd

from mescal.typevars import DataSetType, Flagtype, DataSetConfigType


class DataBase(ABC):
    @abstractmethod
    def get(
            self,
            data_set: DataSetType,
            flag: Flagtype,
            config: DataSetConfigType,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        pass
    
    @abstractmethod
    def set(
            self,
            data_set: DataSetType,
            flag: Flagtype,
            config: DataSetConfigType,
            value,
            **kwargs
    ):
        pass
    
    @abstractmethod
    def key_is_up_to_date(
            self,
            data_set: DataSetType,
            flag: Flagtype,
            config: DataSetConfigType,
            timestamp=None,
            **kwargs
    ):
        pass
