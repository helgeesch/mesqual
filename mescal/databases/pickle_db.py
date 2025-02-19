import os

import pandas as pd

from mescal.typevars import DataSetType, Flagtype, DataSetConfigType
from mescal.databases.data_base import DataBase


class PickleDataBase(DataBase):
    def __init__(self, folder_path: str):
        self._folder_path = folder_path
        self._ensure_folder_exists(folder_path)

    def get(
            self,
            data_set: DataSetType,
            flag: Flagtype,
            config: DataSetConfigType,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        return pd.read_pickle(self._get_file_path(data_set, flag, config, **kwargs))

    def set(
            self,
            data_set: DataSetType,
            flag: Flagtype,
            config: DataSetConfigType,
            value,
            **kwargs
    ):
        file_path = self._get_file_path(data_set, flag, config, **kwargs)
        value.to_pickle(file_path)

    def key_is_up_to_date(
            self,
            data_set: DataSetType,
            flag: Flagtype,
            config: DataSetConfigType,
            **kwargs
    ):
        file_path = self._get_file_path(data_set, flag, config, **kwargs)
        return os.path.exists(file_path)

    def _get_file_path(self, data_set: DataSetType, flag: Flagtype, config: DataSetConfigType = None, **kwargs) -> str:
        components = [data_set.name, str(flag)]

        if config:
            config_hash = hash(frozenset(config.__dict__.items()))
            components.append(f"config_{config_hash}")

        if kwargs:
            kwargs_hash = hash(frozenset(kwargs.items()))
            components.append(f"kwargs_{kwargs_hash}")

        filename = "_".join(components) + ".pickle"
        return os.path.join(self._folder_path, filename)

    @staticmethod
    def _ensure_folder_exists(folder_path: str):
        os.makedirs(folder_path, exist_ok=True)
