import os

import pandas as pd

from mescal.typevars import DataSetType, FlagType, DataSetConfigType
from mescal.databases.data_base import DataBase


class PickleDataBase(DataBase):
    def __init__(self, folder_path: str):
        self._folder_path = folder_path
        self._ensure_folder_exists(folder_path)

    def get(
            self,
            data_set: DataSetType,
            flag: FlagType,
            config: DataSetConfigType,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        return pd.read_pickle(self._get_file_path(data_set, flag, config, **kwargs))

    def set(
            self,
            data_set: DataSetType,
            flag: FlagType,
            config: DataSetConfigType,
            value,
            **kwargs
    ):
        file_path = self._get_file_path(data_set, flag, config, **kwargs)
        value.to_pickle(file_path)

    def key_is_up_to_date(
            self,
            data_set: DataSetType,
            flag: FlagType,
            config: DataSetConfigType,
            **kwargs
    ):
        file_path = self._get_file_path(data_set, flag, config, **kwargs)
        return os.path.exists(file_path)

    def _get_config_hash(self, config: DataSetConfigType = None) -> str:
        if config is None:
            return ""

        attrs = {
            name: getattr(config, name)
            for name in dir(config)
            if not name.startswith('_') and not callable(getattr(config, name))
        }

        sorted_items = sorted(attrs.items())

        # Convert to string representation for hashing
        config_str = str(sorted_items)
        return str(hash(config_str))

    def _get_kwargs_hash(self, kwargs: dict) -> str:
        if not kwargs:
            return ""

        str_dict = {
            str(k): str(v)
            for k, v in kwargs.items()
        }

        sorted_items = sorted(str_dict.items())
        return str(hash(str(sorted_items)))

    def _get_file_path(self, data_set: DataSetType, flag: FlagType, config: DataSetConfigType = None, **kwargs) -> str:
        components = [data_set.name, str(flag)]

        config_hash = self._get_config_hash(config)
        if config_hash:
            components.append(f"config_{config_hash}")

        kwargs_hash = self._get_kwargs_hash(kwargs)
        if kwargs_hash:
            components.append(f"kwargs_{kwargs_hash}")

        filename = "_".join(components) + ".pickle"
        return os.path.join(self._folder_path, filename)

    @staticmethod
    def _ensure_folder_exists(folder_path: str):
        os.makedirs(folder_path, exist_ok=True)
