import os

import pandas as pd

from mescal.databases.data_base import DataBase


class PickleDataBase(DataBase):
    def __init__(self, folder_path: str):
        self._folder_path = folder_path
        self._ensure_folder_exists(folder_path)

    def get(self, key, **kwargs):
        return pd.read_pickle(self._file_path(key))

    def set(self, key, value: pd.Series | pd.DataFrame, **kwargs):
        file_path = self._file_path(key, **kwargs)
        value.to_pickle(file_path)

    def key_is_up_to_date(self, key, timestamp=None, **kwargs):
        file_path = self._file_path(key, **kwargs)
        exists = os.path.exists(file_path)
        if not exists:
            return False
        if timestamp is None:
            return True
        return timestamp <= os.path.getmtime(file_path)

    def _file_path(self, key, **kwargs) -> str:
        kwarg_suffix = f'_{hash(str(kwargs))}' if kwargs else ''
        return os.path.join(self._folder_path, f'{str(key)}{kwarg_suffix}.pickle')

    @staticmethod
    def _ensure_folder_exists(folder_path: str):
        os.makedirs(folder_path, exist_ok=True)
