import os
from datetime import date
import numpy as np
import pandas as pd

from enums import TopologyTypeEnum, VisualizationTypeEnum, ItemTypeEnum
from mescal.flag.flag import Flagtype
from mescal.flag.flag_index import FlagIndex
from mescal.data_sets import DataSet, DataSetLinkCollection
from mescal.databases import DataBase
from mescal.units import Units
from mescal.utils.package_path import PACKAGE_SOURCE_ROOT_PATH


MOCK_MODELS = {
    'Node.Model': os.path.join(PACKAGE_SOURCE_ROOT_PATH, 'mescal/mock/network/nodes.csv'),
    'Line.Model': os.path.join(PACKAGE_SOURCE_ROOT_PATH, 'mescal/mock/network/lines.csv'),
    'Generator.Model': os.path.join(PACKAGE_SOURCE_ROOT_PATH, 'mescal/mock/network/generators.csv'),
}

MOCK_TS_FLAGS = {'Generator.Generation', 'Node.Price', 'Line.Flow'}


class MockFlagIndex(FlagIndex):

    @classmethod
    def get_object_class_from_flag(cls, flag: str) -> str:
        return flag.split('.')[0]

    @classmethod
    def get_any_variable_flag_for_object_class(cls, object_class: str) -> Flagtype:
        for f in MOCK_TS_FLAGS:
            if cls.get_object_class_from_flag(f) == object_class:
                return f
        raise KeyError(f'No flag recognized for object_class {object_class}')

    def _get_linked_model_flag(self, flag: str) -> Flagtype:
        return f'{self.get_object_class_from_flag(flag)}.Model'

    def _get_item_type(self, flag: str) -> ItemTypeEnum:
        pass

    def _get_visualization_type(self, flag: str) -> VisualizationTypeEnum:
        pass

    def _get_topology_type(self, flag: str) -> TopologyTypeEnum:
        pass

    def _get_unit(self, flag: str) -> Units.Unit:
        pass

    def _get_linked_model_flag_for_membership_column(self, membership_column_name: str) -> Flagtype:
        if membership_column_name == 'node':
            return 'Node.Model'
        raise KeyError()

    def _get_membership_column_name_for_model_flag(self, flag: Flagtype) -> str:
        if flag == 'Node.Model':
            return 'node'
        raise KeyError


class _MockTimeSeriesDataSet(DataSet):
    @property
    def accepted_flags(self) -> set[Flagtype]:
        return MOCK_TS_FLAGS

    def _required_flags_for_flag(self, flag: Flagtype) -> set[Flagtype]:
        return set()

    def _fetch(self, flag: str, **kwargs) -> pd.Series | pd.DataFrame:
        model_flag = self.flag_index.get_linked_model_flag(flag)
        model = self.parent_data_set.fetch(model_flag)
        _index = pd.DatetimeIndex(pd.date_range(date.today(), periods=24, freq='1h'))
        _periods = len(_index)
        data_frame = pd.DataFrame(
            {
                obj: np.random.random(_periods)
                for obj in model.index
            },
            index=_index
        )
        return data_frame


class _MockModelDataSet(DataSet):
    @property
    def accepted_flags(self) -> set[Flagtype]:
        return {self.flag_index.get_linked_model_flag(f) for f in MOCK_TS_FLAGS}

    def _required_flags_for_flag(self, flag: Flagtype) -> set[Flagtype]:
        return set()

    def _fetch(self, flag: Flagtype, **kwargs) -> pd.Series | pd.DataFrame:
        model_csv_path = MOCK_MODELS[flag]
        data_frame = pd.read_csv(model_csv_path, index_col=0)
        return data_frame


class MockDataSet(DataSetLinkCollection):
    __flag_index = MockFlagIndex()

    _child_interpreter_registry: list[type[DataSet]] = [
        _MockTimeSeriesDataSet,
        _MockModelDataSet,
    ]

    def __init__(
            self,
            name: str = None,
            attributes: dict = None,
            data_base: DataBase = None,
    ):
        data_sets = [
            interpreter(parent_data_set=self, flag_index=self.__flag_index)
            for interpreter in self._child_interpreter_registry
        ]
        super().__init__(
            data_sets,
            name=name,
            flag_index=self.__flag_index,
            attributes=attributes,
            data_base=data_base,
        )


if __name__ == '__main__':
    ds = MockDataSet()
    accepted_flags = ds.accepted_flags
    for af in sorted(accepted_flags):
        print(af)
        print(ds.fetch(af))
        print('')
