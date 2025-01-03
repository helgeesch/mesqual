import os

from mescal.data_sets.data_set_collection import DataSetConcatCollection, DataSet
from mescal.data_sets.data_set_comparison import DataSetComparison, DataSetConcatCollectionOfComparisons


class StudyManager:
    def __init__(
            self,
            scenarios: DataSetConcatCollection,
            comparisons: DataSetConcatCollectionOfComparisons,
            export_folder: str = None,
    ):
        self._scenarios = scenarios
        self._comparisons = comparisons
        self._scenarios_and_comparisons: DataSetConcatCollection = DataSetConcatCollection(
            [
                scenarios,
                comparisons
            ],
            name='scenarios_and_comparisons',
            concat_level_name='type',
        )
        self._export_folder: str = export_folder
        self._ensure_folder_exists(export_folder)

    @property
    def export_folder(self) -> str:
        return self._export_folder

    @staticmethod
    def _ensure_folder_exists(folder: str):
        if not os.path.exists(folder):
            os.mkdir(folder)

    def export_path(self, file_name: str) -> str:
        if self._export_folder is None:
            raise RuntimeError(f'Export folder must be assigned first.')
        return os.path.join(self._export_folder, file_name)

    @property
    def scen(self) -> DataSetConcatCollection:
        return self._scenarios

    def add_scenario(self, data_set: DataSet):
        self._scenarios.add_data_set(data_set)

    @property
    def comp(self) -> DataSetConcatCollectionOfComparisons:
        return self._comparisons

    def add_comparison(self, data_set: DataSetComparison):
        self._comparisons.add_data_set(data_set)

    @property
    def scen_comp(self) -> DataSetConcatCollection:
        return self._scenarios_and_comparisons

    @classmethod
    def factory_from_scenarios(
            cls,
            scenarios: list[DataSet],
            comparisons: list[tuple[str, str]],
            export_folder: str = None
    ) -> 'StudyManager':
        scen = DataSetConcatCollection(scenarios, name='scenario', concat_level_name='data_set',)
        comp = DataSetConcatCollectionOfComparisons(
            data_sets=[
                DataSetComparison(scen.get_data_set(var), scen.get_data_set(ref))
                for var, ref in comparisons
            ],
            name='comparison',
            concat_level_name='data_set',
        )
        return cls(scen, comp, export_folder)
