import os

from mescal.datasets.dataset_collection import DatasetConcatCollection, Dataset
from mescal.datasets.dataset_comparison import DatasetComparison, DatasetConcatCollectionOfComparisons


class StudyManager:
    def __init__(
            self,
            scenarios: DatasetConcatCollection,
            comparisons: DatasetConcatCollectionOfComparisons,
            export_folder: str = None,
    ):
        self._scenarios = scenarios
        self._comparisons = comparisons
        self._scenarios_and_comparisons: DatasetConcatCollection = DatasetConcatCollection(
            [
                scenarios,
                comparisons
            ],
            name='scenarios_and_comparisons',
            concat_level_name='type',
        )
        self._export_folder: str = export_folder
        if export_folder is not None:
            self._ensure_folder_exists(export_folder)

    @property
    def scen(self) -> DatasetConcatCollection:
        return self._scenarios

    def add_scenario(self, dataset: Dataset):
        self._scenarios.add_dataset(dataset)

    @property
    def comp(self) -> DatasetConcatCollectionOfComparisons:
        return self._comparisons

    def add_comparison(self, dataset: DatasetComparison):
        self._comparisons.add_dataset(dataset)

    @property
    def scen_comp(self) -> DatasetConcatCollection:
        return self._scenarios_and_comparisons

    @property
    def export_folder(self) -> str:
        return self._export_folder

    @export_folder.setter
    def export_folder(self, folder_path: str):
        self._ensure_folder_exists(folder_path)
        self._export_folder = folder_path

    @staticmethod
    def _ensure_folder_exists(folder: str):
        os.makedirs(folder, exist_ok=True)

    def export_path(self, file_name: str) -> str:
        if self._export_folder is None:
            raise RuntimeError(f'Export folder must be assigned first.')
        return os.path.join(self._export_folder, file_name)

    @classmethod
    def factory_from_scenarios(
            cls,
            scenarios: list[Dataset],
            comparisons: list[tuple[str, str]],
            export_folder: str = None
    ) -> 'StudyManager':
        scen = DatasetConcatCollection(scenarios, name='scenario', concat_level_name='dataset',)
        comp = DatasetConcatCollectionOfComparisons(
            datasets=[
                DatasetComparison(scen.get_dataset(var), scen.get_dataset(ref))
                for var, ref in comparisons
            ],
            name='comparison',
            concat_level_name='dataset',
        )
        return cls(scen, comp, export_folder)
