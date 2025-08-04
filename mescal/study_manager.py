import os

from mescal.datasets.dataset_collection import DatasetConcatCollection, Dataset
from mescal.datasets.dataset_comparison import DatasetComparison, DatasetConcatCollectionOfComparisons


class StudyManager:
    """
    Central orchestrator for multi-scenario energy systems analysis using MESCAL's three-tier collection system.
    
    StudyManager provides unified access to scenarios, scenario comparisons, and combined datasets,
    implementing MESCAL's core architectural principle that "Everything is a Dataset". It manages
    the complete lifecycle of multi-scenario studies including data organization, automatic delta
    computation, and export functionality.
    
    The class implements three primary data access patterns:
    - `.scen`: Access to individual scenario data (DatasetConcatCollection)
    - `.comp`: Access to scenario comparison deltas (DatasetConcatCollectionOfComparisons)  
    - `.scen_comp`: Unified access to both scenarios and comparisons with type distinction
    
    Attributes:
        scen (DatasetConcatCollection): Collection of scenario datasets with consistent .fetch() interface
        comp (DatasetConcatCollectionOfComparisons): Collection of scenario comparisons with automatic delta calculation
        scen_comp (DatasetConcatCollection): Unified collection combining scenarios and comparisons
    
    Examples:
        Basic multi-scenario study setup:
        
        >>> from mescal import StudyManager
        >>> from mescal_pypsa import PyPSADataset
        >>> 
        >>> study = StudyManager.factory_from_scenarios(
        ...     scenarios=[
        ...         PyPSADataset(base_network, name='base'),
        ...         PyPSADataset(high_res_network, name='high_res'),
        ...         PyPSADataset(low_gas_network, name='low_gas'),
        ...     ],
        ...     comparisons=[('high_res', 'base'), ('low_gas', 'base')],
        ... )

        Three-tier data access:
        
        >>> # Access scenario data across all scenarios
        >>> scenario_prices = study.scen.fetch('buses_t.marginal_price')
        >>> 
        >>> # Access comparison deltas automatically calculated
        >>> price_changes = study.comp.fetch('buses_t.marginal_price')
        >>> 
        >>> # Access unified view with type-level distinction
        >>> unified_data = study.scen_comp.fetch('buses_t.marginal_price')
    
    Notes:
        - All collections share the same .fetch(flag) interface for consistent data access
        - Scenario comparisons are computed automatically as deltas (variation - reference)
        - The unified collection uses 'type' as the concat_level_name to distinguish data sources
        - Supports dynamic addition of scenarios and comparisons after initialization
    """
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
