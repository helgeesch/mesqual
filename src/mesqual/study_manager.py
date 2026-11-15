from pathlib import Path

from mesqual.datasets.dataset_collection import DatasetConcatCollection, Dataset
from mesqual.datasets.dataset_config import DatasetConfig
from mesqual.datasets.dataset_comparison import DatasetComparison, DatasetConcatCollectionOfComparisons


class StudyManager:
    """The scenarios and the comparisons of a study, each as a collection with one `fetch` interface.

    - `scen`: the scenario datasets, a `DatasetConcatCollection`.
    - `comp`: the comparisons, a `DatasetConcatCollectionOfComparisons`. A
      comparison gives the variation minus its reference.
    - `scen_comp`: the scenarios and the comparisons, with the extra column
      level `type` on top ('scenario' or 'comparison').

    Each collection concatenates the data of its datasets along the columns,
    with the dataset name in the level `dataset`. `add_scenario` and
    `add_comparison` add datasets after creation.

    Attributes:
        scen (DatasetConcatCollection): The scenario datasets.
        comp (DatasetConcatCollectionOfComparisons): The scenario comparisons.
        scen_comp (DatasetConcatCollection): The scenarios and the comparisons.

    Examples:
        >>> from mesqual.datasets import DatasetConfig
        >>> from mesqual.testing import DummyDataset, DummyScenario
        >>> base = DummyScenario(name='base')
        >>> study = StudyManager.factory_from_scenarios(
        ...     scenarios=[DummyDataset(base), DummyDataset(base.replace(name='high_demand', demand_factor=1.2))],
        ...     comparisons=[('high_demand', 'base')],
        ... )
        >>> prices = study.scen_comp.fetch('Node.Results.price')
        >>> prices.columns.droplevel('Node').unique().tolist()
        [('scenario', 'base'), ('scenario', 'high_demand'), ('comparison', 'high_demand vs base')]

        A study config applies to all three collections:

        >>> quarter_hourly = base.replace(resolution='15min')
        >>> study = StudyManager.factory_from_scenarios(
        ...     scenarios=[DummyDataset(quarter_hourly), DummyDataset(quarter_hourly.replace(name='high_demand', demand_factor=1.2))],
        ...     comparisons=[('high_demand', 'base')],
        ...     config=DatasetConfig(granularity='1h'),
        ... )
        >>> len(study.scen_comp.fetch('Node.Results.price'))  # 2 days of quarter-hours converted to hours
        48
    """

    _TYPE_LEVEL_NAME = 'type'

    def __init__(
        self,
        scenarios: DatasetConcatCollection,
        comparisons: DatasetConcatCollectionOfComparisons,
        study_folder: str | Path = None,
        config: DatasetConfig | None = None,
    ):
        """Build the study from a scenario collection and a comparison collection.

        Args:
            scenarios: Collection of the scenario datasets, `scen`.
            comparisons: Collection of the scenario comparisons, `comp`.
            study_folder: Optional folder of the study, see `study_folder`.
            config: Optional config of the study. It merges on top of the
                instance config of each of `scen`, `comp` and `scen_comp`
                (`Dataset.merge_instance_config`): a field of the study config wins,
                and the other explicit fields of a collection stay. This
                changes the collection objects passed in. The three configs
                are independent copies: a later change on one collection does
                not reach the others. Datasets added later with
                `add_scenario` or `add_comparison` get it through their
                collection and keep their own instance configs. Each name in
                `param_defaults` must be a parameter of each collection, so a
                comparison parameter such as `comparison_type` raises on
                `scen`; set it on `comp`.

        Raises:
            ValueError: `scenarios` or `comparisons` does not have
                `concat_axis=1` and `concat_top=True`, the two have
                different `concat_level_name`, or their `concat_level_name`
                is `type`, the level that `scen_comp` adds.
        """
        self._check_layout(scenarios, comparisons)
        self._scenarios = scenarios
        self._comparisons = comparisons
        self._scenarios_and_comparisons: DatasetConcatCollection = DatasetConcatCollection(
            [scenarios, comparisons],
            name='scenarios_and_comparisons',
            concat_level_name=self._TYPE_LEVEL_NAME,
        )
        self._study_folder = Path(study_folder) if study_folder is not None else study_folder
        if config is not None:
            for collection in (self._scenarios, self._comparisons, self._scenarios_and_comparisons):
                collection.merge_instance_config(config)

    @classmethod
    def _check_layout(cls, scenarios: DatasetConcatCollection, comparisons: DatasetConcatCollection) -> None:
        """Raise `ValueError` if `scenarios` or `comparisons` does not have the concat layout that `scen_comp` needs."""
        for collection in (scenarios, comparisons):
            if collection.concat_axis != 1:
                raise ValueError(
                    f'{collection.name}: StudyManager needs concat_axis=1, got {collection.concat_axis}. '
                    f'scen_comp concats scen and comp on the columns, so collections that stack rows '
                    f'give a block-diagonal frame.'
                )
            if collection.concat_top is not True:
                raise ValueError(
                    f'{collection.name}: StudyManager needs concat_top=True, got {collection.concat_top!r}. '
                    f'scen_comp concats the columns of scen and comp, so their concat level must be at the top.'
                )
        if scenarios.concat_level_name != comparisons.concat_level_name:
            raise ValueError(
                f'{comparisons.name}: StudyManager needs the concat_level_name of {scenarios.name}, '
                f'{scenarios.concat_level_name!r}, got {comparisons.concat_level_name!r}. '
                f'scen_comp concats the columns of scen and comp, so their level names must match.'
            )
        if scenarios.concat_level_name == cls._TYPE_LEVEL_NAME:
            raise ValueError(
                f'{scenarios.name} and {comparisons.name}: StudyManager needs a concat_level_name other than '
                f'{cls._TYPE_LEVEL_NAME!r}. scen_comp adds the level {cls._TYPE_LEVEL_NAME!r} on top of their level.'
            )

    @property
    def scen(self) -> DatasetConcatCollection:
        return self._scenarios

    @property
    def scenario_names(self) -> list[str]:
        return [s.name for s in self.scen.dataset_iterator]

    def add_scenario(self, dataset: Dataset):
        self._scenarios.add_dataset(dataset)

    @property
    def comp(self) -> DatasetConcatCollectionOfComparisons:
        return self._comparisons

    @property
    def comparison_names(self) -> list[str]:
        return [c.name for c in self.comp.dataset_iterator]

    def add_comparison(self, dataset: DatasetComparison):
        self._comparisons.add_dataset(dataset)

    @property
    def scen_comp(self) -> DatasetConcatCollection:
        return self._scenarios_and_comparisons

    def describe_study(self, as_records: bool = False) -> dict:
        """Describe the scenarios and comparisons of the study. Loads no time series data.

        The result has these keys:

        - `scenarios`: `scen.describe_datasets(max_depth=1)`, the
          scenario collection and its scenarios.
        - `comparisons`: `comp.describe_datasets(max_depth=1)`.
        - `scenario_flags`: `scen.describe_flags()`, the union over the
          scenarios, with the scenarios that have each flag.
        - `comparison_flags`: `comp.describe_flags()`.

        Args:
            as_records: Give all four parts as lists of plain dicts, ready for JSON.

        Returns:
            A dict of the four descriptions.

        Examples:
            >>> overview = study.describe_study()
            >>> list(overview)
            ['scenarios', 'comparisons', 'scenario_flags', 'comparison_flags']
            >>> overview['scenarios'].index.tolist()
            ['scenario', 'scenario/base', 'scenario/high_demand', 'scenario/without_N4', 'scenario/short']
        """
        return {
            'scenarios': self.scen.describe_datasets(max_depth=1, as_records=as_records),
            'comparisons': self.comp.describe_datasets(max_depth=1, as_records=as_records),
            'scenario_flags': self.scen.describe_flags(as_records=as_records),
            'comparison_flags': self.comp.describe_flags(as_records=as_records),
        }

    @property
    def study_folder(self) -> Path:
        if self._study_folder is None:
            raise RuntimeError(f'Export folder must be assigned first.')
        return self._study_folder

    @study_folder.setter
    def study_folder(self, folder_path: str | Path | None):
        self._study_folder = Path(folder_path) if folder_path is not None else None

    @classmethod
    def factory_from_scenarios(
        cls,
        scenarios: list[Dataset],
        comparisons: list[tuple[str, str]],
        study_folder: str | Path = None,
        config: DatasetConfig | None = None,
    ) -> 'StudyManager':
        """Build the study from scenario datasets and `(variation, reference)` name pairs.

        Args:
            scenarios: The scenario datasets.
            comparisons: One `(variation, reference)` pair of scenario names
                per comparison. The comparison is named
                `'<variation> vs <reference>'`.
            study_folder: Optional folder of the study.
            config: Optional config of the study, see `__init__`.

        Returns:
            The study.
        """
        scen = DatasetConcatCollection(
            scenarios,
            name='scenario',
            concat_level_name='dataset',
        )
        comp = DatasetConcatCollectionOfComparisons(
            datasets=[DatasetComparison(scen.get_dataset(var), scen.get_dataset(ref)) for var, ref in comparisons],
            name='comparison',
            concat_level_name='dataset',
        )
        return cls(scen, comp, study_folder, config)
