from __future__ import annotations

from abc import abstractmethod
from typing import Any, Literal, TYPE_CHECKING, Hashable, Optional

from mesqual.kpis.definitions.base import KPIDefinition
from mesqual.kpis.kpi import KPI
from mesqual.kpis.attributes import KPIAttributes
from mesqual.kpis.aggregations import Aggregation
from mesqual.units import Units
from mesqual.datasets.dataset import Dataset

if TYPE_CHECKING:
    from mesqual.typevars import FlagTypeProtocol


class CustomKPIDefinition(KPIDefinition):
    """Base class of KPI definitions with custom computation logic.

    A subclass overrides one of two methods:

    - `compute_batch()`: all objects at once, for vectorized operations.
    - `compute_for_object()`: one object per call, for logic that differs per object.

    Examples:
        A demand-weighted price per country. `objects` is a list of names
        here: the index of the model flag, for the default
        `'auto_from_model_flag'`. It is the string
        `'auto_from_batch_generation'` only if the definition sets it.

        >>> from mesqual.units import Units
        >>> class DemandWeightedPrice(CustomKPIDefinition):
        ...     def get_unit(self):
        ...         return Units.EUR_per_MWh
        ...
        ...     def required_flags(self):
        ...         return {'Country.Results.price', 'Country.Results.demand_total'}
        ...
        ...     def compute_batch(self, dataset, objects):
        ...         price = dataset.fetch('Country.Results.price')[objects]
        ...         demand = dataset.fetch('Country.Results.demand_total')[objects]
        ...         return ((price * demand).sum() / demand.sum()).to_dict()
        >>> definition = DemandWeightedPrice('Country.Results.demand_weighted_price', model_flag='Country.Model')
        >>> study.scen.add_kpis_from_definitions_to_all_child_datasets([definition])
        >>> {kpi.attributes.object_name: round(kpi.value, 2) for kpi in study.scen.get_dataset('base').kpi_collection}
        {'C1': 42.7, 'C2': 46.83}
    """

    def __init__(
        self,
        kpi_flag: FlagTypeProtocol,
        model_flag: Optional[FlagTypeProtocol] = None,
        objects: list[Hashable]
        | Literal['auto_from_model_flag', 'auto_from_batch_generation'] = 'auto_from_model_flag',
        name_prefix: str = '',
        name_suffix: str = '',
        extra_attributes: dict = None,
        aggregation: Aggregation | None = None,
    ):
        """
        Initialize custom KPI definition.

        Args:
            kpi_flag: Variable flag for the KPI (Can be a hypothetical/custom flag that doesn't exist in datasets),
                        primarily used for naming and optionally for automatic model detection
            model_flag: Optional model flag (auto-inferred from kpi_flag if None)
            objects: List of objects or 'auto_from_model_flag' / 'auto_from_batch_generation' to discover
            name_prefix: Prefix for KPI names
            name_suffix: Suffix for KPI names
            aggregation: Optional aggregation (for metadata only, not used in computation)
        """
        self.kpi_flag = kpi_flag
        self.model_flag = model_flag
        self.objects = objects
        self.name_prefix = name_prefix
        self.name_suffix = name_suffix
        self.extra_attributes = extra_attributes
        self.aggregation = aggregation

    def generate_kpis(self, dataset: Dataset) -> list[KPI]:
        """
        Generate KPIs using either per-object or batch computation.

        Uses compute_batch() if the subclass overrides it, else compute_for_object().
        Errors raised inside these methods reach the caller.

        Args:
            dataset: Dataset to compute KPIs for

        Returns:
            List of computed KPI instances

        Raises:
            NotImplementedError: If the subclass overrides neither method.
            ValueError: If objects is 'auto_from_batch_generation' and compute_batch()
                is not overridden.
        """
        model_flag = self.model_flag or dataset.flag_index.get_linked_model_flag(self.kpi_flag)

        if self.objects == 'auto_from_model_flag':
            objects = dataset.fetch(model_flag).index.tolist()
        else:
            objects = self.objects

        if self._overrides('compute_batch'):
            return self._generate_kpis_batch(dataset, model_flag, objects)

        if self.objects == 'auto_from_batch_generation':
            raise ValueError(f'If you want to use auto_from_batch_generation, you must override compute_batch().')

        if self._overrides('compute_for_object'):
            return self._generate_kpis_per_object(dataset, model_flag, objects)
        raise NotImplementedError('Must override compute_for_object() or compute_batch().')

    def _overrides(self, method_name: str) -> bool:
        return getattr(type(self), method_name) is not getattr(CustomKPIDefinition, method_name)

    def _generate_kpis_per_object(
        self, dataset: Dataset, model_flag: FlagTypeProtocol, objects: list[Hashable]
    ) -> list[KPI]:
        """
        Generate KPIs by calling compute_for_object() for each object.

        Args:
            dataset: Dataset to compute for
            model_flag: Model flag for objects
            objects: List of object names

        Returns:
            List of KPI instances
        """
        kpis = []
        for obj in objects:
            value = self.compute_for_object(dataset, obj)

            attributes = self._build_attributes(obj, dataset, model_flag)

            kpi = KPI(value=value, attributes=attributes, dataset=dataset)
            kpis.append(kpi)

        return kpis

    def _generate_kpis_batch(
        self,
        dataset: Dataset,
        model_flag: FlagTypeProtocol,
        objects: list[Hashable] | Literal['auto_from_batch_generation'],
    ) -> list[KPI]:
        """
        Generate KPIs by calling compute_batch() once.

        Args:
            dataset: Dataset to compute for
            model_flag: Model flag for objects
            objects: List of object names

        Returns:
            List of KPI instances
        """
        # Compute all values at once
        values_dict = self.compute_batch(dataset, objects)

        kpis = []
        for obj, value in values_dict.items():
            attributes = self._build_attributes(obj, dataset, model_flag)

            kpi = KPI(value=value, attributes=attributes, dataset=dataset)
            kpis.append(kpi)

        return kpis

    def compute_for_object(self, dataset: Dataset, object_name: Hashable) -> Any:
        """
        Compute KPI value for a single object.

        Override this for per-object computation pattern.

        Args:
            dataset: Dataset to fetch data from
            object_name: Name of the object to compute for

        Returns:
            Computed KPI value

        Raises:
            NotImplementedError: If not overridden
        """
        raise NotImplementedError('Must override compute_for_object() or compute_batch()')

    def compute_batch(
        self, dataset: Dataset, objects: list[Hashable] | Literal['auto_from_batch_generation']
    ) -> dict[Hashable, Any]:
        """
        Compute KPI values for all objects at once.

        Override this for batch computation pattern.

        Args:
            dataset: Dataset to fetch data from
            objects: List of object names to compute for

        Returns:
            Dict mapping object_name → value

        Raises:
            NotImplementedError: If not overridden
        """
        raise NotImplementedError('Must override compute_for_object() or compute_batch()')

    @abstractmethod
    def get_unit(self) -> Units.Unit:
        """
        Return the unit for this KPI type.

        Returns:
            Physical unit for the KPI values
        """
        pass

    def _build_attributes(self, object_name: Hashable, dataset: Dataset, model_flag: FlagTypeProtocol) -> KPIAttributes:
        """
        Build KPIAttributes for a KPI instance.

        Args:
            object_name: Object identifier
            dataset: Source dataset
            model_flag: Model flag for the object

        Returns:
            KPIAttributes instance
        """

        return KPIAttributes(
            flag=self.kpi_flag,
            model_flag=model_flag,
            object_name=object_name,
            aggregation=self.aggregation,
            dataset_name=dataset.name,
            dataset_type=type(dataset),
            name_prefix=self.name_prefix,
            name_suffix=self.name_suffix,
            unit=self.get_unit(),
            dataset_attributes=dataset.attributes,
            extra_attributes=self.extra_attributes or dict(),
        )
