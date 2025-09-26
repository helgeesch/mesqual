from __future__ import annotations
from typing import Type, Any, Generic
from abc import ABC
from dataclasses import dataclass
import inspect

from mesqual.datasets.dataset import Dataset
from mesqual.datasets.dataset_collection import DatasetLinkCollection
from mesqual.typevars import FlagType, DatasetType, DatasetConfigType, FlagIndexType
from mesqual.databases.database import Database


@dataclass
class InterpreterSignature:
    args: tuple[str, ...]
    defaults: tuple[Any, ...]

    @classmethod
    def from_interpreter(cls, interpreter: Type[Dataset]) -> InterpreterSignature:
        signature = inspect.signature(interpreter.__init__)
        params = signature.parameters
        relevant_params = cls._get_relevant_init_parameters(params)
        return cls(
            args=tuple(relevant_params.keys()),
            defaults=cls._extract_default_values(relevant_params)
        )

    @staticmethod
    def _get_relevant_init_parameters(params: dict) -> dict:
        return {
            name: param for name, param in params.items()
            if name not in ('self', 'parent_dataset')
        }

    @staticmethod
    def _extract_default_values(params: dict) -> tuple:
        return tuple(
            param.default if param.default is not param.empty else None
            for param in params.values()
        )


class PlatformDataset(
    Generic[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    DatasetLinkCollection[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    ABC
):
    """
    Base class for platform-specific datasets with automatic interpreter management.

    PlatformDataset provides the foundation for integrating MESQUAL with specific
    energy modeling platforms (PyPSA, PLEXOS, etc.). It manages a registry of
    data interpreters and automatically instantiates them to handle different
    types of platform data.

    Key Features:
        - Automatic interpreter registration and instantiation
        - Type-safe interpreter management through generics
        - Flexible argument passing to interpreter constructors
        - Support for study-specific interpreter extensions
        - Unified data access through DatasetLinkCollection routing

    Architecture:
        - Uses DatasetLinkCollection for automatic flag routing
        - Manages interpreter registry at class level
        - Auto-instantiates all registered interpreters on construction
        - Supports inheritance and interpreter registration on subclasses

    Type Parameters:
        DatasetType: Base type for all interpreters (must be Dataset subclass)
        DatasetConfigType: Configuration class for dataset behavior
        FlagType: Type used for data flag identification
        FlagIndexType: Flag index implementation for flag mapping

    Class Attributes:
        _interpreter_registry: List of registered interpreter classes

    Usage Pattern:
        1. Create platform dataset class inheriting from PlatformDataset
        2. Define get_child_dataset_type() to specify interpreter base class
        3. Create interpreter classes inheriting from the base interpreter
        4. Register interpreters using @PlatformDataset.register_interpreter
        5. Instantiate platform dataset - interpreters are auto-created

    Example:

        >>> # Define platform dataset
        >>> class PyPSADataset(PlatformDataset[PyPSAInterpreter, ...]):
        ...     @classmethod
        ...     def get_child_dataset_type(cls):
        ...         return PyPSAInterpreter
        ...
        >>> # Register core interpreters
        >>> @PyPSADataset.register_interpreter
        ... class PyPSAModelInterpreter(PyPSAInterpreter):
        ...     @property
        ...     def accepted_flags(self):
        ...         return {'buses', 'generators', 'lines'}
        ...
        >>> @PyPSADataset.register_interpreter  
        ... class PyPSATimeSeriesInterpreter(PyPSAInterpreter):
        ...     @property
        ...     def accepted_flags(self):
        ...         return {'buses_t.marginal_price', 'generators_t.p'}
        ...
        >>> # Register study-specific interpreter
        >>> @PyPSADataset.register_interpreter
        ... class CustomVariableInterpreter(PyPSAInterpreter):
        ...     @property
        ...     def accepted_flags(self):
        ...         return {'custom_metric'}
        ...
        >>> # Use platform dataset
        >>> dataset = PyPSADataset(network=my_network)
        >>> buses = dataset.fetch('buses')  # Routes to ModelInterpreter
        >>> prices = dataset.fetch('buses_t.marginal_price')  # Routes to TimeSeriesInterpreter
        >>> custom = dataset.fetch('custom_metric')  # Routes to CustomVariableInterpreter

    Notes:
        - Interpreters are registered at class level and shared across instances
        - Registration order affects routing (last registered = first checked)
        - All registered interpreters are instantiated for each platform dataset
        - Constructor arguments are automatically extracted and passed to interpreters
    """

    _interpreter_registry: list[Type[DatasetType]] = []

    def __init__(
            self,
            name: str = None,
            flag_index: FlagIndexType = None,
            attributes: dict = None,
            database: Database = None,
            config: DatasetConfigType = None,
            **kwargs
    ):
        super().__init__(
            datasets=[],
            name=name,
            flag_index=flag_index,
            attributes=attributes,
            database=database,
            config=config,
        )
        interpreter_args = self._prepare_interpreter_initialization_args(kwargs)
        datasets = self._initialize_registered_interpreters(interpreter_args)
        self.add_datasets(datasets)

    @classmethod
    def register_interpreter(cls, interpreter: Type[DatasetType]) -> Type['DatasetType']:
        """
        Register a data interpreter class with this platform dataset.
        
        This method is typically used as a decorator to register interpreter classes
        that handle specific types of platform data. Registered interpreters are
        automatically instantiated when the platform dataset is created.
        
        Args:
            interpreter: Interpreter class that must inherit from get_child_dataset_type()
            
        Returns:
            The interpreter class (unchanged) to support decorator usage
            
        Raises:
            TypeError: If interpreter doesn't inherit from the required base class
            
        Example:

            >>> @PyPSADataset.register_interpreter
            ... class CustomInterpreter(PyPSAInterpreter):
            ...     @property
            ...     def accepted_flags(self):
            ...         return {'custom_flag'}
            ...     
            ...     def _fetch(self, flag, config, **kwargs):
            ...         return compute_custom_data()
        """
        cls._validate_interpreter_type(interpreter)
        if interpreter not in cls._interpreter_registry:
            cls._add_interpreter_to_registry(interpreter)
        return interpreter

    @classmethod
    def get_registered_interpreters(cls) -> list[Type[DatasetType]]:
        return cls._interpreter_registry.copy()

    def get_interpreter_instance(self, interpreter_type: Type[DatasetType]) -> DatasetType:
        interpreter = self._find_interpreter_instance(interpreter_type)
        if interpreter is None:
            raise ValueError(
                f'No interpreter instance found for type {interpreter_type.__name__}'
            )
        return interpreter

    def get_flags_by_interpreter(self) -> dict[Type[DatasetType], set[FlagType]]:
        return {
            type(interpreter): interpreter.accepted_flags
            for interpreter in self.datasets.values()
        }

    def _prepare_interpreter_initialization_args(self, kwargs: dict) -> dict:
        interpreter_signature = InterpreterSignature.from_interpreter(self.get_child_dataset_type())
        return {
            arg: kwargs.get(arg, default)
            for arg, default in zip(interpreter_signature.args, interpreter_signature.defaults)
        }

    def _initialize_registered_interpreters(self, interpreter_args: dict) -> list[DatasetType]:
        return [
            interpreter(**interpreter_args, parent_dataset=self)
            for interpreter in self._interpreter_registry
        ]

    @classmethod
    def _validate_interpreter_type(cls, interpreter: Type[DatasetType]) -> None:
        if not issubclass(interpreter, cls.get_child_dataset_type()):
            raise TypeError(
                f'Interpreter must be subclass of {cls.get_child_dataset_type().__name__}'
            )

    @classmethod
    def _validate_interpreter_not_registered(cls, interpreter: Type[DatasetType]) -> None:
        if interpreter in cls._interpreter_registry:
            raise ValueError(f'Interpreter {interpreter.__name__} already registered')

    @classmethod
    def _add_interpreter_to_registry(cls, interpreter: Type[DatasetType]) -> None:
        cls._interpreter_registry.insert(0, interpreter)

    def _find_interpreter_instance(self, interpreter_type: Type[DatasetType]) -> DatasetType | None:
        for interpreter in self.datasets.values():
            if isinstance(interpreter, interpreter_type):
                return interpreter
        return None
