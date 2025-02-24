from __future__ import annotations
from typing import Type, Any, Generic
from abc import ABC
from dataclasses import dataclass
import inspect

from mescal.data_sets.data_set import DataSet
from mescal.data_sets.data_set_collection import DataSetLinkCollection
from mescal.typevars import Flagtype, DataSetType, DataSetConfigType, FlagIndexType
from mescal.databases.data_base import DataBase


@dataclass
class InterpreterSignature:
    args: tuple[str, ...]
    defaults: tuple[Any, ...]

    @classmethod
    def from_interpreter(cls, interpreter: Type[DataSet]) -> InterpreterSignature:
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
            if name not in ('self', 'parent_data_set')
        }

    @staticmethod
    def _extract_default_values(params: dict) -> tuple:
        return tuple(
            param.default if param.default is not param.empty else None
            for param in params.values()
        )


class PlatformDataSet(
    Generic[DataSetType, DataSetConfigType, Flagtype, FlagIndexType],
    DataSetLinkCollection[DataSetType, DataSetConfigType, Flagtype, FlagIndexType],
    ABC
):
    """
    Base class for managing platform-specific data interpreters in a type-safe way.

    PlatformDataSet serves as a registry and container for data interpreters that handle
    different aspects of platform data. It ensures type safety through generics and
    automatically initializes registered interpreters when instantiated.

    Type Parameters
    --------------
    DataSetType : TypeVar
        Must be a subclass of DataSet. Defines the type of interpreters that can be
        registered with this platform dataset.

    Attributes
    ----------
    _interpreter_registry : list[Type[DataSetType]]
        List of registered interpreter classes, ordered by registration time (newest first)

    Usage
    -----
    To create a platform-specific dataset:
    1. Subclass PlatformDataSet
    2. Define get_child_data_set_type
    3. Create interpreters that inherit from collection_member_data_set_type
    4. Register interpreters using the @register_interpreter decorator

    Example
    -------
    >>> class MyPlatformDataSet(PlatformDataSet[MyInterpreterBase]):
    ...     @classmethod
    ...     def get_child_data_set_type(cls) -> type[DataSetType]:
    ...         return MyInterpreterBase
    ...
    >>> @MyPlatformDataSet.register_interpreter
    ... class ModelCSVInterpreter(MyInterpreterBase):
    ...     def _fetch(self, flag: Flagtype, ...) -> pd.DataFrame:
    ...         # Implementation
    ...         pass
    ...
    >>> @MyPlatformDataSet.register_interpreter
    ... class ResultCSVInterpreter(MyInterpreterBase):
    ...     def _fetch(self, flag: Flagtype, ...) -> pd.DataFrame:
    ...         # Implementation
    ...         pass
    ...
    >>> @MyPlatformDataSet.register_interpreter
    ... class StudySpecificVariable(MyInterpreterBase):
    ...     def _fetch(self, flag: Flagtype, ...) -> pd.DataFrame:
    ...         # Implementation
    ...         pass

    Notes
    -----
    - Interpreters are initialized automatically when the platform dataset is instantiated
    - Registration order determines interpreter priority (last registered = first checked)
    - Each interpreter must implement the interface defined by child_data_set_type
    - Arguments required by interpreters are automatically extracted and passed during
      initialization
    """

    _interpreter_registry: list[Type[DataSetType]] = []

    def __init__(
            self,
            name: str = None,
            flag_index: FlagIndexType = None,
            attributes: dict = None,
            data_base: DataBase = None,
            config: DataSetConfigType = None,
            **kwargs
    ):
        super().__init__(
            data_sets=[],
            name=name,
            flag_index=flag_index,
            attributes=attributes,
            data_base=data_base,
            config=config,
        )
        interpreter_args = self._prepare_interpreter_initialization_args(kwargs)
        data_sets = self._initialize_registered_interpreters(interpreter_args)
        self.add_data_sets(data_sets)

    @classmethod
    def register_interpreter(cls, interpreter: Type[DataSetType]) -> Type[DataSetType]:
        cls._validate_interpreter_type(interpreter)
        cls._validate_interpreter_not_registered(interpreter)
        cls._add_interpreter_to_registry(interpreter)
        return interpreter

    @classmethod
    def get_registered_interpreters(cls) -> list[Type[DataSetType]]:
        return cls._interpreter_registry.copy()

    def get_interpreter_instance(self, interpreter_type: Type[DataSetType]) -> DataSetType:
        interpreter = self._find_interpreter_instance(interpreter_type)
        if interpreter is None:
            raise ValueError(
                f'No interpreter instance found for type {interpreter_type.__name__}'
            )
        return interpreter

    def get_flags_by_interpreter(self) -> dict[Type[DataSetType], set[Flagtype]]:
        return {
            type(interpreter): interpreter.accepted_flags
            for interpreter in self.data_sets.values()
        }

    def _prepare_interpreter_initialization_args(self, kwargs: dict) -> dict:
        interpreter_signature = InterpreterSignature.from_interpreter(self.get_child_data_set_type())
        return {
            arg: kwargs.get(arg, default)
            for arg, default in zip(interpreter_signature.args, interpreter_signature.defaults)
        }

    def _initialize_registered_interpreters(self, interpreter_args: dict) -> list[DataSetType]:
        return [
            interpreter(**interpreter_args, parent_data_set=self)
            for interpreter in self._interpreter_registry
        ]

    @classmethod
    def _validate_interpreter_type(cls, interpreter: Type[DataSetType]) -> None:
        if not issubclass(interpreter, cls.get_child_data_set_type()):
            raise TypeError(
                f'Interpreter must be subclass of {cls.get_child_data_set_type().__name__}'
            )

    @classmethod
    def _validate_interpreter_not_registered(cls, interpreter: Type[DataSetType]) -> None:
        if interpreter in cls._interpreter_registry:
            raise ValueError(f'Interpreter {interpreter.__name__} already registered')

    @classmethod
    def _add_interpreter_to_registry(cls, interpreter: Type[DataSetType]) -> None:
        cls._interpreter_registry.insert(0, interpreter)

    def _find_interpreter_instance(self, interpreter_type: Type[DataSetType]) -> DataSetType | None:
        for interpreter in self.data_sets.values():
            if isinstance(interpreter, interpreter_type):
                return interpreter
        return None
