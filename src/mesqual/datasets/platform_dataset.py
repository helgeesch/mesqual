from __future__ import annotations
from typing import Type, Any, Callable, Generic
from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect

import pandas as pd

from mesqual.datasets.dataset import Dataset
from mesqual.datasets.dataset_collection import DatasetLinkCollection
from mesqual.datasets.describe import Availability
from mesqual.datasets.source_inventory import SourceInventory
from mesqual.typevars import FlagType, DatasetType, DatasetConfigType, FlagIndexType
from mesqual.datasets.exceptions import AmbiguousOverrideError, UnknownFlagError
from mesqual.cache import Cache
from mesqual.flag.flag_index import FlagIndex
from mesqual.flag.flag_spec import FlagSpec
from mesqual.flag.layer_spec import LayerFlagSpec
from mesqual.flag.layers import AreaLayer, BorderLayer, LayerSet


@dataclass(frozen=True)
class InterpreterRegistration:
    """One registration of an interpreter on a platform dataset class.

    Attributes:
        interpreter: The interpreter class.
        dataset_class: The platform dataset class that the interpreter is registered on.
        override_interpreter: The interpreter that the override replaces.
            `None` for an interpreter that is not an override.
        version: Version of the override logic.
    """

    interpreter: type[Dataset]
    dataset_class: type
    override_interpreter: type[Dataset] | None = None
    version: int | None = None

    @property
    def is_override(self) -> bool:
        """`True` if this registration replaces another interpreter."""
        return self.override_interpreter is not None


Chain = tuple[InterpreterRegistration, ...]
"""The registrations of one original interpreter, from the original to the override that answers."""


class _ChainResolver:
    """Resolves one chain per original interpreter of `dataset_class` from the own `registrations` of each class.

    Each class along the MRO, base classes first, merges the candidate
    chains of its bases and then applies its own registrations: plain
    registrations first, then overrides. A plain registration of a link
    of a chain keeps the chain. More than one candidate per original means
    that overrides on sibling classes conflict.
    """

    def __init__(self, dataset_class: type, registrations: dict[type, list[InterpreterRegistration]]):
        self._dataset_class = dataset_class
        self._registrations = registrations

    def resolve(self) -> list[Chain]:
        """One chain per original interpreter.

        Raises:
            AmbiguousOverrideError: Overrides conflict.
            ValueError: An override names an interpreter that does not answer.
        """
        resolved: dict[type, dict[type, list[Chain]]] = {}
        for klass in reversed(self._dataset_class.__mro__):
            chains: dict[type, list[Chain]] = {}
            for base in klass.__bases__:
                for original, candidates in resolved[base].items():
                    chains[original] = self._longest_chains([*chains.get(original, ()), *candidates])
            self._apply_registrations(klass, chains)
            resolved[klass] = chains
        return [self._single(candidates) for candidates in resolved[self._dataset_class].values()]

    def _key(self, chain: Chain) -> tuple:
        return tuple((link.interpreter, link.version) for link in chain)

    def _longest_chains(self, candidates: list[Chain]) -> list[Chain]:
        """The chains that are not the start of another chain, each once."""
        keys = [self._key(chain) for chain in candidates]
        return [
            chain
            for i, (chain, key) in enumerate(zip(candidates, keys))
            if key not in keys[:i] and not any(other[: len(key)] == key != other for other in keys)
        ]

    def _apply_registrations(self, klass: type, chains: dict[type, list[Chain]]) -> None:
        registrations = self._registrations.get(klass, ())
        linked = {link.interpreter for candidates in chains.values() for chain in candidates for link in chain}
        for registration in registrations:
            if not registration.is_override and registration.interpreter not in linked:
                chains[registration.interpreter] = [(registration,)]
        overrides = [r for r in registrations if r.is_override]
        candidates = [(original, chain) for original, chains_of in chains.items() for chain in chains_of]
        replaced: dict[type, InterpreterRegistration] = {}
        for registration in overrides:
            original, chain = self._target_chain(klass, registration, candidates)
            if original in replaced:
                raise self._replaced_twice_error(replaced[original], registration, original, klass)
            replaced[original] = registration
            chains[original] = [chain + (registration,)]

    @staticmethod
    def _replaced_twice_error(
        first: InterpreterRegistration,
        second: InterpreterRegistration,
        replaced: type,
        klass: type,
    ) -> AmbiguousOverrideError:
        """The error for two overrides on `klass` that both replace `replaced`."""
        return AmbiguousOverrideError(
            f'{first.interpreter.__name__} and {second.interpreter.__name__} both replace '
            f'{replaced.__name__} on {klass.__name__}. Register one of them on a subclass to override the other.'
        )

    @staticmethod
    def _override_of_override_error(
        outer: InterpreterRegistration,
        inner: InterpreterRegistration,
        klass: type,
    ) -> ValueError:
        """The error for `outer` on `klass` that overrides `inner`, an override on `klass` too."""
        return ValueError(
            f'{outer.interpreter.__name__} on {klass.__name__} overrides {inner.interpreter.__name__}, '
            f'which is an override on {klass.__name__} too. Register one of them on a subclass.'
        )

    def _target_chain(
        self,
        klass: type,
        registration: InterpreterRegistration,
        candidates: list[tuple[type, Chain]],
    ) -> tuple[type, Chain]:
        """The original and the chain that `registration` extends. Raises if the override is not valid."""
        target = registration.override_interpreter
        prefix = f'{registration.interpreter.__name__} on {klass.__name__} overrides {target.__name__}'
        inner = list(
            dict.fromkeys(
                c[-1].interpreter.__name__ for _, c in candidates if target in {l.interpreter for l in c[:-1]}
            )
        )
        ends = [(original, chain) for original, chain in candidates if chain[-1].interpreter is target]
        if inner:
            names = ' or '.join(inner)
            raise ValueError(f'{prefix}, which {names} replaces on a parent class. Override {names} instead.')
        if not ends:
            raise ValueError(f'{prefix}, which is not registered on {klass.__name__} or its parents.')
        if len(ends) > 1:
            links = ' and '.join(
                f'{c[-1].dataset_class.__name__} (replaces {o.__name__}, version {c[-1].version})' for o, c in ends
            )
            raise AmbiguousOverrideError(
                f'{prefix}, which {links} register differently. Register {target.__name__} once on a common parent class.'
            )
        return ends[0]

    def _single(self, candidates: list[Chain]) -> Chain:
        """The one candidate. Raises `AmbiguousOverrideError` for overrides on sibling classes."""
        if len(candidates) == 1:
            return candidates[0]
        keys = [self._key(chain) for chain in candidates]
        common = next(i for i in range(len(keys[0]) + 1) if len({key[: i + 1] for key in keys}) > 1)
        replaced = candidates[0][common - 1].interpreter.__name__
        ends = [chain[-1] for chain in candidates]
        versions_differ = len({end.version for end in ends}) > 1
        version = (lambda end: f', version {end.version}') if versions_differ else (lambda end: '')
        names = ' and '.join(
            f'{end.interpreter.__name__} (on {end.dataset_class.__name__}{version(end)})' for end in ends
        )
        if len({end.interpreter for end in ends}) == 1 and versions_differ:
            advice = f'Register {ends[0].interpreter.__name__} with one version on a common parent class.'
        else:
            advice = f'Register an override of one of them on {self._dataset_class.__name__} or a class between.'
        raise AmbiguousOverrideError(
            f'{names} both replace {replaced}, and neither class is a subclass of the other. {advice}'
        )


class _Routes:
    """The chain of each spec name, and the specs of the answering interpreters.

    A layer spec on a spec that an override declares again is built on the
    spec of the override. A layer spec on a spec that only a replaced
    interpreter declares is left out.
    """

    def __init__(self, chains: list[Chain]):
        self._chains: dict[str, Chain] = {}
        self._specs: dict[str, FlagSpec] = {}
        for chain in chains:
            for spec in chain[-1].interpreter.flag_specs:
                other = self._chains.get(spec.name)
                if other is not None:
                    raise AmbiguousOverrideError(
                        f'{other[-1].interpreter.__name__} and {chain[-1].interpreter.__name__} both declare the spec '
                        f'{spec.name!r}. Let one override the other with register_interpreter(override_interpreter=...).'
                    )
                self._chains[spec.name] = chain
                self._specs[spec.name] = spec
        replaced_specs = (spec for chain in chains for link in chain[:-1] for spec in link.interpreter.flag_specs)
        self._replaced = {spec.name for spec in replaced_specs} - self._specs.keys()
        for spec in list(self._specs.values()):
            if self._current(spec) is None:
                del self._specs[spec.name], self._chains[spec.name]
                self._replaced.add(spec.name)

    def chain_of(self, flag: FlagType, index: FlagIndex) -> Chain:
        """The chain of the interpreter of `flag` in `index`. Empty without one."""
        if not index.has_spec(flag):
            return ()
        return self._chains.get(index.get_spec(flag).name, ())

    def flag_specs(self) -> list[FlagSpec]:
        """The specs of the answering interpreters, with layer specs built on the current spec of their base."""
        return list(self._specs.values())

    def _current(self, spec: FlagSpec) -> FlagSpec | None:
        """The spec of the name of `spec`. A layer spec on a replaced base is built again and stored.

        So all layer specs on one base share the base. `None` for a spec
        that only a replaced interpreter declares, and for a layer spec on it.
        """
        if spec.name not in self._specs:
            return None if spec.name in self._replaced else spec
        current = self._specs[spec.name]
        if isinstance(current, LayerFlagSpec):
            base = self._current(current.base)
            if base is None:
                return None
            if base is not current.base:
                current = self._specs[current.name] = current.on_base(base)
        return current


@dataclass
class InterpreterSignature:
    """The constructor arguments of an interpreter base class, without `self` and `parent_dataset`.

    The interpreter base class is `get_child_dataset_type()` of the platform
    dataset. The platform dataset passes each of these arguments to every
    interpreter it creates: the value of its own keyword argument of the same
    name, else the default. `*args` and `**kwargs` of the constructor are not
    in `args`.

    Attributes:
        args: Names of the arguments, in order.
        defaults: Default value of each argument, `None` for an argument without default.
    """

    args: tuple[str, ...]
    defaults: tuple[Any, ...]

    _NOT_PASSED = ('self', 'parent_dataset')
    _VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)

    @classmethod
    def from_interpreter(cls, interpreter: Type[Dataset]) -> InterpreterSignature:
        params = inspect.signature(interpreter.__init__).parameters.values()
        named = [p for p in params if p.name not in cls._NOT_PASSED and p.kind not in cls._VARIADIC]
        return cls(
            args=tuple(p.name for p in named),
            defaults=tuple(None if p.default is p.empty else p.default for p in named),
        )

    def interpreter_arguments(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """The arguments for each interpreter: each argument from `kwargs`, else its default."""
        return {arg: kwargs.get(arg, default) for arg, default in zip(self.args, self.defaults)}


class PlatformDataset(
    Generic[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    DatasetLinkCollection[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    ABC,
):
    """
    Base class for platform-specific datasets with an interpreter registry.

    A platform dataset holds one instance of each interpreter that answers
    (per original interpreter, the original or the override that replaces
    it) and routes each flag to the interpreter that declares its spec.

    Registry rules:
        - Each subclass has its own registry. `register_interpreter` on a
          class does not change its parent or sibling classes.
        - `get_registered_interpreters()` resolves the registry at call time
          along the MRO, so a parent registration made after the subclass
          definition reaches the subclass.
        - An instance resolves the registry once, in `__init__`, and builds
          the interpreter chains and the flag index there.
          `register_interpreter` raises for invalid arguments and for
          conflicts between overrides in the own registry of the class. All
          other spec and override errors are raised at instance creation
          and in the class lookups. It asks the interpreters
          for their `accepted_flags` at call time (`fetch`,
          `flag_is_accepted`, `accepted_flags`, `get_flags_by_interpreter`),
          so `accepted_flags` of an interpreter can read its siblings.
        - `datasets` and `get_registered_interpreters()` list the answering
          interpreters of base classes first, then in registration order.
        - Any `Dataset` subclass can be registered as interpreter, for
          example an `AreaFlagInterpreter`. `get_child_dataset_type()` gives
          the constructor arguments that the platform passes to them.
        - Each spec has exactly one answering interpreter. Two answering
          interpreters that declare the same spec name raise
          `AmbiguousOverrideError` at instance creation; the message names
          both interpreters.
        - A platform dataset class has no specs of its own. A subclass that
          sets `flag_specs` raises `TypeError` at class creation.
        - Each interpreter is named `f'{platform.name}.{Interpreter.__qualname__}'`.
          Two registered interpreters with the same `__qualname__` raise
          `ValueError` at instance creation; the message names their modules.

    Source inventory and default attributes:
        A subclass implements `build_source_inventory` with cheap reads of
        its source. `source_inventory` builds it once per instance. The
        interpreters read it in their `_is_available`. `_default_attributes`
        adds attributes from the inventory or model flags; explicit
        attributes win. `_default_attributes` runs in `__init__`: set the
        fields that `build_source_inventory` reads before calling
        `super().__init__`.

    Overrides:
        `register_interpreter(Override, override_interpreter=Original, version=1)`
        replaces a whole interpreter. The override answers its own
        `flag_specs` and reuses code by subclassing and `super()._fetch`.
        Layer flags that fetch a flag of the override through
        `self.parent_dataset.fetch(...)` get the override result.
        `get_answering_interpreter(flag)` returns the interpreter that
        answers now, so a package can subclass it to build on it.

    Processing after the interpreter chain:
        `_fetch` is the hook to process the result of each flag after its
        interpreter chain: a subclass overrides it and changes the result
        of `super()._fetch(flag, effective_config, **params)`. The rules
        are in [Interpreter overrides](../architecture/other-contracts.md#interpreter-overrides),
        item "After the chain".

    Examples:
        ```python
        class MyPlatformDataset(PlatformDataset[MyInterpreter, ...]):
            def build_source_inventory(self):
                return MyInventory(fingerprint=hash_of_file_keys(self.folder), files=...)

            @classmethod
            def get_child_dataset_type(cls):
                return MyInterpreter

        @MyPlatformDataset.register_interpreter
        class NodeInterpreter(MyInterpreter):
            flag_specs = (NODE_MODEL_SPEC, NODE_PRICE_SPEC)
            ...
        ```

        A study dataset that rounds the node price after the interpreter chain:

        >>> from mesqual.testing import DummyDataset, DummyScenario
        >>> class RoundedDataset(DummyDataset):
        ...     def _fetch(self, flag, effective_config, **params):
        ...         result = super()._fetch(flag, effective_config, **params)
        ...         return result.round(0) if flag == 'Node.Results.price' else result
        >>> RoundedDataset(DummyScenario()).fetch('Node.Results.price').iloc[0].tolist()
        [38.0, 44.0, 46.0, 55.0]
    """

    _resolves_flag_params = True
    _is_leaf = True
    _reads_source = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'flag_specs' in cls.__dict__:
            raise TypeError(
                f'{cls.__name__} sets flag_specs. A platform dataset has no specs of its own; '
                f'declare them on a registered interpreter.'
            )

    def __init__(
        self,
        name: str = None,
        attributes: dict = None,
        cache: Cache = None,
        config: DatasetConfigType = None,
        **kwargs: Any,
    ):
        """Create the interpreters of this class and build the flag index.

        The keyword arguments are the own arguments below and the constructor
        arguments of the interpreter base `get_child_dataset_type()`, without
        `parent_dataset`. The platform dataset passes
        these to every interpreter it creates: the given value, else the
        default. A base argument with the name of an own argument (`name`,
        `attributes`, `cache`, `config`) binds to the own argument, so the
        interpreters get its default. Any other keyword argument raises
        `TypeError`.

        A subclass with arguments of its own takes them in its own `__init__`
        and passes only the others to `super().__init__`, or makes them
        constructor arguments of its interpreter base. For example,
        `DummyDataset` takes a scenario and passes the interpreter argument
        `source`.

        Args:
            name: Dataset name. Default: the class name and the object id.
            attributes: Attributes of the dataset. Default attributes from
                `_default_attributes` are added; explicit attributes win.
            cache: Optional cache of fetch results.
            config: Optional instance config.
            **kwargs: The constructor arguments of the interpreter base.

        Raises:
            TypeError: A keyword argument is neither an own argument nor a
                constructor argument of the interpreter base. The message
                names the unknown and the accepted arguments.
        """
        signature = InterpreterSignature.from_interpreter(self.get_child_dataset_type())
        self._raise_for_unknown_arguments(signature, kwargs)
        chains = self._resolved_chains()
        self._routes = _Routes(chains)
        self._interpreter_instances: dict[type[DatasetType], DatasetType] = {}
        self._source_inventory: SourceInventory | None = None
        super().__init__(
            datasets=[],
            name=name,
            attributes=attributes,
            cache=cache,
            config=config,
        )
        interpreter_args = signature.interpreter_arguments(kwargs)
        self._interpreter_instances = {
            chain[-1].interpreter: self._new_interpreter(chain[-1].interpreter, interpreter_args) for chain in chains
        }
        self.add_datasets(self._interpreter_instances.values())
        self.flag_index.check_references(self)
        self._add_default_attributes()

    @property
    def flag_index(self) -> FlagIndex:
        """The flag index: built at construction from the specs of the answering interpreters and the layers of the class."""
        return super().flag_index

    def _build_flag_index(self) -> FlagIndex:
        """Return the index of the specs of the answering interpreters and the layers of the class."""
        return FlagIndex(self._routes.flag_specs(), self._class_layers())

    @abstractmethod
    def build_source_inventory(self) -> SourceInventory:
        """Build the source inventory of this dataset. Read file names and small tables only.

        Returns:
            A `SourceInventory`, usually of a platform subclass with the
                fields that the `_is_available` implementations of the
                interpreters and `_default_attributes` need.
        """

    @property
    def source_inventory(self) -> SourceInventory:
        """The source inventory, built once per instance with `build_source_inventory`, then kept in memory."""
        if self._source_inventory is None:
            self._source_inventory = self.build_source_inventory()
        return self._source_inventory

    def _default_attributes(self) -> dict:
        """Hook: attributes that the platform adds to each dataset, for example the number of runs or the first date.

        Computed from the source inventory or model flags, never from time
        series flags. Values must be `bool`, `int`, `float` or `str`.
        Explicit attributes win. Default: none.
        """
        return {}

    def _add_default_attributes(self) -> None:
        explicit = self._attributes
        self._attributes = {}
        self._add_attributes({**self._default_attributes(), **explicit})

    def _checked_availability(self, flag: FlagType) -> Availability:
        """Availability from the `_is_available` of the answering interpreter.

        If it returns `None`, the flag is unknown, and available after its
        first successful fetch from this dataset.
        """
        interpreter = self._answering_instance(flag)
        name = type(interpreter).__qualname__
        checked = self._availability_from_hook(interpreter._is_available(flag), f'{name}._is_available')
        if checked is not None:
            return checked
        if str(flag) in self._fetched_flags:
            return Availability(True, f'fetched successfully from {self.name}')
        return Availability(None, f'no availability check in {name} and the flag was not fetched yet')

    def _flag_chain_names(self, flag: FlagType) -> tuple[str, ...]:
        return tuple(registration.interpreter.__qualname__ for registration in self._chain_of(flag))

    @classmethod
    def _declared_flag_specs(cls) -> tuple[FlagSpec, ...]:
        """The specs of the answering interpreters. A platform class has no specs of its own."""
        return tuple(cls._class_routes().flag_specs())

    @classmethod
    def _class_routes(cls) -> _Routes:
        return _Routes(cls._resolved_chains())

    @staticmethod
    def _check_unique_qualnames(chains: list[Chain]) -> None:
        """Raise `ValueError` if two interpreters of the chains have the same `__qualname__`.

        The qualname names the interpreter instance and its version identity link.
        """
        seen: dict[str, type] = {}
        for interpreter in (link.interpreter for chain in chains for link in chain):
            other = seen.setdefault(interpreter.__qualname__, interpreter)
            if other is not interpreter:
                if other.__module__ == interpreter.__module__:
                    origin = f'two different classes in module {interpreter.__module__!r}, for example from a factory function'
                else:
                    origin = f'one in module {other.__module__!r}, one in module {interpreter.__module__!r}'
                raise ValueError(
                    f'Two interpreters have the qualname {interpreter.__qualname__!r}: {origin}. Rename one of them.'
                )

    def _new_interpreter(self, interpreter: type[DatasetType], interpreter_args: dict) -> DatasetType:
        """Create the instance of an interpreter, named `f'{self.name}.{interpreter.__qualname__}'`.

        The name is stable across sessions, so records of two platform
        datasets with the same name have equal keys.
        """
        instance = interpreter(**interpreter_args, parent_dataset=self)
        instance.name = f'{self.name}.{interpreter.__qualname__}'
        return instance

    @classmethod
    def register_interpreter(
        cls,
        interpreter: Type[DatasetType] | None = None,
        *,
        override_interpreter: Type[DatasetType] | None = None,
        version: int | None = None,
    ) -> Type[DatasetType] | Callable[[Type[DatasetType]], Type[DatasetType]]:
        """Register an interpreter, or an override that replaces a whole interpreter.

        Usable as a decorator, with or without keyword arguments. The
        registration applies to this class and its subclasses only.

        An override replaces `override_interpreter` on the dataset: the
        replaced interpreter is not created, and the override answers its
        own `flag_specs`. A subclass of the replaced interpreter inherits
        its specs and can declare a spec again by name with other content.
        Per original interpreter, the registration on the most derived
        class answers. Reuse code by subclassing and `super()._fetch`; to
        build on what answers now, subclass `get_answering_interpreter(flag)`.

        Args:
            interpreter: The interpreter class.
            override_interpreter: The interpreter that the override replaces:
                an original, or the override that answers on this class or
                a parent class.
            version: Version of the override logic. Required with
                `override_interpreter`, not allowed without it. Part of the
                version identity of each flag of the override.

        Returns:
            The interpreter class, or a decorator if `interpreter` is `None`.

        Raises:
            TypeError: The interpreter or `override_interpreter` is not a
                `Dataset` subclass.
            ValueError: `version` is missing with `override_interpreter`, or
                given without it.
            ValueError: The interpreter names itself as `override_interpreter`.
            ValueError: The interpreter is already registered on this class
                with other arguments.
            ValueError: The override overrides an override on this class,
                or an override on this class overrides it.
            AmbiguousOverrideError: Another override on this class names the
                same `override_interpreter`.

        A failed registration does not change the registry. At instance
        creation and in the class lookups, `ValueError` if
        `override_interpreter` does not answer on the class or a parent
        class (the message names the interpreter to override instead), and
        `AmbiguousOverrideError` if two overrides on one class replace one
        original through different interpreters, or two overrides on
        sibling classes replace one original without an override on a
        common subclass.

        Examples:
            The override declares the price spec of `DummyTimeSeriesInterpreter`
            again, with the extra parameter `cap`, and keeps its other specs:

            >>> import dataclasses
            >>> from mesqual.flag import Dimension
            >>> from mesqual.testing import DummyDataset, DummyScenario
            >>> from mesqual.testing.dataset import DummyTimeSeriesInterpreter as Original
            >>> class StudyDataset(DummyDataset):
            ...     pass
            >>> CAP = Dimension('cap', 'Price cap in EUR/MWh.', default=1000)
            >>> CAPPED_PRICE_SPEC = dataclasses.replace(Original.PRICE_SPEC, params=Original.PRICE_SPEC.params + (CAP,))
            >>> @StudyDataset.register_interpreter(override_interpreter=Original, version=1)
            ... class CappedPrices(Original):
            ...     flag_specs = (CAPPED_PRICE_SPEC,) + Original.flag_specs[1:]
            ...     def _fetch(self, flag, effective_config, cap=1000, **params):
            ...         result = super()._fetch(flag, effective_config, **params)
            ...         return result.clip(upper=cap) if flag == 'Node.Results.price' else result
            >>> float(StudyDataset(DummyScenario()).fetch('Node.Results.price', cap=40).max().max())
            40.0
        """
        if interpreter is None:

            def decorator(interpreter_class: Type[DatasetType]) -> Type[DatasetType]:
                return cls.register_interpreter(
                    interpreter_class, override_interpreter=override_interpreter, version=version
                )

            return decorator

        registration = cls._build_registration(interpreter, override_interpreter, version)
        own_registrations = cls._own_registrations()
        for existing in own_registrations:
            if existing.interpreter is interpreter:
                if existing != registration:
                    raise ValueError(
                        f'{interpreter.__name__} is already registered on {cls.__name__} with other arguments.'
                    )
                return interpreter
        cls._check_own_overrides(registration, own_registrations)
        own_registrations.append(registration)
        return interpreter

    @classmethod
    def register_area_layer(cls, layer: AreaLayer) -> None:
        """Register an area layer on this class and its subclasses.

        The layer applies to every layer spec whose base is on
        `layer.base_model`. An instance checks the layer against the model
        specs at creation: the spec of the base model needs exactly one
        membership (`ModelFlagSpec.memberships`) to the layer model, else
        `ValueError`.

        Raises:
            TypeError: `layer` is not an `AreaLayer`.

        Examples:
            A layer of price zones, here one zone per country through the
            membership `country` of `'Node.Model'`:

            >>> from mesqual.flag import AreaLayer
            >>> from mesqual.testing import DummyDataset, DummyScenario
            >>> class StudyDataset(DummyDataset):
            ...     pass
            >>> StudyDataset.register_area_layer(AreaLayer(
            ...     name='Zone', index='Zone', base_model='Node.Model', layer_model='Country.Model',
            ...     description='Price zones, one per country.',
            ... ))
            >>> StudyDataset(DummyScenario()).fetch('Zone.Results.price').columns.tolist()
            ['C1', 'C2']
        """
        cls._register_layer(layer, AreaLayer, '_area_layer_registry')

    @classmethod
    def register_border_layer(cls, layer: BorderLayer) -> None:
        """Register a border layer on this class and its subclasses.

        Raises:
            TypeError: `layer` is not a `BorderLayer`.

        Examples:
            >>> from mesqual.flag import BorderLayer
            >>> from mesqual.testing import DummyDataset, DummyScenario
            >>> class StudyDataset(DummyDataset):
            ...     pass
            >>> StudyDataset.register_border_layer(BorderLayer(
            ...     name='CountryPair', index='CountryPair', area_layer='Country', edge_model='Line.Model',
            ...     description='Borders between two countries, crossed by lines.',
            ... ))
            >>> StudyDataset(DummyScenario()).fetch('CountryPair.Model').index.tolist()
            ['C1 - C2', 'C2 - C1']
        """
        cls._register_layer(layer, BorderLayer, '_border_layer_registry')

    @classmethod
    def get_area_layers(cls) -> list[AreaLayer]:
        """Return the area layers of this class and its parents, base classes first."""
        return cls._resolve_layers('_area_layer_registry')

    @classmethod
    def get_border_layers(cls) -> list[BorderLayer]:
        """Return the border layers of this class and its parents, base classes first."""
        return cls._resolve_layers('_border_layer_registry')

    @classmethod
    def _class_layers(cls) -> LayerSet:
        return LayerSet(cls.get_area_layers(), cls.get_border_layers())

    @classmethod
    def _register_layer(cls, layer: AreaLayer | BorderLayer, layer_type: type, registry: str) -> None:
        if not isinstance(layer, layer_type):
            raise TypeError(f'Expected a {layer_type.__name__}, got {type(layer).__name__}.')
        if registry not in cls.__dict__:
            setattr(cls, registry, [])
        own = cls.__dict__[registry]
        if layer not in own:
            own.append(layer)

    @classmethod
    def _resolve_layers(cls, registry: str) -> list:
        layers = []
        for klass in reversed(cls.__mro__):
            layers += [layer for layer in klass.__dict__.get(registry, []) if layer not in layers]
        return layers

    @classmethod
    def get_registered_interpreters(cls) -> list[Type[DatasetType]]:
        """Return the interpreters that a dataset of this class creates: per original, the one that answers.

        Resolved at call time along the MRO, base classes first. A replaced
        interpreter is not in the list. Raises as at instance creation.
        """
        return [chain[-1].interpreter for chain in cls._resolved_chains()]

    @classmethod
    def get_answering_interpreter(cls, flag: FlagType) -> Type[DatasetType]:
        """Return the interpreter class that answers a flag on this class, after all registered overrides.

        The lookup reads the specs and the registry, resolved along the MRO
        at call time. It does not check if a source provides the flag: an
        interpreter whose `accepted_flags` leaves out the flag on an
        instance still answers it here. Subclass the result to build an
        override on what answers now.

        Args:
            flag: A flag of a spec of a registered interpreter.

        Returns:
            The interpreter that answers the spec of the flag.

        Raises:
            UnknownFlagError: No registered interpreter declares the flag.
            AmbiguousOverrideError: Two interpreters declare the spec, or
                overrides conflict, as at instance creation.
            ValueError: An override is not valid, as at instance creation.

        Examples:
            >>> from mesqual.testing import DummyDataset
            >>> class StudyDataset(DummyDataset):
            ...     pass
            >>> answering = StudyDataset.get_answering_interpreter('Node.Results.price')
            >>> @StudyDataset.register_interpreter(override_interpreter=answering, version=1)
            ... class PriceFloor(answering):
            ...     def _fetch(self, flag, effective_config, **params):
            ...         return super()._fetch(flag, effective_config, **params).clip(lower=0)
            >>> StudyDataset.get_answering_interpreter('Node.Results.price').__name__
            'PriceFloor'
        """
        routes = cls._class_routes()
        chain = routes.chain_of(flag, FlagIndex(routes.flag_specs(), cls._class_layers()))
        if not chain:
            raise UnknownFlagError(f'No interpreter registered on {cls.__name__} declares the flag {flag!r}.')
        return chain[-1].interpreter

    def get_interpreter_instance(self, interpreter_type: Type[DatasetType]) -> DatasetType:
        """Return the interpreter instance of exactly this class (`type(instance) is interpreter_type`).

        Args:
            interpreter_type: Interpreter class.

        Returns:
            The interpreter instance of this dataset.

        Raises:
            ValueError: No interpreter of this class is registered.
        """
        interpreter = self._interpreter_instances.get(interpreter_type)
        if interpreter is None:
            raise ValueError(f'No interpreter instance found for type {interpreter_type.__name__}')
        return interpreter

    def get_flags_by_interpreter(self) -> dict[Type[DatasetType], set[FlagType]]:
        """Return the flags each interpreter answers. A replaced interpreter is not on the dataset."""
        return {
            type(interpreter): set(interpreter.accepted_flags) for interpreter in self._interpreter_instances.values()
        }

    @property
    def accepted_flags(self) -> set[FlagType]:
        """The flags in the `accepted_flags` of each interpreter."""
        return set().union(*self.get_flags_by_interpreter().values())

    def flag_is_accepted(self, flag: FlagType) -> bool:
        """`True` if the interpreter that declares the spec of the flag accepts it."""
        interpreter = self._answering_instance(flag)
        return interpreter is not None and interpreter.flag_is_accepted(flag)

    def _fetch(self, flag: FlagType, effective_config: DatasetConfigType, **kwargs) -> pd.Series | pd.DataFrame:
        child = self._answering_instance(flag)
        if child is None:
            raise KeyError(f"Key '{flag}' not recognized by any interpreter of {self.name}.")
        return child.fetch(flag, effective_config, **self._kwargs_for_child(child, flag))

    def version_identity(self, flag: FlagType) -> tuple:
        """Return one `(class name, version)` pair per link of the interpreter chain of the flag.

        The first link is the original interpreter with the version of the
        spec in the flag index. Each override along the dataset class
        hierarchy adds its `register_interpreter(version=)`. A layer spec
        adds the identity of its aggregation after the first link.

        Examples:
            >>> high_demand = study.scen.get_dataset('high_demand')
            >>> high_demand.version_identity('Node.Results.price')
            (('DummyTimeSeriesInterpreter', 1),)
            >>> high_demand.version_identity('Country.Results.price')
            (('DummyAreaFlags', 1), ('WeightedMean', {'weight': 'Node.Results.demand_total', 'version': 2}))
        """
        chain = self._chain_of(flag)
        if not chain:
            return ()
        spec = self.flag_index.get_spec(flag)
        first = ((chain[0].interpreter.__qualname__, spec.version),) + self._aggregation_links(spec)
        return first + tuple(
            (registration.interpreter.__qualname__, registration.version) for registration in chain[1:]
        )

    @staticmethod
    def _aggregation_links(spec: FlagSpec) -> tuple:
        """The aggregation link of a layer spec, right after its interpreter, else nothing."""
        return (spec.aggregation.identity(),) if isinstance(spec, LayerFlagSpec) else ()

    def _answering_instance(self, flag: FlagType) -> DatasetType | None:
        chain = self._chain_of(flag)
        return self._interpreter_instances[chain[-1].interpreter] if chain else None

    def _chain_of(self, flag: FlagType) -> Chain:
        """The interpreter chain of `flag`, from the original to the answering interpreter. Empty without one."""
        return self._routes.chain_of(flag, self.flag_index)

    @classmethod
    def _own_registrations(cls) -> list[InterpreterRegistration]:
        if '_interpreter_registry' not in cls.__dict__:
            cls._interpreter_registry = []
        return cls._interpreter_registry

    @classmethod
    def _check_own_overrides(cls, registration: InterpreterRegistration, own: list[InterpreterRegistration]) -> None:
        """Raises if `registration` conflicts with an override in the own registry of the class."""
        if not registration.is_override:
            return
        for existing in (r for r in own if r.is_override):
            if existing.interpreter is registration.override_interpreter:
                raise _ChainResolver._override_of_override_error(registration, existing, cls)
            if existing.override_interpreter is registration.interpreter:
                raise _ChainResolver._override_of_override_error(existing, registration, cls)
            if existing.override_interpreter is registration.override_interpreter:
                raise _ChainResolver._replaced_twice_error(
                    existing, registration, registration.override_interpreter, cls
                )

    @classmethod
    def _resolved_chains(cls) -> list[Chain]:
        """One chain per original interpreter, resolved along the MRO at call time. Raises on conflicts."""
        registrations = {
            klass: klass._own_registrations() for klass in cls.__mro__ if issubclass(klass, PlatformDataset)
        }
        chains = _ChainResolver(cls, registrations).resolve()
        cls._check_unique_qualnames(chains)
        return chains

    @classmethod
    def _build_registration(
        cls,
        interpreter: Type[DatasetType],
        override_interpreter: Type[DatasetType] | None,
        version: int | None,
    ) -> InterpreterRegistration:
        cls._validate_interpreter_type(interpreter)
        if override_interpreter is not None:
            cls._validate_interpreter_type(override_interpreter)
        if override_interpreter is not None and version is None:
            raise ValueError(f'{interpreter.__name__}: version is required with override_interpreter.')
        if override_interpreter is None and version is not None:
            raise ValueError(f'{interpreter.__name__}: version needs override_interpreter.')
        if override_interpreter is interpreter:
            raise ValueError(
                f'{interpreter.__name__} names itself as override_interpreter. Register it without '
                f'override_interpreter, or name the interpreter that it replaces.'
            )
        return InterpreterRegistration(interpreter, cls, override_interpreter=override_interpreter, version=version)

    def _raise_for_unknown_arguments(self, signature: InterpreterSignature, kwargs: dict[str, Any]) -> None:
        """Raise `TypeError` for a keyword argument that is neither an own argument nor an interpreter argument."""
        unknown = sorted(set(kwargs) - set(signature.args))
        if unknown:
            own = inspect.signature(PlatformDataset.__init__).parameters.values()
            named = {p.name for p in own if p.name != 'self' and p.kind is not p.VAR_KEYWORD}
            accepted = sorted(named | set(signature.args))
            raise TypeError(
                f'{type(self).__name__} got unknown keyword arguments {unknown}. '
                f'Accepted by PlatformDataset.__init__: {accepted} (its own arguments and the constructor '
                f'arguments of {self.get_child_dataset_type().__name__}).'
            )

    @classmethod
    def _validate_interpreter_type(cls, interpreter: Type[DatasetType]) -> None:
        if not cls._accepts_child_type(interpreter):
            raise TypeError(f'Interpreter must be a Dataset subclass, got {interpreter!r}.')

    @classmethod
    def _accepts_child_type(cls, dataset_type: type[Dataset]) -> bool:
        """`True` for any `Dataset` subclass: an interpreter need not subclass `get_child_dataset_type()`."""
        return isinstance(dataset_type, type) and issubclass(dataset_type, Dataset)
