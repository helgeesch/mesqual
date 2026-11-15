"""Dimensions of flag specs: variants and parameters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Hashable, Mapping

import numpy as np

from mesqual.flag.exceptions import FlagUnavailableError

if TYPE_CHECKING:
    from mesqual.datasets.dataset import Dataset

logger = logging.getLogger(__name__)


class _Required:
    """Type of the `REQUIRED` marker. There is one instance."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __reduce__(self):
        return _Required, ()

    def __repr__(self) -> str:
        return 'REQUIRED'


REQUIRED = _Required()
"""Marker for a spec field or a parameter default that has no default value."""


@dataclass(frozen=True)
class Dimension:
    """One dimension: a variant of a flag spec, or a parameter of a flag spec or a dataset class.

    The place of the dimension decides its role:

    - `FlagSpec(variants=...)`: a variant. It is a placeholder in the flag
      name, and each value is a flag of its own.
    - `FlagSpec(params=...)`: a flag parameter. It is a setting of one flag,
      given as a keyword argument of `fetch`.
    - `Dataset.fetch_params`: a dataset parameter. It is a setting of every
      fetch of that dataset class, for example `comparison_type` of a
      comparison.

    A parameter of both kinds resolves the same way: the keyword argument of
    `fetch`, then `param_defaults` of the effective config, then `default`.
    A value outside `values` raises `ValueError`.

    Attributes:
        name: Name of the dimension. For a variant, also the placeholder in
            the flag name. For a parameter, the keyword argument of `fetch`.
        description: What the dimension selects.
        values: Allowed values, each with a short description. A function of
            the dataset gives values per dataset (a flag family). A function
            that fetches an unavailable flag gives no values. `None`:
            parameters only, any value that `validate` accepts.
        default: Parameters only. `REQUIRED`: the caller or the config must
            give a value.
        validate: Parameters only. Raises `ValueError` for an invalid value,
            for open value sets.

    Examples:
        A variant dimension and a parameter:

        >>> FUEL_TYPE = Dimension('fuel_type', 'Fuel type of the supply.', {
        ...     'res': 'renewable sources',
        ...     'coal': 'coal plants',
        ...     'gas': 'gas plants',
        ... })
        >>> SOLUTION = Dimension('solution', 'Solution of the run to read.', {
        ...     'first': 'the first solution of the run',
        ...     'final': 'the final solution of the run',
        ... }, default='final')

        The synthetic platform has `SOLUTION` in the `params` of the spec of
        `'Node.Results.price'`:

        - `dataset.fetch(flag)` gives `_fetch` the value `solution='final'` (default).
        - With `param_defaults={'solution': 'first'}` in the config, the same
          call gives `solution='first'`.
        - `dataset.fetch(flag, solution='first')` gives `solution='first'`
          (keyword argument).
        - `dataset.fetch(flag, solution='best')` raises `ValueError`.
    """

    name: str
    description: str
    values: Mapping[Hashable, str] | Callable[[Dataset], Mapping[Hashable, str]] | None = None
    default: Any = REQUIRED
    validate: Callable[[Any], None] | None = None

    def __post_init__(self):
        if isinstance(self.values, Mapping):
            object.__setattr__(self, 'values', MappingProxyType(dict(self.values)))
        elif self.values is not None and not callable(self.values):
            raise TypeError(f'Dimension {self.name!r}: values must be a mapping, a function or None.')

    @property
    def is_resolved_per_dataset(self) -> bool:
        """`True` if the values come from a function of the dataset."""
        return self.values is not None and not isinstance(self.values, Mapping)

    @property
    def has_default(self) -> bool:
        """`True` if the dimension has a default value."""
        return self.default is not REQUIRED

    def values_for(self, dataset: Dataset | None = None) -> Mapping[Hashable, str] | None:
        """Return the allowed values with their descriptions.

        Args:
            dataset: Dataset that resolves values given as a function.

        Returns:
            The values, or `None` for an open parameter. No values if the
                values function fetches a flag that is not available on
                `dataset` (`FlagUnavailableError`).

        Raises:
            ValueError: The values are resolved per dataset and `dataset` is `None`.
        """
        return self._values_and_unavailable_error(dataset)[0]

    def _values_and_unavailable_error(
        self,
        dataset: Dataset | None,
    ) -> tuple[Mapping[Hashable, str] | None, FlagUnavailableError | None]:
        """The values, and the `FlagUnavailableError` that left them empty, else `None`."""
        if not self.is_resolved_per_dataset:
            return self.values, None
        if dataset is None:
            raise ValueError(f'Dimension {self.name!r} is resolved per dataset. Pass a dataset.')
        try:
            return MappingProxyType(dict(self.values(dataset))), None
        except FlagUnavailableError as error:
            logger.debug('Values of %r are empty on %s: %s', self.name, dataset.name, error)
            return MappingProxyType({}), error

    def check_value(self, value: Any, dataset: Dataset | None = None) -> None:
        """Raise `ValueError` if `value` is not allowed or the values of the dimension are unavailable.

        Args:
            value: Value to check.
            dataset: Dataset that resolves values given as a function.

        A `bool` or `numpy.bool_` matches only a `bool` or `numpy.bool_`
        value, and a number matches only a number that is not a boolean. So `False` is not a value of a dimension
        with the values `0` and `1`.

        Raises:
            FlagUnavailableError: The values function fetches an unavailable
                flag. The reason names the dimension and that flag.
            ValueError: The value is not in `values`, or `validate` rejects it.
        """
        values, unavailable_error = self._values_and_unavailable_error(dataset)
        if unavailable_error is not None:
            reason = (
                f'values of {self.name!r} need the unavailable flag {unavailable_error.flag}: '
                f'{unavailable_error.reason}'
            )
            raise FlagUnavailableError(unavailable_error.flag, unavailable_error.dataset, reason) from unavailable_error
        if values is not None and not self._is_one_of(value, values):
            raise ValueError(f'{value!r} is not a value of {self.name!r}. Allowed: {list(values)}.')
        if self.validate is not None:
            self.validate(value)

    def resolve(self, given: Mapping[str, Any], dataset: Dataset | None = None) -> Any:
        """Return the value of this parameter: the value in `given`, else `default`, then check it.

        Args:
            given: Given values by parameter name. Other names are ignored.
            dataset: Dataset that resolves values given as a function. With
                `None` and such values, neither the values nor `validate`
                are checked.

        Returns:
            The value.

        Raises:
            FlagUnavailableError: The values function fetches an unavailable
                flag, as in `check_value`.
            ValueError: `given` has no value and the default is `REQUIRED`, or
                `check_value` rejects the value.

        Examples:
            `solution` of the node price has the values `'first'` and `'final'`, default `'final'`:

            >>> solution = dataset.flag_index.get_spec('Node.Results.price').params[0]
            >>> solution.resolve({})
            'final'
            >>> solution.resolve({'solution': 'first'})
            'first'
        """
        if self.name in given:
            value = given[self.name]
        elif self.has_default:
            value = self.default
        else:
            raise ValueError(f'parameter {self.name!r} is required.')
        if dataset is not None or not self.is_resolved_per_dataset:
            self.check_value(value, dataset)
        return value

    @staticmethod
    def _is_one_of(value: Any, values: Mapping[Hashable, str]) -> bool:
        if value not in values:
            return False
        if not (Dimension._is_bool(value) or value in (0, 1)):
            return True
        return any(v == value and Dimension._is_bool(v) == Dimension._is_bool(value) for v in values)

    @staticmethod
    def _is_bool(value: Any) -> bool:
        return isinstance(value, (bool, np.bool_))

    def describe_value(self, value: Hashable, dataset: Dataset | None = None) -> str:
        """Return the description of one value, or `''` for an open parameter."""
        values = self.values_for(dataset)
        return '' if values is None else values[value]

    def same_as(self, other: Dimension) -> bool:
        """`True` if both have the same name, values and default.

        Values given as a function compare by identity.
        """
        return (
            self.name == other.name
            and self._comparable(self.values) == self._comparable(other.values)
            and self.default == other.default
        )

    @staticmethod
    def _comparable(values: Any) -> Any:
        if isinstance(values, Mapping):
            return tuple(values.items())
        return id(values) if values is not None else None


class VariantValues:
    """The variant values of one flag, with one attribute per dimension.

    A wrong attribute raises `AttributeError`.

    Examples:
        >>> flow_spec = dataset.flag_index.get_spec('Line.Results.flow_sent_net')
        >>> variant = flow_spec.variant_of('Line.Results.flow_sent_net')
        >>> variant.side, variant.direction
        ('sent', 'net')
        >>> variant.as_dict()
        {'side': 'sent', 'direction': 'net'}
    """

    __slots__ = ('_values',)

    def __init__(self, values: Mapping[str, Hashable]):
        object.__setattr__(self, '_values', dict(values))

    def __getattr__(self, name: str) -> Hashable:
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(f'No variant dimension {name!r}. Dimensions: {list(self._values)}.') from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError('VariantValues is read-only.')

    def as_dict(self) -> dict[str, Hashable]:
        """Return the values as a dict, dimension name to value."""
        return dict(self._values)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VariantValues) and self._values == other._values

    def __hash__(self) -> int:
        return hash(tuple(self._values.items()))

    def __repr__(self) -> str:
        return f'VariantValues({", ".join(f"{k}={v!r}" for k, v in self._values.items())})'
