"""The base class of the dataset checks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import ClassVar

import pandas as pd

from mesqual.datasets import Availability, Dataset
from mesqual.flag.exceptions import FlagUnavailableError, UnknownFlagError
from mesqual.typevars import FlagType
from mesqual.validation.comparison import Comparison
from mesqual.validation.result import CheckResult, CheckStatus


class DatasetCheck(ABC):
    """Base class of a check: a rule on one dataset, run by `run(dataset) -> CheckResult`.

    A subclass implements `_run`. It declares the flags it reads in `FLAGS`, or
    overrides `flags` when they depend on the instance. Settings, for example
    tolerances, are constructor arguments. A subclass can be a dataclass.
    A subclass keeps no state between runs: `run` gives the same result for
    the same dataset and settings.

    `run` returns `SKIP` when a declared flag is accepted but not available on
    the dataset (`Dataset.availability`). A flag with unknown availability
    counts as available: only a fetch tells. A `FlagUnavailableError` raised in
    `_run` also gives `SKIP`. A declared flag that the dataset does not accept
    raises `UnknownFlagError`, so a typo in a flag name is an `ERROR` in the
    validator, not a `SKIP`. A subclass with more needs, for example a dataset
    class or a model input, overrides the hook `_unmet(dataset)` and calls
    `super()._unmet(dataset)`.

    Examples:
        >>> from mesqual.validation import CheckStatus, Comparison
        >>> class PricesNotNegative(DatasetCheck):
        ...     FLAGS = ('Node.Results.price',)
        ...
        ...     def _run(self, dataset):
        ...         prices = Comparison.long(dataset.fetch('Node.Results.price'))
        ...         negative = prices[prices < 0].rename('price').reset_index()
        ...         if len(negative):
        ...             return self.result(CheckStatus.FAIL, f'{len(negative)} negative node prices.', negative)
        ...         return self.result(CheckStatus.PASS, f'{len(prices)} node prices, none negative.')
        >>> result = PricesNotNegative().run(dataset)
        >>> result.status.name, result.summary
        ('PASS', '1328 node prices, none negative.')

    Attributes:
        FLAGS: The flags the check reads.
    """

    FLAGS: ClassVar[tuple[FlagType, ...]] = ()

    @property
    def name(self) -> str:
        """The check name: the class name. A validator holds one check per name."""
        return type(self).__name__

    @property
    def flags(self) -> tuple[FlagType, ...]:
        """The flags the check reads: `FLAGS`."""
        return tuple(self.FLAGS)

    def run(self, dataset: Dataset) -> CheckResult:
        """Run the check on a dataset.

        Returns:
            The result. `SKIP` if a need is not met (`_unmet`) or the check
                fetches an unavailable flag.

        Raises:
            UnknownFlagError: The dataset does not accept a flag of `flags`.
        """
        try:
            return self._unmet(dataset) or self._run(dataset)
        except FlagUnavailableError as error:
            return self.skip(str(error))

    def _unmet(self, dataset: Dataset) -> CheckResult | None:
        """Hook: a `SKIP` result if the dataset does not meet a need of the check, else `None`.

        Default: `SKIP` if a flag of `flags` is not available. The details list
        each such flag with the reason. A subclass with more needs overrides it
        and calls `super()._unmet(dataset)`.

        Raises:
            UnknownFlagError: The dataset does not accept a flag of `flags`.
        """
        self._check_accepted(dataset, self.flags)
        missing = {flag: dataset.availability(flag) for flag in self.flags}
        missing = {flag: a for flag, a in missing.items() if not self._counts_as_available(a)}
        if not missing:
            return None
        details = pd.DataFrame({'flag': list(missing), 'reason': [a.reason for a in missing.values()]})
        return self.skip(f'flags not available: {", ".join(map(str, missing))}.', details)

    @abstractmethod
    def _run(self, dataset: Dataset) -> CheckResult:
        """Hook: test the rule on a dataset that meets the needs of the check."""

    def result(self, status: CheckStatus, summary: str, details: pd.DataFrame | None = None) -> CheckResult:
        """Return a result of this check."""
        return CheckResult(self.name, status, summary, pd.DataFrame() if details is None else details)

    def skip(self, reason: str, details: pd.DataFrame | None = None) -> CheckResult:
        """Return a `SKIP` result with the reason."""
        return self.result(CheckStatus.SKIP, reason, details)

    def compared(self, comparison: Comparison, what: str) -> CheckResult:
        """Return `PASS` or `FAIL` for a comparison, `SKIP` when it compared nothing.

        The summary is `'<violations> of <compared> <what>.'`.

        Args:
            comparison: The comparison.
            what: The noun phrase of a violation, for example `'line MTUs off'`.
        """
        if not comparison.compared:
            return self.skip('nothing to compare.')
        violations = comparison.violations
        status = CheckStatus.FAIL if len(violations) else CheckStatus.PASS
        return self.result(status, f'{len(violations)} of {comparison.compared} {what}.', violations)

    @classmethod
    def has_flags(cls, dataset: Dataset, flags: Collection[FlagType]) -> bool:
        """Return `True` if every flag counts as available on the dataset (see the class docstring).

        Raises:
            UnknownFlagError: The dataset does not accept a flag.
        """
        cls._check_accepted(dataset, flags)
        return all(cls._counts_as_available(dataset.availability(flag)) for flag in flags)

    @staticmethod
    def _check_accepted(dataset: Dataset, flags: Collection[FlagType]) -> None:
        unknown = [flag for flag in flags if not dataset.flag_is_accepted(flag)]
        if unknown:
            raise UnknownFlagError(f'{dataset.name} does not accept the flags {", ".join(map(str, unknown))}.')

    @staticmethod
    def _counts_as_available(availability: Availability) -> bool:
        return availability.available is not False
