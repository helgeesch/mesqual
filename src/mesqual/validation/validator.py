"""The validator that runs a set of checks on one dataset and returns the report."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from mesqual.datasets import Dataset
from mesqual.validation.check import DatasetCheck
from mesqual.validation.report import ValidationReport
from mesqual.validation.result import CheckResult, CheckStatus

logger = logging.getLogger(__name__)


class DatasetValidator:
    """Holds checks by name and runs them on a dataset.

    Check names are unique. A platform package subclasses the validator to
    give a default set of checks or extra constructor inputs. It passes the
    default checks to `super().__init__(checks)`. A subclass that binds a
    dataset at construction overrides `run(self, dataset=None)` and falls back
    to the bound dataset: `return super().run(self.dataset if dataset is None else dataset)`.

    Examples:
        >>> from mesqual.validation import ConstraintCheck
        >>> validator = DatasetValidator([
        ...     ConstraintCheck('Line.Constraints.capacity_up', min_value=0),
        ...     ConstraintCheck('Generator.Results.generation', min_value=0, isna_ok=False),
        ... ])
        >>> validator.remove('ConstraintCheck[Line.Constraints.capacity_up]')
        >>> report = validator.run(dataset)
        >>> report.passed
        True

    Args:
        checks: The checks, in run order.

    Raises:
        TypeError: A check is not a `DatasetCheck`.
        ValueError: Two checks have the same name.
    """

    def __init__(self, checks: Iterable[DatasetCheck] = ()):
        self._checks: dict[str, DatasetCheck] = {}
        for check in checks:
            self.add(check)

    @property
    def checks(self) -> tuple[DatasetCheck, ...]:
        """The checks, in run order."""
        return tuple(self._checks.values())

    def add(self, check: DatasetCheck, *, replace: bool = False) -> None:
        """Add a check at the end.

        Args:
            check: The check.
            replace: Replace a check with the same name in place. A check with
                a new name is added at the end.

        Raises:
            TypeError: The check is not a `DatasetCheck`.
            ValueError: The validator has a check with the same name and `replace` is `False`.
        """
        if not isinstance(check, DatasetCheck):
            raise TypeError(f'A validator runs DatasetCheck instances, not {type(check).__name__}.')
        if check.name in self._checks and not replace:
            raise ValueError(f'The validator has a check {check.name}; add it with replace=True to replace it.')
        self._checks[check.name] = check

    def remove(self, name: str) -> None:
        """Remove the check with this name.

        Raises:
            KeyError: No check has the name.
        """
        if name not in self._checks:
            raise KeyError(f'No check {name} in the validator.')
        del self._checks[name]

    def run(self, dataset: Dataset) -> ValidationReport:
        """Run every check on the dataset and return the report.

        A check that raises an exception gets the status `ERROR` with
        `"Type: text"` of the exception as summary, and the other checks still
        run. The traceback goes to the log at level `WARNING`.
        """
        return ValidationReport(dataset.name, tuple(self._run_check(check, dataset) for check in self.checks))

    @staticmethod
    def _run_check(check: DatasetCheck, dataset: Dataset) -> CheckResult:
        try:
            return check.run(dataset)
        except Exception as error:
            logger.warning('Check %s raised on dataset %s.', check.name, dataset.name, exc_info=True)
            return check.result(CheckStatus.ERROR, f'{type(error).__name__}: {error}')
