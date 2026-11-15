"""Checks on datasets that return results, and a validator that collects them in a report."""

from mesqual.validation.check import DatasetCheck
from mesqual.validation.comparison import Comparison
from mesqual.validation.constraint_check import ConstraintCheck
from mesqual.validation.report import ValidationReport
from mesqual.validation.result import CheckResult, CheckStatus
from mesqual.validation.validator import DatasetValidator

__all__ = [
    'CheckResult',
    'CheckStatus',
    'Comparison',
    'ConstraintCheck',
    'DatasetCheck',
    'DatasetValidator',
    'ValidationReport',
]
