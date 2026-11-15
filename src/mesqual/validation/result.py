"""The status and the result of a check."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class CheckStatus(str, Enum):
    """Status of a check.

    Attributes:
        PASS: The rule holds.
        FAIL: The rule does not hold. The details list the violations.
        INFO: A finding to read, not a failure.
        SKIP: The check did not run, for example because a flag is not available.
        ERROR: The check raised an exception. The summary gives its type and text.
    """

    PASS = 'PASS'
    FAIL = 'FAIL'
    INFO = 'INFO'
    SKIP = 'SKIP'
    ERROR = 'ERROR'

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class CheckResult:
    """The result of one check on one dataset.

    Attributes:
        check: Name of the check.
        status: The status.
        summary: One line on the result.
        details: One row per violation or finding; empty if there is none.
            Equality of two results does not compare the details.
    """

    check: str
    status: CheckStatus
    summary: str
    details: pd.DataFrame = field(default_factory=pd.DataFrame, compare=False)
