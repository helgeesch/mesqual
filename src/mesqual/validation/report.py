"""The validation report: the results of all checks on one dataset, as a frame or as Markdown."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mesqual.validation.result import CheckResult, CheckStatus


@dataclass(frozen=True)
class ValidationReport:
    """The results of the checks of one validation run, in run order.

    Attributes:
        dataset: Name of the dataset.
        results: One result per check.
    """

    dataset: str
    results: Sequence[CheckResult]

    def __getitem__(self, check: str) -> CheckResult:
        """Return the result of the check with this name.

        Raises:
            KeyError: No result has the name.
        """
        for result in self.results:
            if result.check == check:
                return result
        raise KeyError(f'No check {check} in the report.')

    @property
    def failed(self) -> tuple[CheckResult, ...]:
        """The results with status `FAIL`."""
        return self._with(CheckStatus.FAIL)

    @property
    def errors(self) -> tuple[CheckResult, ...]:
        """The results with status `ERROR`: the checks that raised an exception."""
        return self._with(CheckStatus.ERROR)

    @property
    def passed(self) -> bool:
        """`True` if no check failed or raised."""
        return not self.failed and not self.errors

    def _with(self, status: CheckStatus) -> tuple[CheckResult, ...]:
        return tuple(result for result in self.results if result.status is status)

    def to_frame(self) -> pd.DataFrame:
        """Return one row per check: `check`, `status`, `summary` and the number of detail rows `details`."""
        rows = [(r.check, str(r.status), r.summary, len(r.details)) for r in self.results]
        return pd.DataFrame(rows, columns=['check', 'status', 'summary', 'details'])

    def to_markdown(self, path: Path | str | None = None, max_rows: int = 20) -> str:
        """Return the report as Markdown and write it to `path` if given.

        The report has one table with the status and summary of every check,
        then one section per check with details: the first `max_rows` detail
        rows and the number of rows left out. A details index other than the
        default range index becomes columns.

        Args:
            path: File to write, or `None`.
            max_rows: Detail rows per check.
        """
        lines = [f'# Validation of {self.dataset}', '', *self._table(self.to_frame().drop(columns='details'))]
        for result in self.results:
            if len(result.details):
                lines += ['', f'## {result.check}', '', f'{result.status}: {result.summary}', '']
                lines += self._table(self._flat(result.details).head(max_rows))
                if len(result.details) > max_rows:
                    lines += ['', f'{len(result.details) - max_rows} more rows.']
        text = '\n'.join(lines) + '\n'
        if path is not None:
            Path(path).write_text(text, encoding='utf-8')
        return text

    @staticmethod
    def _flat(details: pd.DataFrame) -> pd.DataFrame:
        if isinstance(details.index, pd.RangeIndex) and details.index.name is None:
            return details
        return details.reset_index()

    @classmethod
    def _table(cls, frame: pd.DataFrame) -> list[str]:
        header = [cls._cell(column) for column in frame.columns]
        cells = frame.map(cls._cell)
        return [
            f'| {" | ".join(header)} |',
            f'|{"|".join(" --- " for _ in header)}|',
            *(f'| {" | ".join(row)} |' for row in cells.itertuples(index=False)),
        ]

    @staticmethod
    def _cell(value) -> str:
        if pd.api.types.is_scalar(value) and pd.isna(value):
            return ''
        return str(value).replace('|', '\\|').replace('\n', ' ')
