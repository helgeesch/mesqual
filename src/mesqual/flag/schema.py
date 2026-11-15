"""Frame schemas: the index and column levels of a flag result, with their roles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Level:
    """One index or column level of a result.

    Attributes:
        name: Name of the level. `None`: the level is unnamed, as in pandas.
        role: Role of the level:

            - `time`: the time level.
            - `object`: the ids (names or numbers) of the objects of the flag.
            - `counterpart`: a second object of the same or another type that
              each value relates to, for example the partner area of a trade.
            - `object_from`, `object_to`: the two ends of an edge.
            - `dimension`: any other categorical level.
            - `dataset`: added by collections.
    """

    ROLES: ClassVar[frozenset[str]] = frozenset(
        {'time', 'object', 'counterpart', 'object_from', 'object_to', 'dimension', 'dataset'}
    )

    name: str | None
    role: str

    def __post_init__(self):
        if self.role not in self.ROLES:
            raise ValueError(f'Level {self.name!r}: unknown role {self.role!r}. Roles: {sorted(self.ROLES)}.')

    def __str__(self) -> str:
        return f'{self.name}: {self.role}'


@dataclass(frozen=True)
class FrameSchema:
    """Index and column levels of a flag result.

    The number and order of the levels is fixed per spec.

    Attributes:
        index: Levels of the row index.
        columns: Levels of the columns. Empty for a Series. A frame has at
            least one column level, also when its column names are free,
            for example `Level(None, 'dimension')` of a model table.
        dtype: Optional dtype of the values.

    Examples:
        >>> SPREAD_SCHEMA = FrameSchema(
        ...     index=(Level('time', 'time'),),
        ...     columns=(Level('Country', 'object'), Level('partner_country', 'counterpart')),
        ... )
    """

    index: tuple[Level, ...]
    columns: tuple[Level, ...] = ()
    dtype: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'index', tuple(self.index))
        object.__setattr__(self, 'columns', tuple(self.columns))
        for level in self.index + self.columns:
            if not isinstance(level, Level):
                raise TypeError(f'Schema levels must be of type Level, got {level!r}.')

    @property
    def time_levels(self) -> tuple[Level, ...]:
        """The index levels with role `time`."""
        return tuple(level for level in self.index if level.role == 'time')

    def __str__(self) -> str:
        text = f'index ({", ".join(map(str, self.index))})'
        if self.columns:
            text += f', columns ({", ".join(map(str, self.columns))})'
        return text
