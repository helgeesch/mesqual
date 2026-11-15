"""Source inventory: what a platform dataset finds in its source, read with cheap reads only."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceInventory:
    """What the source of a platform dataset holds, read with cheap reads only.

    A platform dataset builds its inventory once (`PlatformDataset.build_source_inventory`),
    for example from the file list and small tables of the source. mesqual
    reads no field of the inventory. A platform subclass adds the fields
    that its `_is_available` implementations and its default attributes
    need.

    Attributes:
        fingerprint: Changes when the source changes, for example a hash of
            the relative name and the `FileRead.key` of each source file.

    Examples:
        >>> from dataclasses import dataclass
        >>> @dataclass(frozen=True)
        ... class MyInventory(SourceInventory):
        ...     files: frozenset[str] = frozenset()
        >>> MyInventory(fingerprint='3f2a', files=frozenset({'prices.csv'}))
        MyInventory(fingerprint='3f2a', files=frozenset({'prices.csv'}))
    """

    fingerprint: str
