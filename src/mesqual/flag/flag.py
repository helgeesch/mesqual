from typing import Protocol


class FlagTypeProtocol(Protocol):
    """
    Protocol for mesqual flag types.

    Flags must be hashable (usable as dict keys) and convertible to strings.
    Compatible types include strings, enums, and custom classes with `__hash__` and `__str__`.

    Examples of flag types:

    - Strings, for example `'Generator.Model'` or `'Node.Results.price'`.
    - Enum members.
    - Custom classes that implement the protocol.
    """

    def __hash__(self) -> int:
        """Return stable hash value for use as dictionary key."""
        ...

    def __str__(self) -> str:
        """Return human-readable string representation."""
        ...
