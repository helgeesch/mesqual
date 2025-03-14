from typing import Protocol


class FlagTypeProtocol(Protocol):
    """A FlagType can be anything that is Hashable as well as Stringable."""
    def __hash__(self) -> int:
        ...  # Placeholder: just indicates that __hash__ must exist

    def __str__(self) -> str:
        ...  # Placeholder: just indicates that __str__ must exist
