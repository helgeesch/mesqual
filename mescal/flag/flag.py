"""
Flag type protocol for MESCAL.

Defines the interface requirements for flag types: must be hashable (for use as
dictionary keys) and convertible to strings (for logging and serialization).
"""

from typing import Protocol


class FlagTypeProtocol(Protocol):
    """
    Protocol for MESCAL flag types.
    
    Flags must be hashable (usable as dict keys) and convertible to strings.
    Compatible types include strings, enums, and custom classes with __hash__ and __str__.
    
    Examples:
        - "Generator.Model", "Node.p_nom_opt" 
        - Enum members
        - Custom classes implementing the protocol
    """
    
    def __hash__(self) -> int:
        """Return stable hash value for use as dictionary key."""
        ...

    def __str__(self) -> str:
        """Return human-readable string representation."""
        ...
