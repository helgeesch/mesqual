"""
Flag management system for MESCAL energy analysis.

Provides flexible identification and metadata management for energy system variables
and models. Flags serve as universal identifiers with associated units, visualization
preferences, and model relationships.

Key features:
- Type-safe flag protocols
- Registry-based metadata management  
- Hierarchical model relationships
- Energy-specific enum integration
"""

from .flag import FlagTypeProtocol
from .flag_index import FlagIndex, EmptyFlagIndex
