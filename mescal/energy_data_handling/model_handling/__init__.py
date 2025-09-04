"""
Model handling utilities for MESCAL energy data processing.

This package provides tools for enriching energy system DataFrames with properties
from related model objects and creating combination columns for paired relationships.

Key Components:
- Membership property enrichers for adding related object properties to DataFrames
- Directional relationship handling for from/to column pairs in network data
- Membership pairs appenders for creating combination identifiers from paired columns

Common use cases:
- Enriching generator data with node properties (spatial analysis)
- Adding fuel/technology properties to generation assets
- Creating transmission line endpoints analysis with node characteristics
- Building combination identifiers for network connections
"""

from .membership_property_enrichers import (
    MembershipTagging,
    MembershipPropertyEnricher,
    DirectionalMembershipPropertyEnricher,
)

from .membership_pairs_appender import (
    BaseMembershipPairsAppender,
    StringMembershipPairsAppender,
    TupleMembershipPairsAppender,
)

__all__ = [
    # Enums
    'MembershipTagging',
    
    # Property enrichers
    'MembershipPropertyEnricher',
    'DirectionalMembershipPropertyEnricher',
    
    # Pairs appenders
    'BaseMembershipPairsAppender',
    'StringMembershipPairsAppender',
    'TupleMembershipPairsAppender',
]