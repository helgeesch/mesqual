"""Properties of related model objects as new columns.

- `MembershipPropertyEnricher` adds the properties of the objects that a
  column refers to. For example, the generator model gets `voltage` and
  `country` from its `node` column, or `node_voltage` and `node_country`
  with `MembershipTagging.PREFIX`.
- `DirectionalMembershipPropertyEnricher` does the same for from/to column
  pairs. For example, the line model gets `country_from` and `country_to`.
- `StringMembershipPairsAppender` and `TupleMembershipPairsAppender` add combination columns of
  from/to pairs, for example `'N1 - N2'`.
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
