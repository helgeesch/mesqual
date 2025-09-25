"""MESCAL pandas utilities for energy systems data analysis.

This package provides specialized pandas utilities designed for energy systems analysis
workflows in the MESCAL framework. These utilities are optimized for working with
MultiIndex data structures that are common in multi-scenario energy modeling studies.

The utilities support MESCAL's core data flow patterns where energy data is typically
structured with multiple dimensions (scenarios, time periods, network components,
technologies) and requires specialized operations for filtering, transformation,
and aggregation.

Modules:
    prepend_model_prop_levels: Add model properties as MultiIndex levels to time-series data
    filter_by_model_query: Filter time-series using model metadata queries
    flatten_df: Convert MultiIndex DataFrames to flat format for visualization
    sort_multiindex: Sort MultiIndex levels with custom ordering
    xs_df: Enhanced cross-section interface for MultiIndex DataFrames
    merge_multi_index_levels: Combine multiple MultiIndex levels into single level
    add_index_as_column: Convert index levels to DataFrame columns

Examples:

    Basic usage for energy systems analysis:
    >>> from mescal.utils.pandas_utils import prepend_model_prop_levels, filter_by_model_query
    >>>
    >>> # Load generator model and time-series data
    >>> generators = study.scen.fetch('generators')  # Model metadata
    >>> generation = study.scen.fetch('generators_t.p')  # Time-series data
    >>>
    >>> # Add technology and zone properties to time-series
    >>> gen_with_props = prepend_model_prop_levels(
    ...     generation, generators, 'technology', 'zone'
    ... )
    >>>
    >>> # Filter for renewable generators only
    >>> renewable_gen = filter_by_model_query(
    ...     gen_with_props, generators, 'technology.isin(["solar", "wind"])'
    ... )
    >>>
    >>> # Aggregate by technology and zone
    >>> tech_zone_totals = renewable_gen.T.groupby(level=['technology', 'zone']).sum().T

Architecture Integration:
    These utilities integrate seamlessly with MESCAL's three-tier architecture:

    - **General utilities** (this package): Platform-agnostic data transformations
    - **Platform-specific**: Used by platform interfaces (mescal-pypsa, etc.)
    - **Study-specific**: Extended in individual studies for custom analysis

    They preserve MESCAL's MultiIndex data flow patterns while enabling flexible
    data manipulation and transformation for energy systems analysis workflows.

Performance Notes:
    - Operations maintain MultiIndex structures for memory efficiency
    - Bulk operations are preferred over iterative transformations
    - Query-based filtering minimizes data copying
    - Lazy evaluation patterns supported where possible
"""

from mescal.utils.pandas_utils.pend_props import prepend_model_prop_levels
from mescal.utils.pandas_utils.filter import filter_by_model_query
from mescal.utils.pandas_utils.flatten_df import flatten_df
from mescal.utils.pandas_utils.sort_multiindex import sort_multiindex
from mescal.utils.pandas_utils.xs_df import xs_df
from mescal.utils.pandas_utils.merge_multi_index_levels import merge_multi_index_levels
from mescal.utils.pandas_utils.add_index_as_column import add_index_as_column

__all__ = [
    'prepend_model_prop_levels',
    'filter_by_model_query',
    'flatten_df',
    'sort_multiindex',
    'xs_df',
    'merge_multi_index_levels',
    'add_index_as_column',
]
