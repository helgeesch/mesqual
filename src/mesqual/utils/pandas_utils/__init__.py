"""pandas helpers for the MultiIndex frames that datasets return.

A fetched frame has objects, scenarios or properties as index or column
levels. These helpers add model properties as levels, filter by model
queries, select and reorder levels and flatten frames for plotting.

Functions:

- `prepend_model_prop_levels`: add model properties as index levels.
- `filter_by_model_query`: keep the objects that match a query on the model.
- `flatten_df`: convert a frame to long format, one value per row.
- `sort_multiindex`: sort one index level in a custom order.
- `xs_df`: select one or more keys from an index level.
- `merge_multi_index_levels`: combine index levels into one level.
- `add_index_as_column`: add the index levels as columns.

Examples:
    >>> from mesqual.utils.pandas_utils import prepend_model_prop_levels
    >>> generation = dataset.fetch('Generator.Results.generation')
    >>> generators = dataset.fetch('Generator.Model')
    >>> by_fuel = prepend_model_prop_levels(generation, generators, 'fuel')
    >>> by_fuel.T.groupby(level='fuel').sum().T.columns.tolist()
    ['coal', 'gas', 'nuclear', 'solar', 'wind']
"""

from mesqual.utils.pandas_utils.pend_props import prepend_model_prop_levels
from mesqual.utils.pandas_utils.filter import filter_by_model_query
from mesqual.utils.pandas_utils.flatten_df import flatten_df
from mesqual.utils.pandas_utils.sort_multiindex import sort_multiindex
from mesqual.utils.pandas_utils.xs_df import xs_df
from mesqual.utils.pandas_utils.merge_multi_index_levels import merge_multi_index_levels
from mesqual.utils.pandas_utils.add_index_as_column import add_index_as_column

__all__ = [
    'prepend_model_prop_levels',
    'filter_by_model_query',
    'flatten_df',
    'sort_multiindex',
    'xs_df',
    'merge_multi_index_levels',
    'add_index_as_column',
]
