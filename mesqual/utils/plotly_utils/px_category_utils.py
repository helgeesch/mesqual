"""Utilities for working with categorical data in Plotly Express subplots and faceted figures.

This module provides functions to navigate and manipulate subplot structures created
by Plotly Express, particularly when dealing with faceted plots and categorical data.
It enables precise positioning of additional elements on specific subplot axes.
"""

import plotly.graph_objects as go
import plotly.express as px


def get_x_y_axis_for_category(fig: go.Figure, category_args: dict[str, str]) -> tuple[str, str]:
    """Find the x and y axis names for a subplot matching specific category values.

    Searches through figure traces to find one whose hovertemplate contains all
    specified category key-value pairs, then returns the corresponding axis names.

    Args:
        fig: Plotly figure object containing subplot traces.
        category_args: Dictionary mapping category names to their values (e.g.,
            {'sex': 'Male', 'smoker': 'Yes'}).

    Returns:
        Tuple containing the x-axis and y-axis names (e.g., ('x', 'y') or
        ('x2', 'y3')).

    Raises:
        KeyError: If no trace contains all the specified category key-value pairs
            in its hovertemplate.

    Example:

        >>> category_args = {'sex': 'Male', 'smoker': 'Yes'}
        >>> x_axis, y_axis = get_x_y_axis_for_category(fig, category_args)
        >>> print(f"Found axes: {x_axis}, {y_axis}")
            Found axes: x2, y3
    """
    keys = [f'{k}={i}' for k, i in category_args.items()]
    for trace in fig.data:
        if all(k in trace.hovertemplate for k in keys):
            x_axis = trace.xaxis if 'xaxis' in trace else 'x'
            y_axis = trace.yaxis if 'yaxis' in trace else 'y'
            return x_axis, y_axis
    raise KeyError(f'No trace with matching key: value pairs {keys} found in any hovertemplate.')


def get_all_x_axis_names(fig: go.Figure) -> list[str]:
    """Get all x-axis names from a Plotly figure layout.

    Args:
        fig: Plotly figure object.

    Returns:
        List of x-axis attribute names found in the figure layout (e.g.,
        ['xaxis', 'xaxis2', 'xaxis3']).
    """
    return [attr for attr in fig.layout if attr.startswith('xaxis')]


def get_all_y_axis_names(fig: go.Figure) -> list[str]:
    """Get all y-axis names from a Plotly figure layout.

    Args:
        fig: Plotly figure object.

    Returns:
        List of y-axis attribute names found in the figure layout (e.g.,
        ['yaxis', 'yaxis2', 'yaxis3']).
    """
    return [attr for attr in fig.layout if attr.startswith('yaxis')]


def get_row_col_for_x_y_axis(fig: go.Figure, x_axis: str, y_axis: str) -> tuple[int, int]:
    """Convert axis names to subplot row and column indices.

    Determines the subplot grid position by analyzing axis domains within the
    figure layout. This is useful for adding elements to specific subplots.

    Args:
        fig: Plotly figure object containing subplots.
        x_axis: X-axis name (e.g., 'x', 'x2', 'xaxis', 'xaxis2').
        y_axis: Y-axis name (e.g., 'y', 'y2', 'yaxis', 'yaxis2').

    Returns:
        Tuple of (row, col) indices for the subplot (1-indexed).

    Raises:
        KeyError: If the specified axis names are not found in the figure layout.

    Example:

        >>> row, col = get_row_col_for_x_y_axis(fig, 'x2', 'y3')
        >>> print(f"Subplot at row {row}, column {col}")
            Subplot at row 2, column 3
    """
    all_x_axis_names = get_all_x_axis_names(fig)
    all_y_axis_names = get_all_y_axis_names(fig)

    x_domains = list(sorted(set([tuple(fig.layout[x].domain) for x in all_x_axis_names])))
    y_domains = list(sorted(set([tuple(fig.layout[y].domain) for y in all_y_axis_names])))

    if 'axis' not in x_axis:
        x_axis = f'xaxis{x_axis[1:]}'
    if 'axis' not in y_axis:
        y_axis = f'yaxis{y_axis[1:]}'

    if x_axis not in all_x_axis_names:
        raise KeyError(f'No matching axis found for {x_axis}.')
    if y_axis not in all_y_axis_names:
        raise KeyError(f'No matching axis found for {y_axis}.')

    col = x_domains.index(tuple(fig.layout[x_axis].domain)) + 1
    row = y_domains.index(tuple(fig.layout[y_axis].domain)) + 1
    return row, col


def get_subplot_row_and_col_for_category(fig: go.Figure, category_args: dict[str, str]) -> tuple[int, int]:
    """Get subplot row and column for a specific category combination.

    Combines axis lookup and position conversion to directly find the subplot
    location for given category values. This is the main convenience function
    for adding elements to category-specific subplots.

    Args:
        fig: Plotly figure object containing faceted subplots.
        category_args: Dictionary mapping category names to their values.

    Returns:
        Tuple of (row, col) indices for the subplot (1-indexed).

    Raises:
        KeyError: If no subplot matches the specified category combination or
            if axis names cannot be found.

    Example:

        >>> category_args = {'sex': 'Female', 'smoker': 'No'}
        >>> row, col = get_subplot_row_and_col_for_category(fig, category_args)
        >>> fig.add_trace(trace, row=row, col=col)
    """
    x_axis, y_axis = get_x_y_axis_for_category(fig, category_args)
    row, col = get_row_col_for_x_y_axis(fig, x_axis, y_axis)
    return row, col


def get_index_for_category_on_axis(fig: go.Figure, axis: str, category_value: str) -> int:
    """Get the numerical index of a categorical value on a specific axis.

    Converts a categorical string value to its corresponding numerical position
    on the specified axis. Useful for precise positioning of annotations or
    additional traces on categorical axes.

    Args:
        fig: Plotly figure object.
        axis: Axis name (e.g., 'x', 'x2', 'y', 'y2', 'xaxis', 'yaxis2').
        category_value: String value of the category to find.

    Returns:
        1-indexed numerical position of the category on the axis.

    Raises:
        TypeError: If category_value is not a string.
        KeyError: If the axis name is invalid or the category value is not
            found in the axis categoryarray.

    Example:

        >>> index = get_index_for_category_on_axis(fig, 'x', 'Dinner')
        >>> print(f"'Dinner' is at position {index}")
            'Dinner' is at position 2
    """
    if not isinstance(category_value, str):
        raise TypeError(
            'Method only works with string categories. '
            'Sure you need this? In case you already have an int / float, just use the value as an index directly.'
        )
    if 'axis' not in axis:
        if axis.startswith('x'):
            axis = f'xaxis{axis[1:]}'
        elif axis.startswith('y'):
            axis = f'yaxis{axis[1:]}'
        else:
            raise KeyError(f'Unknown axis {axis}')
    cat_array = fig.layout[axis].categoryarray
    if category_value not in cat_array:
        raise KeyError(f'Unknown category {category_value}. Recognised categories for axis: {cat_array}')
    index = cat_array.index(category_value) + 1
    return index


if __name__ == '__main__':

    data = px.data.tips().groupby(['sex', 'time', 'day', 'smoker'])['tip'].mean().to_frame().reset_index()
    figg = px.bar(
        data,
        x="time",
        y="tip",
        color="day",
        facet_col="sex",
        facet_row="smoker",
        barmode='group',
        title="Sample Bar Chart with Facets"
    )
    data_mean = px.data.tips().groupby(['sex', 'smoker'])['tip'].mean()
    # for (sex, smoker), value in data_mean.items():
    #     row, col = get_subplot_row_and_col_for_category(fig, category_args={'sex': sex, 'smoker': smoker})
    #     figg.add_trace(
    #         go.Scatter(
    #             x=[1, 2],
    #             y=[value, value],
    #             mode='lines',
    #             line=dict(color='black'),
    #             hovertext=f'mean for sex={sex} and smoker={smoker}'
    #         ),
    #         row=row,
    #         col=col
    #     )

    data_mean = px.data.tips().groupby(['sex', 'smoker', 'time'])['tip'].mean()
    # for (sex, smoker, time), value in data_mean.items():
    #     xax, yax = get_x_y_axis_for_category(fig, category_args={'sex': sex, 'smoker': smoker})
    #     x_index = get_index_for_category_on_axis(fig, xax, time)
    #     figg.add_trace(
    #         go.Scatter(
    #             x=[x_index-0.4, x_index+0.4],
    #             y=[value, value],
    #             mode='lines',
    #             line=dict(color='blue'),
    #             xaxis=xax,
    #             yaxis=yax,
    #             hovertext=f'mean for sex={sex} and smoker={smoker} and time={time}'
    #         )
    #     )

    figg.write_html('tmp/tmp.html')