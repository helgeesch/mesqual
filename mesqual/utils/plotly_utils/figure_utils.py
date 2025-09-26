"""Utility functions for styling and modifying Plotly figures.

This module provides convenience functions for common figure modifications
such as title formatting, annotation styling, axis configuration, and
adding interactive controls to Plotly figures.
"""

import plotly.graph_objects as go


def set_title(fig: go.Figure, title: str):
    """Set a centered, bold title for the figure.

    Args:
        fig: Plotly figure object to modify.
        title: Title text to display.

    Example:

        >>> set_title(fig, "Sales Performance Dashboard")
    """
    title = f'<b>{title}</b>'
    fig.update_layout(title_text=title, title_x=0.5)


def remove_category_in_annotations(fig: go.Figure):
    """Remove category names from subplot annotations, keeping only values.

    Modifies annotation text to show only the part after '=' for cleaner
    subplot labels (e.g., 'sex=Male' becomes 'Male').

    Args:
        fig: Plotly figure object with annotations to modify.

    Example:

        >>> remove_category_in_annotations(fig)  # 'smoker=Yes' → 'Yes'
    """
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))


def make_annotations_bold(fig: go.Figure):
    """Apply bold formatting to all figure annotations.

    Args:
        fig: Plotly figure object with annotations to modify.

    Example:

        >>> make_annotations_bold(fig)  # Makes all subplot labels bold
    """
    fig.for_each_annotation(lambda a: a.update(text='<b>' + a.text + '</b>'))


def unmatch_xaxes(fig: go.Figure):
    """Remove x-axis matching across subplots.

    Allows each subplot to have independent x-axis ranges and scaling.

    Args:
        fig: Plotly figure object to modify.

    Example:

        >>> unmatch_xaxes(fig)  # Each subplot can have different x-ranges
    """
    fig.update_xaxes(matches=None)


def unmatch_yaxes(fig: go.Figure):
    """Remove y-axis matching across subplots.

    Allows each subplot to have independent y-axis ranges and scaling.

    Args:
        fig: Plotly figure object to modify.

    Example:

        >>> unmatch_yaxes(fig)  # Each subplot can have different y-ranges
    """
    fig.update_yaxes(matches=None)


def reverse_legend_traceorder(fig: go.Figure):
    """Reverse the order of legend entries.

    Args:
        fig: Plotly figure object to modify.

    Example:

        >>> reverse_legend_traceorder(fig)  # Last trace appears first in legend
    """
    fig.update_layout(legend_traceorder="reversed")


def add_datetime_rangeslider(fig: go.Figure):
    """Add an interactive datetime range slider and selector to the figure.

    Adds a range slider below the plot and time period selector buttons
    for easy navigation of time series data.

    Args:
        fig: Plotly figure object to modify.

    Returns:
        Modified figure object with range controls.

    Example:

        >>> fig = add_datetime_rangeslider(fig)
        >>> fig.show()  # Now includes 1d, 1w, 1m, 6m, YTD, 1y buttons
    """
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.update_layout(yaxis_fixedrange=False)
    return fig
