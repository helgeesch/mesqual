"""Plotly utilities for enhanced figure styling and subplot manipulation.

This package provides utilities for working with Plotly figures in the MESCAL framework,
including:

- **plotly_theme**: Custom themes and color palettes (template) for consistent visualization styling
- **figure_utils**: Common figure modifications like titles, annotations, and axis controls
- **px_category_utils**: Tools for working with categorical data in faceted Plotly Express plots

The utilities enable precise control over figure appearance and facilitate adding
elements to specific subplots in complex multi-panel visualizations.

Example:

    >>> from mescal.utils.plotly_utils import PlotlyTheme, figure_utils
    >>> from mescal.utils.plotly_utils.plotly_theme import colors
    >>>
    >>> # Apply consistent theming
    >>> theme = PlotlyTheme(default_colorway=colors.qualitative.default)
    >>> theme.apply()
    >>>
    >>> # Style figures
    >>> figure_utils.set_title(fig, "Energy Analysis Dashboard")
    >>> figure_utils.make_annotations_bold(fig)
"""

from .plotly_theme import PlotlyTheme
from . import figure_utils
from . import px_category_utils

__all__ = [
    'PlotlyTheme',
    'figure_utils',
    'px_category_utils',
]
