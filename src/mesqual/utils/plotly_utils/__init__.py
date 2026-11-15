"""Plotly helpers: a theme, figure styling and subplot lookup.

- `plotly_theme`: `PlotlyTheme` and the color palettes `colors`.
- `figure_utils`: titles, annotations, axes and a datetime range slider.
- `px_category_utils`: the subplot of a category combination in a faceted
  Plotly Express figure.
"""

from .plotly_theme import PlotlyTheme
from . import figure_utils
from . import px_category_utils

__all__ = [
    'PlotlyTheme',
    'figure_utils',
    'px_category_utils',
]
