"""Maps, dashboards and tables for multi-scenario results.

- `folium_viz_system` (alias `folviz`): build folium maps from model and
  result data, with legends.
- `value_mapping_system` (alias `valmap`): map data values to colors, widths,
  opacities and other visual properties.
- `TimeSeriesDashboardGenerator`: heatmap dashboards of time series with KPI
  statistics, faceted by scenario, object or aggregation.
- `HTMLDashboard`: combine figures, maps, tables and section dividers into
  one HTML file.
- `HTMLTable`: interactive HTML tables from DataFrames.
"""

from . import folium_viz_system
from . import folium_viz_system as folviz
from . import value_mapping_system
from . import value_mapping_system as valmap
from .plotly_figures.timeseries_dashboard import TimeSeriesDashboardGenerator
from .html_dashboard import HTMLDashboard
from .html_table import HTMLTable

__all__ = [
    'value_mapping_system',
    'valmap',
    'folium_viz_system',
    'folviz',
    'TimeSeriesDashboardGenerator',
    'HTMLDashboard',
    'HTMLTable',
]
