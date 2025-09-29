"""MESQUAL Visualizations Package.

This package provides comprehensive visualization capabilities for energy systems analysis,
including interactive maps, time series dashboards, and data export functionality.

The visualizations package supports multiple output formats and interactive components
designed specifically for multi-scenario energy modeling analysis and comparison.

Modules:
    - folium_legend_system: Legend creation and management for Folium maps
    - folium_viz_system: Interactive map visualization system using Folium
    - value_mapping_system: Data value mapping and color scaling utilities

Classes:
    - TimeSeriesDashboardGenerator: Creates interactive Plotly time series dashboards
    - HTMLDashboard: Generates comprehensive HTML analysis dashboards
    - HTMLTable: Creates formatted HTML tables for data presentation

Example:

    >>> from mesqual.visualizations import HTMLDashboard, TimeSeriesDashboardGenerator
    >>> dashboard = HTMLDashboard()
    >>> ts_gen = TimeSeriesDashboardGenerator()
"""

from . import folium_legend_system
from . import folium_viz_system
from . import value_mapping_system
from .plotly_figures.timeseries_dashboard import TimeSeriesDashboardGenerator
from .html_dashboard import HTMLDashboard
from .html_table import HTMLTable

__all__ = [
    'folium_legend_system',
    'folium_viz_system',
    'value_mapping_system',
    'TimeSeriesDashboardGenerator',
    'HTMLDashboard',
    'HTMLTable',
]
