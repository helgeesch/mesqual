"""Folium maps from model and KPI data.

A generator (`AreaGenerator`, `LineGenerator`, `ArrowIconGenerator`,
`CircleMarkerGenerator`, `TextOverlayGenerator`, `LineTextOverlayGenerator`)
draws one map element per data item. Its feature resolver computes the
visual properties with `PropertyMapper` objects from the model attributes or
the KPI value. `KPICollectionMapVisualizer` and `CustomKPIGroupGenerator`
build feature groups from KPI collections. `legends` holds the map legends.
"""

from .base_viz_system import PropertyMapper, ResolvedFeature, FeatureResolver, FoliumObjectGenerator
from .kpi_collection_map_visualizer import KPICollectionMapVisualizer, KPIGroupingManager
from .custom_kpi_group_generator import CustomKPIGroupGenerator
from .visualizable_data_item import VisualizableDataItem, ModelDataItem, KPIDataItem
from .viz_areas import ResolvedAreaFeature, AreaFeatureResolver, AreaGenerator
from .viz_arrow_icon import ResolvedArrowIconFeature, ArrowIconFeatureResolver, ArrowIconGenerator
from .viz_circle_marker import ResolvedCircleMarkerFeature, CircleMarkerFeatureResolver, CircleMarkerGenerator

# from .viz_icons import ResolvedIconFeature, IconFeatureResolver, IconGenerator
from .viz_line_text_overlay import (
    ResolvedLineTextOverlayFeature,
    LineTextOverlayFeatureResolver,
    LineTextOverlayGenerator,
)
from .viz_lines import ResolvedLineFeature, LineFeatureResolver, LineGenerator
from .viz_text_overlay import ResolvedTextOverlayFeature, TextOverlayFeatureResolver, TextOverlayGenerator

from . import legends

__all__ = [
    # Base visualization components
    'PropertyMapper',
    'ResolvedFeature',
    'FeatureResolver',
    'FoliumObjectGenerator',
    # KPI visualization and management
    'KPICollectionMapVisualizer',
    'KPIGroupingManager',
    'CustomKPIGroupGenerator',
    # Data item abstractions
    'VisualizableDataItem',
    'ModelDataItem',
    'KPIDataItem',
    # Area visualization
    'ResolvedAreaFeature',
    'AreaFeatureResolver',
    'AreaGenerator',
    # Arrow icon visualization
    'ResolvedArrowIconFeature',
    'ArrowIconFeatureResolver',
    'ArrowIconGenerator',
    # Circle marker visualization
    'ResolvedCircleMarkerFeature',
    'CircleMarkerFeatureResolver',
    'CircleMarkerGenerator',
    # Line text overlay visualization
    'ResolvedLineTextOverlayFeature',
    'LineTextOverlayFeatureResolver',
    'LineTextOverlayGenerator',
    # Line visualization
    'ResolvedLineFeature',
    'LineFeatureResolver',
    'LineGenerator',
    # Text overlay visualization
    'ResolvedTextOverlayFeature',
    'TextOverlayFeatureResolver',
    'TextOverlayGenerator',
    'legends',
]
