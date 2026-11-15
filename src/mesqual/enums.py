from __future__ import annotations

from enum import Enum


class ItemTypeEnum(Enum):
    """Kind of result of a flag. It follows from the class of the flag spec."""

    Model = 'Model'
    TimeSeries = 'TimeSeries'
    Other = 'Other'


class VisualizationTypeEnum(Enum):
    """How a map shows the objects of a flag."""

    Area = 'Area'
    Point = 'Point'
    Line = 'Line'
    Border = 'Border'
    Other = 'Other'


class TopologyTypeEnum(Enum):
    """Role of the objects of a flag in the network topology.

    Nodes and node-connected elements aggregate to areas, edges to borders.
    A border joins two areas; the two areas need not be geographic neighbours.
    """

    Area = 'Area'
    Border = 'Border'
    Node = 'Node'
    NodeConnectedElement = 'NodeConnectedElement'
    Edge = 'Edge'
    Other = 'Other'

    @property
    def visualization_type(self) -> VisualizationTypeEnum:
        """Default visualization type of objects with this topology type."""
        match self:
            case TopologyTypeEnum.Node | TopologyTypeEnum.NodeConnectedElement:
                return VisualizationTypeEnum.Point
            case TopologyTypeEnum.Edge:
                return VisualizationTypeEnum.Line
            case TopologyTypeEnum.Area:
                return VisualizationTypeEnum.Area
            case TopologyTypeEnum.Border:
                return VisualizationTypeEnum.Border
        return VisualizationTypeEnum.Other


class QuantityTypeEnum(Enum):
    """
    Physical property that determines how a quantity behaves under time aggregation.

    For time series resampling, the type decides what happens when the time
    granularity changes:

    - `INTENSIVE`: quantities per unit, such as rates. A finer granularity
      (hourly to 15 min) repeats the value; a coarser one (hourly to daily)
      takes the mean. Examples: prices [€/MWh], power [MW], flows [MW].
    - `EXTENSIVE`: totals or amounts. A finer granularity splits the value; a
      coarser one takes the sum. Examples: welfare [€], volume [MWh],
      energy [MWh].
    - `AMBIGUOUS`: the unit alone does not decide the type. No unit in `Units`
      is ambiguous; a platform can mark its own units so. The flag spec states
      the type.
    - `UNKNOWN`: no type is known, for example for a value without unit.
    """

    INTENSIVE = 'intensive'
    EXTENSIVE = 'extensive'
    AMBIGUOUS = 'ambiguous'
    UNKNOWN = 'unknown'


class ComparisonTypeEnum(Enum):
    """Various Comparison Types for DatasetComparison class' fetch method."""

    VARIATION = 'variation'  # Show only variation's model_df
    BOTH = 'both'  # Show both variation and reference side by side
    DELTA = 'delta'  # Show delta for numeric values, variation for non-numeric
