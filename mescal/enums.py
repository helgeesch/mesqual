from __future__ import annotations

from aenum import Enum


class ItemTypeEnum(Enum):
    Model = 'Model'
    TimeSeries = 'TimeSeries'
    Other = 'Other'


class VisualizationTypeEnum(Enum):
    Area = 'Area'
    Point = 'Point'
    Line = 'Line'
    Border = 'Border'
    Other = 'Other'


class TopologyTypeEnum(Enum):
    Area = 'Area'
    Node = 'Node'
    NodeConnectedElement = 'NodeConnectedElement'
    Edge = 'Edge'
    Other = 'Other'
