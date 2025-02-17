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


class QuantityTypeEnum(Enum):
    """
    Physical property that determines how a quantity behaves under time aggregation.

    For time series resampling operations, quantities are classified based on their
    behavior when changing the time granularity:

    INTENSIVE: Quantities measured per unit (rates, densities, concentrations)
       - Replication when increasing granularity (e.g. hourly -> 15min)
       - Averaging when reducing granularity (e.g. hourly -> daily)
       Examples: prices [€/MWh], power [MW], flow rates [MW]

    EXTENSIVE: Quantities representing totals or amounts
       - Splitting when increasing granularity (e.g. hourly -> 15min)
       - Summation when reducing granularity (e.g. hourly -> daily)
       Examples: welfare [€], volume [MWh], energy [MWh]
   """

    INTENSIVE = "intensive"  # price, power, flow rate
    EXTENSIVE = "extensive"  # welfare, volume, energy


# class GranularityTypeEnum(Enum):
#     PT15 = 15
#     PT30 = 30
#     PT60 = 60
#     PT120 = 120
