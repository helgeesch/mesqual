"""Area and border models and area variables from a node-line topology.

An area is a group of nodes, for example a country or a bidding zone. The
area column of the node model gives the area of each node.

- Models: `AreaModelGenerator` builds the area model,
  `AreaBorderModelGenerator` the border model from the lines between areas,
  and `AreaBorderGeometryCalculator` the border geometries for maps.
- Variables: `AreaPriceCalculator` aggregates node prices to area prices,
  optionally weighted, and `AreaSumCalculator` sums node values per area.
- `AreaMembership` gives the area of each node and of the ends of each line.
"""

from .area_membership import AreaMembership
from .area_model_generator import AreaModelGenerator
from .border_model_generator import AreaBorderModelGenerator, AreaBorderNamingConventions
from .border_model_geometry_calculator import AreaBorderGeometryCalculator, NonCrossingPathFinder

from .area_variable_base import AreaVariableCalculatorBase
from .model_generator_base import GeoModelGeneratorBase

from .area_variable_price_calculator import AreaPriceCalculator
from .area_variable_sum_calculator import AreaSumCalculator

__all__ = [
    # Core model generators
    'AreaModelGenerator',
    'AreaBorderModelGenerator',
    'AreaBorderNamingConventions',
    'AreaBorderGeometryCalculator',
    'NonCrossingPathFinder',
    # Base classes for extensibility
    'AreaMembership',
    'AreaVariableCalculatorBase',
    'GeoModelGeneratorBase',
    # Area variable calculators
    'AreaPriceCalculator',
    'AreaSumCalculator',
]
