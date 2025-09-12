"""
MESCAL Area and Area Border Accounting Package

Tools for transforming node-line topology into area-based models and variables. 
Supports multi-level aggregation (countries, bidding zones, market regions) and 
cross-border flow analysis for energy market studies.

Core Components
---------------

**Model Generators:**
- Area model creation from node-to-area mappings with geographic visualization
- Border identification from transmission topology with standardized naming
- Network graph generation and geometric analysis for spatial representation

**Variable Calculators:**
- Area variables: Price aggregation (volume-weighted) and sum-based calculations
- Border variables: Cross-border flows, capacity aggregation, and price spreads
- Flexible node-to-area mapping with time series support
"""

from .area_model_generator import AreaModelGenerator
from .border_model_generator import AreaBorderModelGenerator, AreaBorderNamingConventions
from .border_model_geometry_calculator import AreaBorderGeometryCalculator, NonCrossingPathFinder

from .area_variable_base import AreaVariableCalculatorBase
from .border_variable_base import AreaBorderVariableCalculatorBase
from .model_generator_base import GeoModelGeneratorBase

from .area_variable_price_calculator import AreaPriceCalculator
from .area_variable_sum_calculator import AreaSumCalculator

from .border_variable_flow_calculator import BorderFlowCalculator
from .border_variable_price_spread_calculator import BorderPriceSpreadCalculator
from .border_variable_capacity_calculator import BorderCapacityCalculator

__all__ = [
    # Core model generators
    'AreaModelGenerator',
    'AreaBorderModelGenerator', 
    'AreaBorderNamingConventions',
    'AreaBorderGeometryCalculator',
    'NonCrossingPathFinder',
    
    # Base classes for extensibility
    'AreaVariableCalculatorBase',
    'AreaBorderVariableCalculatorBase', 
    'GeoModelGeneratorBase',
    
    # Area variable calculators
    'AreaPriceCalculator',
    'AreaSumCalculator',
    
    # Border variable calculators
    'BorderFlowCalculator',
    'BorderPriceSpreadCalculator', 
    'BorderCapacityCalculator',
]

__version__ = '0.1.0'
