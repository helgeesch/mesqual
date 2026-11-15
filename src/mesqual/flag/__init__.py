"""Flag specs, their dimensions and schemas, area and border layers, and the flag index that looks them up."""

from .flag import FlagTypeProtocol
from .dimension import REQUIRED, Dimension, VariantValues
from .schema import FrameSchema, Level
from .flag_spec import FlagSpec, ModelFlagSpec, TimeSeriesFlagSpec, OtherFlagSpec
from .layers import AreaLayer, BorderLayer, LayerSet
from .layer_spec import LayerFlagSpec, AreaFlagSpec, BorderFlagSpec, LayerVariantValues, area_flag_specs
from .flag_index import FlagIndex
from .exceptions import UnknownFlagError, AmbiguousOverrideError, FlagUnavailableError

__all__ = [
    'FlagTypeProtocol',
    'REQUIRED',
    'Dimension',
    'VariantValues',
    'FrameSchema',
    'Level',
    'FlagSpec',
    'ModelFlagSpec',
    'TimeSeriesFlagSpec',
    'OtherFlagSpec',
    'AreaLayer',
    'BorderLayer',
    'LayerSet',
    'LayerFlagSpec',
    'AreaFlagSpec',
    'BorderFlagSpec',
    'LayerVariantValues',
    'area_flag_specs',
    'FlagIndex',
    'UnknownFlagError',
    'AmbiguousOverrideError',
    'FlagUnavailableError',
]
