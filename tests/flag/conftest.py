import pytest

from mesqual.enums import QuantityTypeEnum
from mesqual.units import Units, ureg

ureg.define('EUR_per_slot = EUR / MW')


@pytest.fixture
def ambiguous_unit(monkeypatch) -> Units.Unit:
    """A platform unit marked AMBIGUOUS."""
    monkeypatch.setattr(Units, '_QUANTITY_TYPES', dict(Units._QUANTITY_TYPES))
    Units.register_quantity_type(ureg.EUR_per_slot, QuantityTypeEnum.AMBIGUOUS)
    return ureg.EUR_per_slot
