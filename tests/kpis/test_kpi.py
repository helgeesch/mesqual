from types import SimpleNamespace

import pytest

from mesqual.enums import TopologyTypeEnum
from mesqual.flag import FlagIndex, ModelFlagSpec
from mesqual.kpis.kpi import KPI
from mesqual.units import Units


def kpi_for(unit) -> KPI:
    spec = ModelFlagSpec(name='Node.Model', description='Nodes.', index='Node', topology_type=TopologyTypeEnum.Node)
    dataset = SimpleNamespace(flag_index=FlagIndex([spec]))
    attributes = SimpleNamespace(unit=unit, flag='Node.Model', custom_name='count')
    return KPI(value=3, attributes=attributes, dataset=dataset)


def test_quantity_without_unit_raises_a_clear_error():
    with pytest.raises(ValueError, match='no unit'):
        kpi_for(None).quantity


def test_quantity_uses_the_kpi_unit():
    assert kpi_for(Units.MW).quantity == 3 * Units.MW
