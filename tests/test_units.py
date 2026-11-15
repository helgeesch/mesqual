import logging

import pytest

from mesqual.enums import QuantityTypeEnum
from mesqual.units import Units, ureg


class TestQuantityType:
    def test_no_unit_is_ambiguous(self):
        assert all(
            Units.get_quantity_type_enum(unit) is not QuantityTypeEnum.AMBIGUOUS for unit in Units._QUANTITY_TYPES
        )

    @pytest.mark.parametrize(
        'unit',
        [
            Units.W_per_period,
            Units.MW_per_period,
            Units.EUR_per_W_per_period,
            Units.EUR_per_MW_per_period,
            Units.MWh,
            Units.EUR,
        ],
    )
    def test_extensive_units(self, unit):
        assert Units.get_quantity_type_enum(unit) is QuantityTypeEnum.EXTENSIVE

    @pytest.mark.parametrize(
        'unit',
        [
            Units.EUR_per_W,
            Units.EUR_per_MW,
            Units.percent,
            Units.percent_base,
            Units.EUR_per_W_per_min,
            Units.EUR_per_MW_per_min,
            Units.EUR_per_MW_per_hour,
            Units.EUR_per_MWh,
            Units.MW,
        ],
    )
    def test_intensive_units(self, unit):
        assert Units.get_quantity_type_enum(unit) is QuantityTypeEnum.INTENSIVE

    def test_not_a_unit_is_unknown(self):
        assert Units.get_quantity_type_enum(Units.NaU) is QuantityTypeEnum.UNKNOWN


class TestCapacityPricePerPeriod:
    def test_converts_between_w_and_mw(self):
        quantity = (2 * Units.EUR_per_MW_per_period).to(Units.EUR_per_W_per_period)
        assert quantity.magnitude == pytest.approx(2e-6)

    def test_does_not_convert_to_the_capacity_price_per_hour(self):
        assert not Units.units_have_same_base(Units.EUR_per_MW_per_period, Units.EUR_per_MW)

    def test_keeps_its_unit_in_pretty_output(self):
        quantity = Units.get_quantity_in_pretty_unit(12345 * Units.EUR_per_MW_per_period)
        assert quantity.units == Units.EUR_per_MW_per_period


def test_percent_base_is_one_hundredth():
    quantity = (1 * Units.percent_base).to_base_units()
    assert quantity.dimensionless
    assert quantity.magnitude == pytest.approx(1e-2)


class TestPrettyUnit:
    def test_price_keeps_its_unit(self):
        quantity = Units.get_quantity_in_pretty_unit(12345 * Units.EUR_per_MWh)
        assert quantity.units == Units.EUR_per_MWh
        assert quantity.magnitude == 12345

    def test_energy_rescales(self):
        assert Units.get_quantity_in_pretty_unit(12345 * Units.MWh).units == Units.GWh

    def test_common_unit_of_prices_is_their_unit(self):
        quantities = [12345 * Units.EUR_per_MWh, 5 * Units.EUR_per_MWh]
        assert Units.get_common_pretty_unit_for_quantities(quantities) == Units.EUR_per_MWh

    def test_missing_target_unit_logs_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger='mesqual.units'):
            quantity = Units.get_quantity_in_target_oom(5 * Units.EUR_per_MWh, 1e30)
        assert quantity.units == Units.EUR_per_MWh
        assert 'order of magnitude' in caplog.text


class TestRegisterQuantityType:
    @pytest.fixture(autouse=True)
    def own_table(self, monkeypatch):
        monkeypatch.setattr(Units, '_QUANTITY_TYPES', dict(Units._QUANTITY_TYPES))

    def test_platform_unit_gets_its_type(self):
        ureg.define('EUR_per_block = EUR / MW')
        Units.register_quantity_type(ureg.EUR_per_block, QuantityTypeEnum.AMBIGUOUS)
        assert Units.get_quantity_type_enum(ureg.EUR_per_block) is QuantityTypeEnum.AMBIGUOUS

    def test_same_type_again_does_nothing(self):
        Units.register_quantity_type(Units.MW, QuantityTypeEnum.INTENSIVE)
        assert Units.get_quantity_type_enum(Units.MW) is QuantityTypeEnum.INTENSIVE

    def test_other_type_for_a_known_unit_raises(self):
        with pytest.raises(ValueError, match='already has the quantity type'):
            Units.register_quantity_type(Units.EUR_per_MW, QuantityTypeEnum.AMBIGUOUS)
