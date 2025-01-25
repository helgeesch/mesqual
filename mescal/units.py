from typing import Iterator
import numpy as np

from pint import UnitRegistry


class UnitNotFound(Exception):
    pass


class UnitRegistryNotComplete(Exception):
    def __init__(self, message: str = None):
        base = f'You should never end up here. Your units are not properly registered in the {Units.__name__} class.'
        message = message or ''
        super().__init__(base + message)


ureg = UnitRegistry()

ureg.define("Wh = [energy]")
ureg.define("kWh = 1e3 Wh = kWh")
ureg.define("MWh = 1e6 Wh = MWh")
ureg.define("GWh = 1e9 Wh = GWh")
ureg.define("TWh = 1e12 Wh = TWh")

ureg.define("W = [power]")
ureg.define("kW = 1e3 W = kW")
ureg.define("MW = 1e6 W = MW")
ureg.define("GW = 1e9 W = GW")
ureg.define("TW = 1e12 W = TW")

ureg.define("EUR = [currency]")
ureg.define("EUR_cent = 1e-2 EUR")
ureg.define("kEUR = 1e3 EUR = kEUR")
ureg.define("MEUR = 1e6 EUR = MEUR")
ureg.define("BEUR = 1e9 EUR = BEUR")
ureg.define("TEUR = 1e12 EUR = TEUR")

ureg.define("EUR_per_Wh = EUR / Wh = [price]")
ureg.define("EUR_per_MWh = EUR / MWh")

ureg.define("minute = [time]")
ureg.define("hour = 60 minute = hour")
ureg.define("day = 24 hour = day")
ureg.define("week = 7 day = week")
ureg.define("year = 365 day = year")

ureg.define("MTU = [mtu]")
ureg.define("per_unit = [pu]")
ureg.define("perc = [percentage]")
ureg.define("percent_base = 1e-2 percent = percent_base")


ureg.define("NaU = []")  # Not a Unit; no physical meaning, dimensionless
ureg.define("MissingUnit = []")  # For missing units


class _IterableUnitsMeta(type):
    def __iter__(cls) -> Iterator[ureg.Unit]:
        return (u for name, u in cls.__dict__.items() if isinstance(u, ureg.Unit))


class Units(metaclass=_IterableUnitsMeta):
    Unit = ureg.Unit
    Quantity = ureg.Quantity

    Wh = ureg.Wh
    kWh = ureg.kWh
    MWh = ureg.MWh
    GWh = ureg.GWh
    TWh = ureg.TWh

    W = ureg.W
    kW = ureg.kW
    MW = ureg.MW
    GW = ureg.GW
    TW = ureg.TW

    EUR = ureg.EUR
    kEUR = ureg.kEUR
    MEUR = ureg.MEUR
    BEUR = ureg.BEUR
    TEUR = ureg.TEUR

    EUR_per_Wh = ureg.EUR_per_Wh
    EUR_per_MWh = ureg.EUR_per_MWh

    percent_base = ureg.percent_base
    percent = ureg.perc
    per_unit = ureg.per_unit
    MTU = ureg.MTU
    NaU = ureg.NaU
    MissingUnit = ureg.MissingUnit

    _STRING_REPLACEMENTS = {
        '_per_': '/',
        'EUR': '€',
        'per_unit': 'pu',
        'perc': '%',
    }

    @classmethod
    def units_have_same_base(cls, unit_1: ureg.Unit, unit_2: ureg.Unit) -> bool:
        return unit_1.dimensionality == unit_2.dimensionality

    @classmethod
    def get_base_unit_for_unit(cls, unit: ureg.Unit) -> ureg.Unit:
        return cls.get_target_unit_for_oom(unit, 1)

    @classmethod
    def get_oom_of_unit(cls, unit: ureg.Unit) -> float:
        return (1 * unit).to_base_units().magnitude

    @classmethod
    def get_target_unit_for_oom(cls, reference_unit: ureg.Unit, target_oom: float) -> ureg.Quantity:
        units = cls.get_all_units_with_equal_base(reference_unit)
        for u in units:
            if cls.get_oom_of_unit(u) == target_oom:
                return u
        raise UnitNotFound(f'No unit with order of mag {target_oom:.0e} for {reference_unit}')

    @classmethod
    def get_closest_unit_for_oom(cls, reference_unit: ureg.Unit, target_oom: float) -> ureg.Quantity:
        units_with_same_dimension = cls.get_all_units_with_equal_base(reference_unit)
        if len(units_with_same_dimension) == 0:
            raise UnitRegistryNotComplete
        base_unit = cls.get_base_unit_for_unit(reference_unit)
        sorted_units = sorted(units_with_same_dimension, key=lambda x: (1 * x).to(base_unit).magnitude, reverse=True)
        for u in sorted_units:
            if (1 * u).to(base_unit).magnitude <= target_oom:
                return u
        return sorted_units[0]

    @classmethod
    def get_quantity_in_target_oom(cls, quantity: ureg.Quantity, target_oom: float) -> ureg.Quantity:
        try:
            target_unit = cls.get_target_unit_for_oom(quantity.units, target_oom)
            return quantity.to(target_unit)
        except UnitNotFound:
            RuntimeWarning(f'# TODO:')
            return quantity

    @classmethod
    def get_quantity_in_target_unit(cls, quantity: ureg.Quantity, target_unit: ureg.Unit) -> ureg.Quantity:
        return quantity.to(target_unit)

    @classmethod
    def get_quantity_in_pretty_unit(cls, quantity: ureg.Quantity) -> str:
        base_unit = cls.get_base_unit_for_unit(quantity.units)
        units = cls.get_all_units_with_equal_base(base_unit)
        units = sorted(units, key=lambda x: (1 * x).to(base_unit).magnitude, reverse=False)
        for u in units:
            if quantity.to(u).magnitude < 10_000:
                return quantity.to(u)
        return quantity.to(units[-1])

    @classmethod
    def get_all_units_with_equal_base(cls, unit: ureg.Unit) -> list[ureg.Unit]:
        return [u for u in Units if cls.units_have_same_base(unit, u)]

    @classmethod
    def get_pretty_text_for_quantity(
            cls,
            quantity: ureg.Quantity,
            decimals: int = None,
            thousands_separator: str = None,
            include_unit: bool = True,
            include_oom: bool = True,
            always_include_sign: bool = False,
    ) -> str:
        if decimals is None:
            decimals = cls._get_pretty_decimals(quantity)
        if thousands_separator is None:
            thousands_separator = ''

        sign_str = cls._get_sign_str_for_quantity(quantity, always_include_sign)
        value_str = f'{abs(quantity.magnitude):,.{decimals}f}'
        value_str = value_str.replace(',', thousands_separator)

        if include_unit:
            if not include_oom:
                raise NotImplementedError('Why would you do that?')
            unit_str = str(quantity.units)
        elif include_oom:
            unit_str = cls._get_units_oom_prefix(quantity.units)
        else:
            unit_str = ''

        components = []
        if sign_str:
            components.append(sign_str)

        components.append(value_str)

        if unit_str:
            for r, v in cls._STRING_REPLACEMENTS.items():
                unit_str = unit_str.replace(r, v)
            components.append(' ' + unit_str)

        return ''.join(components)

    @classmethod
    def _get_sign_str_for_quantity(cls, quantity: ureg.Quantity, always_include_sign: bool) -> str:

        value = quantity.magnitude
        if np.isnan(value):
            return ''
        if value == 0:
            return ''
        if value < 0:
            return '-'
        if value > 0:
            if always_include_sign:
                return '+'
            else:
                return ''
        raise Exception(f'How did you end up here for value {quantity}')


    @classmethod
    def _get_pretty_decimals(cls, quantity: ureg.Quantity) -> int:

        # if quantity.units == Units.per_unit:
        #     return 3

        if isinstance(quantity.magnitude, int):
            return 0

        abs_value = abs(quantity.magnitude)
        if abs_value > 100:
            return 0
        elif abs_value > 10:
            return 1
        elif abs_value > 0.1:
            return 2
        elif abs_value > 0.01:
            return 3
        else:
            return 5

    @classmethod
    def _get_units_oom_prefix(cls, unit: ureg.Unit) -> str:
        base_unit = cls.get_base_unit_for_unit(unit)
        return str(unit).replace(str(base_unit), '')


if __name__ == '__main__':
    test_values = [0.0123, 1.234, 1234.5678, 12345678.90123]
    test_units = [
        Units.Wh,
        Units.MWh,
        Units.GWh,
        Units.MW,
        Units.GW,
        Units.EUR,
        Units.EUR_per_MWh,
        Units.per_unit,
        Units.percent
    ]
    for uu in test_units:
        for vv in test_values:
            q = vv * uu
            qq = Units.get_quantity_in_pretty_unit(q)
            print(Units.get_pretty_text_for_quantity(qq))
