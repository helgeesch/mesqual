import logging
from typing import Iterator
import numpy as np
from pint import UnitRegistry, Unit, Quantity
from math import log10, ceil

from mesqual.enums import QuantityTypeEnum

logger = logging.getLogger(__name__)


class UnitNotFound(Exception):
    """Raised when a requested unit with specific order of magnitude is not found in the registry."""

    pass


class UnitRegistryNotComplete(Exception):
    """Raised when the Units class is missing expected unit definitions for a dimensionality."""

    def __init__(self, message: str = None):
        base = f'You should never end up here. Your units are not properly registered in the {Units.__name__} class.'
        message = message or ''
        super().__init__(base + message)


ureg = UnitRegistry(on_redefinition='ignore')

ureg.define('Wh = [energy]')
ureg.define('kWh = 1e3 Wh = kWh')
ureg.define('MWh = 1e6 Wh = MWh')
ureg.define('GWh = 1e9 Wh = GWh')
ureg.define('TWh = 1e12 Wh = TWh')

ureg.define('W = [power]')
ureg.define('kW = 1e3 W = kW')
ureg.define('MW = 1e6 W = MW')
ureg.define('GW = 1e9 W = GW')
ureg.define('TW = 1e12 W = TW')

ureg.define('EUR = [currency]')
ureg.define('EUR_cent = 1e-2 EUR')
ureg.define('kEUR = 1e3 EUR = kEUR')
ureg.define('MEUR = 1e6 EUR = MEUR')
ureg.define('BEUR = 1e9 EUR = BEUR')
ureg.define('TEUR = 1e12 EUR = TEUR')

ureg.define('EUR_per_W = EUR / W = [price_for_capacity]')
ureg.define('EUR_per_MW = EUR / MW')

ureg.define('EUR_per_W_per_period = EUR / (W * period) = [price_for_capacity_per_period]')
ureg.define('EUR_per_MW_per_period = EUR / (MW * period)')

ureg.define('EUR_per_Wh = EUR / Wh = [price_for_energy]')
ureg.define('EUR_per_MWh = EUR / MWh')

ureg.define('minute = [time]')
ureg.define('hour = 60 minute = hour')
ureg.define('day = 24 hour = day')
ureg.define('week = 7 day = week')
ureg.define('year = 365 day = year')

ureg.define('MTU = [mtu]')
ureg.define('period = [period]')
ureg.define('per_unit = [pu]')  # 0 - 1
ureg.define('ratio = [ratio]')  # 0 - 1
ureg.define('perc = [percentage]')  # 0 - 100
ureg.define('percent_base = 1e-2 = percent_base')

ureg.define('W_per_min = W / minute = [ramping]')
ureg.define('MW_per_min = MW / minute')
ureg.define('MW_per_hour = MW / hour')

ureg.define('W_per_period = W / period = [ramping_per_period]')
ureg.define('MW_per_period = MW / period')

ureg.define('EUR_per_W_per_min = EUR / (W / minute) = [price_for_ramping]')
ureg.define('EUR_per_MW_per_min = EUR / (MW / minute)')
ureg.define('EUR_per_MW_per_hour = EUR / (MW / hour)')

ureg.define('NaU = []')  # Not a Unit; no physical meaning, dimensionless
ureg.define('MissingUnit = []')  # For missing units


class _IterableUnitsMeta(type):
    def __iter__(cls) -> Iterator[Unit]:
        return (u for name, u in cls.__dict__.items() if isinstance(u, Unit))


class Units(metaclass=_IterableUnitsMeta):
    """Registry of energy system units with conversion and formatting helpers.

    `Units` extends the pint registry `ureg` with energy, power, currency,
    price, ramping and time units. Its class methods pick readable units,
    find a common unit for a list of quantities and format quantities as text.

    Unit families:

    - Energy: `Wh`, `kWh`, `MWh`, `GWh`, `TWh`
    - Power: `W`, `kW`, `MW`, `GW`, `TW`
    - Ramping: `W_per_min`, `MW_per_min`, `MW_per_hour`
    - Currency: `EUR`, `kEUR`, `MEUR`, `BEUR`, `TEUR`
    - Energy prices: `EUR_per_Wh`, `EUR_per_MWh`
    - Capacity prices: `EUR_per_W`, `EUR_per_MW` (per MW and hour, intensive),
      `EUR_per_W_per_period`, `EUR_per_MW_per_period` (per MW for one period, extensive)
    - Ramping prices: `EUR_per_W_per_min`, `EUR_per_MW_per_min`, `EUR_per_MW_per_hour`
    - Time: `minute`, `hour`, `day`, `week`, `year`
    - Dimensionless: `per_unit`, `percent`, `percent_base`, `MTU`, `NaU`, `MissingUnit`

    See Also:
        [`QuantityToTextConverter`][mesqual.units.QuantityToTextConverter]:
            Formats many quantities with the same settings.
    """

    _ureg = ureg
    Unit = Unit
    Quantity = Quantity

    Wh = _ureg.Wh
    kWh = _ureg.kWh
    MWh = _ureg.MWh
    GWh = _ureg.GWh
    TWh = _ureg.TWh

    W = _ureg.W
    kW = _ureg.kW
    MW = _ureg.MW
    GW = _ureg.GW
    TW = _ureg.TW

    W_per_min = _ureg.W_per_min
    MW_per_min = _ureg.MW_per_min
    MW_per_hour = _ureg.MW_per_hour

    W_per_period = _ureg.W_per_period
    MW_per_period = _ureg.MW_per_period

    EUR = _ureg.EUR
    kEUR = _ureg.kEUR
    MEUR = _ureg.MEUR
    BEUR = _ureg.BEUR
    TEUR = _ureg.TEUR

    EUR_per_W = _ureg.EUR_per_W
    EUR_per_MW = _ureg.EUR_per_MW

    EUR_per_W_per_period = _ureg.EUR_per_W_per_period
    EUR_per_MW_per_period = _ureg.EUR_per_MW_per_period

    EUR_per_Wh = _ureg.EUR_per_Wh
    EUR_per_MWh = _ureg.EUR_per_MWh

    EUR_per_W_per_min = _ureg.EUR_per_W_per_min
    EUR_per_MW_per_min = _ureg.EUR_per_MW_per_min
    EUR_per_MW_per_hour = _ureg.EUR_per_MW_per_hour

    percent_base = _ureg.percent_base
    percent = _ureg.perc
    ratio = _ureg.ratio
    per_unit = _ureg.per_unit
    MTU = _ureg.MTU
    NaU = _ureg.NaU
    MissingUnit = _ureg.MissingUnit

    _STRING_REPLACEMENTS = {
        '_per_': '/',
        'EUR': '€',
        'per_unit': 'pu',
        'perc': '%',
        'inf': '∞',
        'nan': 'N/A',
    }

    _QUANTITY_TYPES = {
        **dict.fromkeys([W, kW, MW, GW, TW], QuantityTypeEnum.INTENSIVE),
        **dict.fromkeys([Wh, kWh, MWh, GWh, TWh], QuantityTypeEnum.EXTENSIVE),
        **dict.fromkeys([W_per_min, MW_per_min, MW_per_hour], QuantityTypeEnum.INTENSIVE),
        **dict.fromkeys([W_per_period, MW_per_period], QuantityTypeEnum.EXTENSIVE),
        **dict.fromkeys([EUR, kEUR, MEUR, BEUR, TEUR], QuantityTypeEnum.EXTENSIVE),
        **dict.fromkeys([EUR_per_Wh, EUR_per_MWh], QuantityTypeEnum.INTENSIVE),
        **dict.fromkeys([EUR_per_W_per_min, EUR_per_MW_per_min, EUR_per_MW_per_hour], QuantityTypeEnum.INTENSIVE),
        **dict.fromkeys([EUR_per_W, EUR_per_MW], QuantityTypeEnum.INTENSIVE),
        **dict.fromkeys([EUR_per_W_per_period, EUR_per_MW_per_period], QuantityTypeEnum.EXTENSIVE),
        **dict.fromkeys([percent, percent_base, ratio, per_unit], QuantityTypeEnum.INTENSIVE),
        MTU: QuantityTypeEnum.EXTENSIVE,
        NaU: QuantityTypeEnum.UNKNOWN,
        MissingUnit: QuantityTypeEnum.UNKNOWN,
    }
    """Quantity type per unit. A new unit gets an entry here."""

    _RESCALED_BASE_UNITS = (Wh, W, EUR, W_per_min, W_per_period)
    """Units of these families change their order of magnitude in pretty output.
    Prices, ratios and percentages keep their unit."""

    @classmethod
    def get_quantity_type_enum(cls, unit: Unit) -> QuantityTypeEnum:
        """
        Classify a unit as intensive or extensive quantity.

        Intensive quantities (e.g., power, prices) keep their value when the
        time resolution changes; extensive quantities (e.g., energy, cost) scale
        with the period length. A limit per period (`MW_per_period`) and a cost
        per MW for one period (`EUR_per_MW_per_period`) are extensive. A
        capacity price per MW and hour (`EUR_per_MW`) is intensive. No unit in
        `Units` is AMBIGUOUS. A platform marks its own ambiguous units with
        `register_quantity_type`; their flag specs state the quantity type.

        Args:
            unit: The unit to classify

        Returns:
            INTENSIVE or EXTENSIVE. UNKNOWN for `NaU`, `MissingUnit`
                and units outside the registry.

        Examples:
            >>> Units.get_quantity_type_enum(Units.MW)
            <QuantityTypeEnum.INTENSIVE: 'intensive'>
            >>> Units.get_quantity_type_enum(Units.EUR_per_MW_per_period).name
            'EXTENSIVE'
        """
        return cls._QUANTITY_TYPES.get(unit, QuantityTypeEnum.UNKNOWN)

    @classmethod
    def register_quantity_type(cls, unit: Unit, quantity_type: QuantityTypeEnum) -> None:
        """Set the quantity type of a unit that is not in `Units`.

        A platform calls this for its own units, for example to mark a unit
        `AMBIGUOUS` so that each flag spec with that unit states its quantity
        type. Registering the same type again does nothing.

        Args:
            unit: A unit of the mesqual unit registry (`mesqual.units.ureg`).
            quantity_type: The quantity type of the unit.

        Raises:
            ValueError: If the unit already has another quantity type.

        Examples:
            >>> from mesqual.enums import QuantityTypeEnum
            >>> unit = Units.EUR / Units.MTU
            >>> Units.register_quantity_type(unit, QuantityTypeEnum.AMBIGUOUS)
            >>> Units.get_quantity_type_enum(unit).name
            'AMBIGUOUS'
        """
        known = cls._QUANTITY_TYPES.get(unit)
        if known is not None and known is not quantity_type:
            raise ValueError(f'The unit {unit} already has the quantity type {known.name}.')
        cls._QUANTITY_TYPES[unit] = quantity_type

    @classmethod
    def is_rescaled_in_pretty_output(cls, unit: Unit) -> bool:
        """`True` if pretty output may change the order of magnitude of the unit.

        Energy, power, currency and ramping units rescale. Prices, ratios and
        percentages keep their unit.

        Args:
            unit: The unit to check.
        """
        return any(cls.units_have_same_base(unit, base) for base in cls._RESCALED_BASE_UNITS)

    @classmethod
    def units_have_same_base(cls, unit_1: Unit, unit_2: Unit) -> bool:
        """
        Check if two units have the same dimensionality (are convertible).

        Args:
            unit_1: First unit to compare
            unit_2: Second unit to compare

        Returns:
            True if units have same dimensionality, False otherwise

        Examples:
            >>> Units.units_have_same_base(Units.kWh, Units.MWh)
            True
            >>> Units.units_have_same_base(Units.kWh, Units.MW)
            False
        """
        return unit_1.dimensionality == unit_2.dimensionality

    @classmethod
    def get_base_unit_for_unit(cls, unit: Unit) -> Unit:
        """
        Get the base unit (order of magnitude 1) for a given unit's dimensionality.

        Args:
            unit: Unit to find base unit for

        Returns:
            The base unit with order of magnitude 1 (e.g., W for power, Wh for energy)

        Examples:
            >>> print(Units.get_base_unit_for_unit(Units.GWh))
            Wh
        """
        return cls.get_target_unit_for_oom(unit, 1)

    @classmethod
    def get_oom_of_unit(cls, unit: Unit) -> float:
        """
        Get the order of magnitude of a unit relative to its base unit.

        Args:
            unit: Unit to determine order of magnitude for

        Returns:
            Order of magnitude as float (e.g., 1e6 for MW, 1e9 for GW)

        Examples:
            >>> Units.get_oom_of_unit(Units.MW)
            1000000.0
            >>> Units.get_oom_of_unit(Units.kWh)
            1000.0
        """
        return (1 * unit).to_base_units().magnitude

    @classmethod
    def get_target_unit_for_oom(cls, reference_unit: Unit, target_oom: float) -> Quantity:
        """
        Find a unit with exact order of magnitude within the same dimensionality.

        Args:
            reference_unit: Unit defining the dimensionality
            target_oom: Target order of magnitude (e.g., 1e6 for mega, 1e9 for giga)

        Returns:
            Unit with the exact target order of magnitude

        Raises:
            UnitNotFound: If no unit with exact target order of magnitude exists

        Examples:
            >>> print(Units.get_target_unit_for_oom(Units.EUR, 1e9))
            BEUR
        """
        units = cls.get_all_units_with_equal_base(reference_unit)
        for u in units:
            if cls.get_oom_of_unit(u) == target_oom:
                return u
        raise UnitNotFound(f'No unit with order of mag {target_oom:.0e} for {reference_unit}')

    @classmethod
    def get_closest_unit_for_oom(cls, reference_unit: Unit, target_oom: float) -> Quantity:
        """
        Find the closest unit for a target order of magnitude (doesn't require exact match).

        Selects the largest unit whose order of magnitude is less than or equal to
        the target order of magnitude.

        Args:
            reference_unit: Unit defining the dimensionality
            target_oom: Target order of magnitude

        Returns:
            Closest unit with order of magnitude <= target_oom

        Raises:
            UnitRegistryNotComplete: If no units found for the dimensionality

        Examples:
            >>> print(Units.get_closest_unit_for_oom(Units.W, 5e5))  # between kW and MW
            kW
        """
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
    def get_quantity_in_target_oom(cls, quantity: Quantity, target_oom: float) -> Quantity:
        """
        Convert quantity to unit with specific order of magnitude.

        Args:
            quantity: Quantity to convert
            target_oom: Target order of magnitude

        Returns:
            Quantity converted to target order of magnitude, or original if not found

        Examples:
            >>> print(Units.get_quantity_in_target_oom(5000 * Units.kWh, 1e6))
            5.0 MWh
        """
        try:
            target_unit = cls.get_target_unit_for_oom(quantity.units, target_oom)
            return quantity.to(target_unit)
        except UnitNotFound:
            logger.warning(f'No unit with order of magnitude {target_oom:.0e} for {quantity.units}. Keeping the unit.')
            return quantity

    @classmethod
    def get_quantity_in_target_unit(cls, quantity: Quantity, target_unit: Unit) -> Quantity:
        """
        Convert quantity to a specific target unit.

        Simple wrapper around pint's to() method for consistency with other unit conversion methods.

        Args:
            quantity: Quantity to convert
            target_unit: Target unit

        Returns:
            Quantity converted to target unit

        Examples:
            >>> print(Units.get_quantity_in_target_unit(5000 * Units.kWh, Units.MWh))
            5.0 MWh
        """
        return quantity.to(target_unit)

    @classmethod
    def get_quantity_in_pretty_unit(cls, quantity: Quantity) -> Quantity:
        """
        Convert quantity to the most readable unit (magnitude between 1 and 10,000).

        Automatically selects a unit where the magnitude is less than 10,000,
        making the value easy to read and comprehend. Only energy, power,
        currency and ramping units rescale; prices, ratios and percentages
        keep their unit.

        Args:
            quantity: Quantity to convert

        Returns:
            Quantity in "pretty" readable unit

        Examples:
            >>> print(Units.get_quantity_in_pretty_unit(54_321_000 * Units.Wh))
            54.321 MWh
            >>> print(Units.get_quantity_in_pretty_unit(12345 * Units.EUR_per_MWh))  # prices keep their unit
            12345 EUR_per_MWh
        """
        if not cls.is_rescaled_in_pretty_output(quantity.units):
            return quantity
        base_unit = cls.get_base_unit_for_unit(quantity.units)
        units = cls.get_all_units_with_equal_base(base_unit)
        units = sorted(units, key=lambda x: (1 * x).to(base_unit).magnitude, reverse=False)
        for u in units:
            if abs(quantity.to(u).magnitude) < 10_000:
                return quantity.to(u)
        return quantity.to(units[-1])

    @classmethod
    def get_common_pretty_unit_for_quantities(cls, quantities: list[Quantity]) -> Unit:
        """
        Find common "pretty" unit for a collection of quantities.

        The method selects the unit in which most values have an absolute
        magnitude below 10,000. If units tie, it selects the unit with the
        largest median magnitude.

        Args:
            quantities: List of quantities with same dimensionality

        Returns:
            Pretty unit that works well for the collection

        Raises:
            ValueError: If quantities have different dimensionalities or list is empty

        Examples:
            >>> print(Units.get_common_pretty_unit_for_quantities([1_500 * Units.EUR, 25_000_000 * Units.EUR]))
            MEUR
        """
        if not quantities:
            raise ValueError('Cannot find common unit for empty list of quantities')

        units = {q.units for q in quantities}
        if len(units) == 1 and not cls.is_rescaled_in_pretty_output(quantities[0].units):
            return quantities[0].units

        # Verify all quantities have same dimensionality
        base_unit = cls.get_base_unit_for_unit(quantities[0].units)
        for q in quantities[1:]:
            if not cls.units_have_same_base(q.units, base_unit):
                raise ValueError(
                    f'All quantities must have same dimensionality. Found {q.units} which differs from {base_unit}'
                )

        # Handle all-zero case
        non_zero_quantities = [q for q in quantities if abs(q.magnitude) > 0]
        if not non_zero_quantities:
            # All values are zero, return common unit if all same, else base unit
            if len(set([q.units for q in quantities])) == 1:
                return quantities[0].units
            return base_unit

        # Get all available units for this dimensionality
        available_units = cls.get_all_units_with_equal_base(base_unit)
        if not available_units:
            # return common unit if all same, else base unit
            if len(set([q.units for q in quantities])) == 1:
                return quantities[0].units
            return base_unit

        # Evaluate each unit
        best_unit = base_unit
        best_count_under_10k = -1
        best_median_magnitude = -1

        for unit in available_units:
            # Convert all quantities to this unit
            magnitudes = [abs(q.to(unit).magnitude) for q in non_zero_quantities]

            # Count how many values are under 10,000
            count_under_10k = sum(1 for m in magnitudes if m < 10_000)

            # Get median magnitude for tie-breaking
            median_magnitude = np.median(magnitudes)

            # Update best if this unit is better
            if count_under_10k > best_count_under_10k or (
                count_under_10k == best_count_under_10k and median_magnitude > best_median_magnitude
            ):
                best_unit = unit
                best_count_under_10k = count_under_10k
                best_median_magnitude = median_magnitude

        return best_unit

    @classmethod
    def get_all_units_with_equal_base(cls, unit: Unit) -> list[Unit]:
        """
        Get all registered units with the same dimensionality as the input unit.

        Args:
            unit: Reference unit to match dimensionality

        Returns:
            List of all units with same dimensionality (e.g., all energy units, all power units)

        Examples:
            >>> [str(u) for u in Units.get_all_units_with_equal_base(Units.MW)]
            ['W', 'kW', 'MW', 'GW', 'TW']
        """
        return [u for u in Units if cls.units_have_same_base(unit, u)]

    @classmethod
    def get_pretty_text_for_quantity(
        cls,
        quantity: Quantity,
        decimals: int = None,
        thousands_separator: str = None,
        include_unit: bool = True,
        include_oom: bool = True,
        include_sign: bool = None,
    ) -> str:
        """
        Format a quantity as human-readable text with customizable formatting.

        Applies string replacements for better readability (e.g., '_per_' → '/', 'EUR' → '€').

        Args:
            quantity: Quantity to format
            decimals: Number of decimal places (auto-determined if None)
            thousands_separator: Separator for thousands (default: '')
            include_unit: Whether to include unit in output (default: True)
            include_oom: Whether to include order of magnitude prefix (default: True)
            include_sign: Whether to include '+' for positive values (default: None/auto)

        Returns:
            Formatted text string representation of the quantity

        Examples:
            >>> price = 45.678 * Units.EUR_per_MWh
            >>> Units.get_pretty_text_for_quantity(price, decimals=2)
            '45.68 €/MWh'

            >>> cost = 1234567 * Units.EUR
            >>> Units.get_pretty_text_for_quantity(cost, thousands_separator=' ')
            '1 234 567 €'

            >>> power = 5.2 * Units.MW
            >>> Units.get_pretty_text_for_quantity(power, include_sign=True)
            '+5.20 MW'
        """
        if decimals is None:
            decimals = cls.get_pretty_decimals(quantity)
        if thousands_separator is None:
            thousands_separator = ''

        sign_str = cls._get_sign_str_for_quantity(quantity, include_sign)
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
            components.append(' ' + unit_str)

        pretty_text = ''.join(components)

        for r, v in cls._STRING_REPLACEMENTS.items():
            pretty_text = pretty_text.replace(r, v)

        return pretty_text

    @classmethod
    def _get_sign_str_for_quantity(cls, quantity: Quantity, include_sign: bool = None) -> str:
        """
        Get the sign string for a quantity value.

        Internal helper method for formatting quantities with appropriate sign representation.

        Args:
            quantity: Quantity to get sign for
            include_sign: Whether to include '+' for positive values

        Returns:
            Sign string: '+', '-', or '' (empty)

        Note:
            Returns empty string for NaN, zero, or when include_sign is False.
            For positive values, only returns '+' if include_sign is explicitly True.
        """
        if include_sign is False:
            return ''

        value = quantity.magnitude
        if np.isnan(value):
            return ''
        if value == 0:
            return ''
        if value < 0:
            return '-'
        if value > 0:
            if include_sign:
                return '+'
            else:
                return ''
        raise Exception(f'How did you end up here for value {quantity}')

    @classmethod
    def get_pretty_decimals(cls, quantity: Quantity) -> int:
        """
        Determine appropriate number of decimal places for a quantity's magnitude.

        A magnitude of the Python type `int` gets 0 decimals. Else the
        number of decimals depends on the absolute magnitude:

        - Above 100: 0 decimals
        - Above 10, up to 100: 1 decimal
        - Above 0.1, up to 10: 2 decimals
        - Above 0.01, up to 0.1: 3 decimals
        - Zero: 0 decimals
        - Else, above 0 up to 0.01: 5 decimals

        Args:
            quantity: Quantity to determine decimal places for

        Returns:
            Number of decimal places (0-5)

        Examples:
            >>> [Units.get_pretty_decimals(value * Units.MW) for value in (1234.5, 12.34, 0.0012)]
            [0, 1, 5]
        """
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
        elif abs_value == 0:
            return 0
        else:
            return 5

    @classmethod
    def _get_units_oom_prefix(cls, unit: Unit) -> str:
        """
        Extract the order of magnitude prefix from a unit.

        Internal helper that returns the prefix (k, M, G, T) by removing the base unit.

        Args:
            unit: Unit to extract prefix from

        Returns:
            Order of magnitude prefix string (e.g., 'k', 'M', 'G')

        Examples:
            >>> Units._get_units_oom_prefix(Units.MW)
            'M'
            >>> Units._get_units_oom_prefix(Units.kWh)
            'k'
        """
        base_unit = cls.get_base_unit_for_unit(unit)
        return str(unit).replace(str(base_unit), '')


class QuantityToTextConverter:
    """
    Configurable converter for formatting Quantity objects as text strings.

    Stores formatting configuration that can be reused across multiple quantity
    conversions, enabling consistent formatting across KPI collections and visualizations.

    Args:
        target_unit: Target unit for conversion (if None, uses pretty unit selection)
        decimals: Number of decimal places (if None, auto-determined)
        thousands_separator: Separator for thousands (default: '')
        include_unit: Whether to include unit in output (default: True)
        include_oom: Whether to include order of magnitude prefix (default: True)
        include_sign: Whether to include + sign for positive values (default: None/auto)

    Examples:
        >>> converter = QuantityToTextConverter(target_unit=Units.MWh, decimals=2)
        >>> converter.convert(5432.1 * Units.kWh)
        '5.43 MWh'
    """

    def __init__(
        self,
        target_unit: Unit = None,
        decimals: int = None,
        thousands_separator: str = None,
        include_unit: bool = True,
        include_oom: bool = True,
        include_sign: bool = None,
    ):
        self.target_unit = target_unit
        self.decimals = decimals
        self.thousands_separator = thousands_separator or ''
        self.include_unit = include_unit
        self.include_oom = include_oom
        self.include_sign = include_sign

    def convert(self, quantity: Quantity) -> str:
        """
        Convert a Quantity to formatted text string using stored configuration.

        Args:
            quantity: The quantity to format

        Returns:
            Formatted text representation
        """
        # Apply target unit conversion if specified
        if self.target_unit is not None:
            quantity = Units.get_quantity_in_target_unit(quantity, self.target_unit)
        else:
            quantity = Units.get_quantity_in_pretty_unit(quantity)

        # Use Units.get_pretty_text_for_quantity with stored configuration
        return Units.get_pretty_text_for_quantity(
            quantity,
            decimals=self.decimals,
            thousands_separator=self.thousands_separator,
            include_unit=self.include_unit,
            include_oom=self.include_oom,
            include_sign=self.include_sign,
        )

    @classmethod
    def from_quantities(
        cls,
        quantities: list[Quantity],
        target_unit: Unit = None,
        decimals: int = None,
        thousands_separator: str = None,
        include_unit: bool = True,
        include_oom: bool = True,
        include_sign: bool = None,
    ) -> 'QuantityToTextConverter':
        """
        Create converter auto-configured for a collection of quantities.

        Analyzes the provided quantities to determine an appropriate common unit
        (if target_unit not specified) that works well for all values.

        Args:
            quantities: Collection of quantities to analyze
            target_unit: Override auto-selected unit with specific target
            decimals: Number of decimal places (if None, will be auto-determined per value)
            thousands_separator: Separator for thousands (default: '')
            include_unit: Whether to include unit in output (default: True)
            include_oom: Whether to include order of magnitude prefix (default: True)
            include_sign: Whether to include + sign for positive values (default: None/auto)

        Returns:
            Configured QuantityToTextConverter instance

        Examples:
            >>> quantities = [1000 * Units.EUR, 5000 * Units.EUR, 10000 * Units.EUR]
            >>> converter = QuantityToTextConverter.from_quantities(quantities, decimals=0)
            >>> [converter.convert(quantity) for quantity in quantities]
            ['1 k€', '5 k€', '10 k€']
        """
        # Determine common pretty unit if not explicitly provided
        if target_unit is None and quantities:
            target_unit = Units.get_common_pretty_unit_for_quantities(quantities)

        if decimals is None:
            decimals = cls._pretty_decimal_precision([q.magnitude for q in quantities])

        return cls(
            target_unit=target_unit,
            decimals=decimals,
            thousands_separator=thousands_separator,
            include_unit=include_unit,
            include_oom=include_oom,
            include_sign=include_sign,
        )

    @staticmethod
    def _pretty_decimal_precision(values):
        """Determine minimal decimal places to differentiate values,
        with special rule: any |value| < 0.1 -> at least 2 decimals."""
        if len(values) <= 1:
            return None

        sorted_vals = np.sort(values)
        diffs = np.diff(sorted_vals)
        non_zero_diffs = diffs[diffs > 0]

        if len(non_zero_diffs) == 0:
            return None  # all values identical

        median_diff = np.median(non_zero_diffs)
        decimals = max(0, ceil(-log10(median_diff)))

        # Apply special rule for small absolute values
        if np.any(np.abs(values) < 0.1):
            decimals = max(decimals, 2)

        return decimals
