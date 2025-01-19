from pint import UnitRegistry, Unit, Quantity


ureg = UnitRegistry()

ureg.define("EUR = [currency]")  # Define EUR as a custom currency unit
ureg.define("USD = [currency]")  # Define USD as a custom currency unit
ureg.define("per_unit = []")
W = ureg.watt
kW = ureg.kilowatt
MW = ureg.megawatt
Wh = ureg.watt_hour
kWh = ureg.kilowatt_hour
MWh = ureg.megawatt_hour
EUR = ureg("EUR")
USD = ureg("USD")
second = ureg.second
minute = ureg.minute
hour = ureg.hour
day = ureg.day
week = ureg.week
year = ureg.year

# Compound units
EUR_per_Wh = EUR / Wh
EUR_per_kWh = EUR / kWh
EUR_per_MWh = EUR / MWh
USD_per_Wh = USD / Wh
USD_per_kWh = USD / kWh
USD_per_MWh = USD / MWh
per_unit = ureg("per_unit")
perc = ureg.percent

# Custom units for specific use cases
ureg.define("MTU = []")  # Market Time Unit
MTU = ureg("MTU")

ureg.define("NaU = []")  # Not a Unit; no physical meaning, dimensionless
NaU = ureg("NaU")

ureg.define("MissingUnit = []")  # For missing units
MissingUnit = ureg("MissingUnit")


def convert(value: float, from_unit: Unit, to_unit: Unit) -> float:
    """Converts value from one unit to another using pint."""
    quantity = value * from_unit
    return quantity.to(to_unit).magnitude


def get_pretty_text_value(
        quantity: Quantity,
        decimals: int = None,
        order_of_magnitude: float = None,
        include_unit: bool = True
) -> str:
    if order_of_magnitude is not None:
        order_of_magnitude = _check_order_of_magnitude(order_of_magnitude)

    if _is_currency(quantity):
        value, unit = _get_value_unit_for_currency(quantity, order_of_magnitude, include_unit)
        decimals = 2 if decimals is None else decimals
    elif quantity.units == per_unit.units:
        value = float(quantity.magnitude)
        unit = str(quantity.units)
        decimals = 2
    elif quantity.units == MTU.units:
        value = int(quantity.magnitude)
        unit = str(quantity.units)
        if value > 1:
            unit += 's'
        decimals = None
    else:
        value, unit = _get_value_unit_for_physical_quantity(quantity, order_of_magnitude, include_unit)
        if decimals is None:
            decimals = _get_default_decimals(value)

    if quantity.units == NaU.units:
        unit = ''

    if decimals is not None:
        value = round(value, decimals)

    return f"{value}{' ' + unit if unit else ''}"


def _get_value_unit_for_currency(
        quantity: Quantity,
        order_of_magnitude: float = None,
        include_unit: bool = True
) -> tuple[float, str]:
    value = float(quantity.magnitude)
    unit = str(quantity.units) if include_unit else ""

    if order_of_magnitude is None:
        # Auto-select order of magnitude to keep max 4 digits before decimal
        if abs(value) >= 1e9:
            order_of_magnitude = 1e9
        elif abs(value) >= 1e6:
            order_of_magnitude = 1e6
        elif abs(value) >= 1e4:
            order_of_magnitude = 1e3

    if order_of_magnitude is not None:
        value = value / order_of_magnitude
        if order_of_magnitude == 1e3:
            prefix = "k"
        elif order_of_magnitude == 1e6:
            prefix = "M"
        elif order_of_magnitude == 1e9:
            prefix = "B"
        else:
            prefix = ""
        unit = f"{prefix}{unit}"
    unit = _format_unit_string(unit)
    return value, unit


def _get_value_unit_for_physical_quantity(
        quantity: Quantity,
        order_of_magnitude: float = None,
        include_unit: bool = True
) -> tuple[float, str]:
    if order_of_magnitude is not None:
        value = quantity.magnitude / order_of_magnitude
        scaled_quantity = value * quantity.units
    else:
        scaled_quantity = quantity.to_compact()
        value = scaled_quantity.magnitude

    unit = str(scaled_quantity.units) if include_unit else ""
    unit = _format_unit_string(unit)
    return value, unit


def _format_unit_string(unit: str) -> str:
    replacements = {
        'microwatt_hour': 'µWh',
        'milliwatt_hour': 'mWh',
        'terawatt_hour': 'TWh',
        'gigawatt_hour': 'GWh',
        'megawatt_hour': 'MWh',
        'kilowatt_hour': 'kWh',
        'watt_hour': 'Wh',

        'microwatt': 'µW',
        'terawatt': 'TW',
        'megawatt': 'MW',
        'milliwatt': 'mW',
        'gigawatt': 'GW',
        'kilowatt': 'kW',
        'watt': 'W',
    }

    parts = unit.split(' / ')
    formatted_parts = []

    for part in parts:
        formatted_part = part
        for old, new in replacements.items():
            formatted_part = formatted_part.replace(old, new)
        formatted_parts.append(formatted_part)

    return '/'.join(formatted_parts)


def _check_order_of_magnitude(order: float) -> float:
    log_order = round(float(f"{order:.2e}".split('e')[1]))

    if log_order % 3 != 0:
        raise ValueError(f"Order of magnitude must be a multiple of 3 (e.g., 1e-3, 1e3, 1e6). Got: {order}")

    return float(f"1e{log_order}")


def _get_default_decimals(value: float) -> int:
    if abs(value) >= 100:
        return 1
    elif abs(value) >= 10:
        return 2
    return 3


def _is_currency(quantity: Quantity) -> bool:
    base_units = quantity.dimensionality
    return '[currency]' in str(base_units)


if __name__ == "__main__":
    test_values = [
        1234.5678912345 * W,
        1.23456789 * MW,
        123456.789 * EUR,
        0.123456789 * EUR,
        9876.54321 * EUR_per_MWh,
        1234567.89 * Wh,
        0.9876544 * per_unit,
        0.9876544 * NaU,
        123 * MTU,
    ]

    print("\nTesting normal formatting:")
    for value in test_values:
        print(f"Original: {value}")
        print(f"Formatted: {get_pretty_text_value(value)}")
        print("---")

    print("\nTesting order of magnitude validation:")
    try:
        print(get_pretty_text_value(1000 * W, order_of_magnitude=1e2))
    except ValueError as e:
        print(f"Caught expected error: {e}")
