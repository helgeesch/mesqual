from pint import UnitRegistry, Unit, Quantity


ureg = UnitRegistry()

ureg.define("EUR = [currency]")  # Define EUR as a custom currency unit
ureg.define("USD = [currency]")  # Define USD as a custom currency unit

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
per_unit = ureg.dimensionless
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


def get_pretty_text(value: float, unit: Unit, **kwargs) -> str:
    """Returns a formatted string of the value with the unit."""
    # TODO: more advanced
    quantity = value * unit
    return f'{quantity:.2f}'


if __name__ == "__main__":
    value_in_watts = convert(1, MW, W)
    print(f"1 MW is {value_in_watts} W")

    value_in_seconds = convert(1, hour, second)
    print(f"1 hour is {value_in_seconds} seconds")

    pretty_text_mtu = get_pretty_text(100, MTU)
    print(f"Pretty text for 100 MTU: {pretty_text_mtu}")

    # Using NoneUnit as a placeholder (it will display without a unit symbol)
    pretty_text_none = get_pretty_text(100, NoneUnit)
    print(f"Pretty text for 100 NoneUnit: {pretty_text_none}")

    # Using NA as a placeholder
    pretty_text_na = get_pretty_text(100, NA)
    print(f"Pretty text for 100 NA: {pretty_text_na}")

    # Define a project-specific unit
    ureg.define("my_project_specific_unit = 1 []")  # Defines a dimensionless custom unit
    my_project_specific_unit = ureg("my_project_specific_unit")
    pretty_text_project_specific_unit = get_pretty_text(100, my_project_specific_unit)
    print(f"Pretty text for 100 project_specific_unit: {pretty_text_project_specific_unit}")
