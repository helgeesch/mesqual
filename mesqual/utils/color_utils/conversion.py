import re
from typing import Literal
import matplotlib.colors as mcolors

COLOR_INPUT_TYPES = str | tuple[float, float, float] | tuple[float, float, float, float]
TARGET_FORMAT_TYPES = Literal['hex', 'rgb_tuple', 'rgba_tuple', 'rgb_string', 'rgba_string', 'name', 'hex_a']


def detect_color_type(color: COLOR_INPUT_TYPES) -> Literal['hex', 'hex_a', 'name', 'rgb_tuple', 'rgba_tuple', 'rgb_string']:
    """Detect the type of the color input."""
    if isinstance(color, str):
        if color.startswith('#'):
            if len(color) == 7:
                return 'hex'
            elif len(color) == 9:
                return 'hex_a'
        elif color in mcolors.CSS4_COLORS:
            return 'name'
        elif color.startswith('rgb'):
            return 'rgb_string'
    elif isinstance(color, tuple):
        if len(color) == 3:
            return 'rgb_tuple'
        elif len(color) == 4:
            return 'rgba_tuple'
    raise ValueError("Unsupported color format")


def parse_rgb_string(color: str) -> tuple[float, float, float, float]:
    """Parse an rgb or rgba string into a tuple of floats."""
    match = re.fullmatch(r'rgb\((\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3})\)', color)
    if match:
        r, g, b = map(int, match.groups())
        return (r / 255, g / 255, b / 255, 1.0)
    match = re.fullmatch(r'rgba\((\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3}),\s*(\d*\.?\d+)\)', color)
    if match:
        r, g, b, a = match.groups()
        return (int(r) / 255, int(g) / 255, int(b) / 255, float(a))
    raise ValueError("Invalid RGB or RGBA string format")


def to_rgba(color: COLOR_INPUT_TYPES) -> tuple[float, float, float, float]:
    color_type = detect_color_type(color)

    if color_type in ['hex', 'hex_a']:
        return mcolors.to_rgba(color)
    elif color_type == 'name':
        return mcolors.to_rgba(mcolors.CSS4_COLORS[color])
    elif color_type == 'rgb_tuple':
        return color + (1.0,)
    elif color_type == 'rgba_tuple':
        return color
    elif color_type == 'rgb_string':
        return parse_rgb_string(color)
    else:
        raise ValueError("Unsupported color format")


def to_hex(color: COLOR_INPUT_TYPES) -> str:
    return mcolors.to_hex(to_rgba(color))


def to_hex_a(color: COLOR_INPUT_TYPES) -> str:
    return mcolors.to_hex(to_rgba(color), keep_alpha=True)


def to_rgb_tuple(color: COLOR_INPUT_TYPES) -> tuple[float, float, float]:
    return to_rgba(color)[:3]


def to_rgba_tuple(color: COLOR_INPUT_TYPES) -> tuple[float, float, float, float]:
    return to_rgba(color)


def to_rgb_string(color: COLOR_INPUT_TYPES) -> str:
    rgba = to_rgba(color)
    return f'rgb({int(rgba[0] * 255)}, {int(rgba[1] * 255)}, {int(rgba[2] * 255)})'


def to_rgba_string(color: COLOR_INPUT_TYPES) -> str:
    rgba = to_rgba(color)
    return f'rgba({int(rgba[0] * 255)}, {int(rgba[1] * 255)}, {int(rgba[2] * 255)}, {rgba[3]})'


def to_name(color: COLOR_INPUT_TYPES) -> str:
    rgba = to_rgba(color)
    hex_color = mcolors.to_hex(rgba)
    for name, hex_code in mcolors.CSS4_COLORS.items():
        if hex_code == hex_color:
            return name
    return hex_color


def convert_color(color: COLOR_INPUT_TYPES, target_format: TARGET_FORMAT_TYPES) -> COLOR_INPUT_TYPES:
    if target_format == 'hex':
        return to_hex(color)
    elif target_format == 'hex_a':
        return to_hex_a(color)
    elif target_format == 'rgb_tuple':
        return to_rgb_tuple(color)
    elif target_format == 'rgba_tuple':
        return to_rgba_tuple(color)
    elif target_format == 'rgb_string':
        return to_rgb_string(color)
    elif target_format == 'rgba_string':
        return to_rgba_string(color)
    elif target_format == 'name':
        return to_name(color)
    else:
        raise ValueError("Unsupported target format")


if __name__ == '__main__':
    color_input = "#1f77b4"
    converted_color = convert_color(color_input, "rgb_tuple")
    print(f"Converted to rgb_tuple: {converted_color}")

    color_input = "skyblue"
    converted_color = convert_color(color_input, "rgba_string")
    print(f"Converted to rgba_string: {converted_color}")

    color_input = (0.1, 0.2, 0.3)
    converted_color = convert_color(color_input, "hex")
    print(f"Converted to hex: {converted_color}")

    color_input = (0.1, 0.2, 0.3, 0.5)
    converted_color = convert_color(color_input, "name")
    print(f"Converted to name: {converted_color}")

    color_input = "rgb(17, 36, 20)"
    converted_color = convert_color(color_input, "hex")
    print(f"Converted to hex: {converted_color}")
