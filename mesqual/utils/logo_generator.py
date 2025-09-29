import math


def generate_hexagon_svg(
        num_hexagons=2,
        line_width=7,
        spacing=11,
        q_line_length=40,
        q_line_width=10,
        center_x=50,
        center_y=50,
        stroke_color="#00A98F"
):
    """
    Generates SVG code for a multi-layered, point-topped hexagon logo with a 'Q' tail.

    Args:
        num_hexagons (int): The number of concentric hexagons.
        line_width (float): The stroke width for the hexagon lines.
        spacing (float): The distance between the centerlines of adjacent hexagons.
        q_line_length (float): The total length of the 'Q' tail line.
        q_line_width (float): The stroke width for the 'Q' tail.
        center_x (float): The X-coordinate of the center of the hexagons.
        center_y (float): The Y-coordinate of the center of the hexagons.
        stroke_color (str): The color of the lines.

    Returns:
        str: A string containing the complete SVG code.
    """
    svg_paths = []

    # --- Calculate initial radius to fit within 100x100 viewBox ---
    # The radius is from the center to a top/bottom vertex for a point-topped hexagon.
    initial_radius = 50 - (line_width / 2)

    # --- Generate Hexagon Paths ---
    for i in range(num_hexagons):
        radius = initial_radius - (i * spacing)
        if radius <= line_width / 2:  # Ensure hexagon is not too small
            continue

        points = []
        # Angles for a point-topped hexagon
        for angle_deg in range(30, 390, 60):
            angle_rad = math.radians(angle_deg)
            x = center_x + radius * math.cos(angle_rad)
            y = center_y + radius * math.sin(angle_rad)
            points.append(f"{x:.3f},{y:.3f}")

        path_data = f"M {points[0]} L {' '.join(points[1:])} Z"
        svg_paths.append(
            f'  <path d="{path_data}" fill="none" stroke="{stroke_color}" stroke-width="{line_width}"/>'
        )

    # --- Dynamically Calculate and Add the 'Q' Tail Path using Line Intersection ---
    if num_hexagons > 0:
        # Get radius of the middle hexagon
        middle_hex_index = (num_hexagons - 1) // 2
        middle_radius = initial_radius - ((num_hexagons - 1) * spacing / 2)

        # Get vertices of the middle hexagon
        hex_points = []
        for angle_deg in range(30, 390, 60):
            angle_rad = math.radians(angle_deg)
            x = center_x + middle_radius * math.cos(angle_rad)
            y = center_y + middle_radius * math.sin(angle_rad)
            hex_points.append((x, y))

        # Define the ray from the center at 60 degrees
        q_angle_deg = 60
        q_angle_rad = math.radians(q_angle_deg)
        ray_start = (center_x, center_y)
        # A point far along the ray to guarantee intersection
        ray_end = (center_x + 200 * math.cos(q_angle_rad), center_y + 200 * math.sin(q_angle_rad))

        intersection_point = None
        # Check for intersection with each hexagon edge
        for i in range(6):
            p1 = hex_points[i]
            p2 = hex_points[(i + 1) % 6]

            # Standard line-segment intersection formula
            x1, y1 = ray_start
            x2, y2 = ray_end
            x3, y3 = p1
            x4, y4 = p2

            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if den != 0:
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
                # We need t > 0 (in the direction of the ray) and 0 <= u <= 1 (on the hex segment)
                if t > 0 and 0 <= u <= 1:
                    px = x1 + t * (x2 - x1)
                    py = y1 + t * (y2 - y1)
                    intersection_point = (px, py)
                    break  # Found the intersection

        if intersection_point:
            q_center_x, q_center_y = intersection_point

            # The Q-line itself lies along the same 60-degree line
            half_len = q_line_length / 2
            dx = half_len * math.cos(q_angle_rad)
            dy = half_len * math.sin(q_angle_rad)

            start_x, start_y = q_center_x - dx, q_center_y - dy
            end_x, end_y = q_center_x + dx, q_center_y + dy

            q_tail_data = f"M {start_x:.3f},{start_y:.3f} L {end_x:.3f},{end_y:.3f}"
            svg_paths.append(
                f'  <path d="{q_tail_data}" stroke="{stroke_color}" stroke-width="{q_line_width}"/>'
            )

    # --- Assemble the Final SVG ---
    view_box_str = "0 0 100 100"
    svg_template = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="32" height="32" viewBox="{view_box_str}" fill="none" xmlns="http://www.w3.org/2000/svg">
{chr(10).join(svg_paths)}
</svg>'''

    return svg_template


if __name__ == "__main__":
    HEXAGON_COUNT = 2
    HEXAGON_LINE_WIDTH = 7
    DISTANCE_BETWEEN_CENTERS = 11
    Q_LINE_LENGTH = 40

    logo_svg_code = generate_hexagon_svg(
        num_hexagons=HEXAGON_COUNT,
        line_width=HEXAGON_LINE_WIDTH,
        spacing=DISTANCE_BETWEEN_CENTERS,
        q_line_length=Q_LINE_LENGTH
    )

    print(logo_svg_code)

    with open("generated_logo.svg", "w") as f:
        f.write(logo_svg_code)

    print("\\nLogo saved to generated_logo.svg")
