from typing import Literal


class SpotlightFlowArrowGenerator:
    def __init__(
            self,
            color: str = "#2563eb",
            stroke_width: int = 10,
            width: int = 100,
            height: int = 100,
            speed: float = 20.0,  # pixels_per_second
            direction: Literal["right", "left", "up", "down"] = "right",
            num_arrows: int = 3,
            spotlight_size: float = 0.3,
            dim_opacity: float = 0.2,
    ):
        self.color = color
        self.width = width
        self.height = height
        self.speed_pixels_per_second = speed
        self.direction = direction.lower()
        self.num_arrows = max(1, num_arrows)
        self.stroke_width = max(2, stroke_width)
        self.spotlight_size = max(0.1, min(1.0, spotlight_size))
        self.dim_opacity = max(0.0, min(1.0, dim_opacity))

    def generate_svg(self) -> str:
        clip_bounds = self._get_clip_bounds()
        arrow_elements = self._generate_arrow_elements()
        gradient_defs = self._generate_gradient_defs()

        return f"""
        <svg width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <clipPath id="arrowClip">
              <rect x="{clip_bounds['x']}" y="{clip_bounds['y']}" width="{clip_bounds['width']}" height="{clip_bounds['height']}"/>
            </clipPath>
        
            {gradient_defs}
          </defs>
        
          <style>
            .arrow {{
              stroke: url(#spotlightGradient);
              stroke-width: {self.stroke_width};
              stroke-linecap: round;
              stroke-linejoin: round;
              fill: none;
            }}
          </style>
        
          <g clip-path="url(#arrowClip)">
            {arrow_elements}
          </g>
        </svg>
        """

    def _generate_gradient_defs(self) -> str:
        duration = self._calculate_animation_duration()

        if self.direction in ["up", "down"]:
            gradient_attrs = f'x1="0" y1="0" x2="0" y2="{self.height}" gradientUnits="userSpaceOnUse"'
            if self.direction == "down":
                from_transform = f"0 -{self.height}"
                to_transform = f"0 {self.height * 2}"
            else:  # up
                from_transform = f"0 {self.height * 2}"
                to_transform = f"0 -{self.height}"
        else:
            gradient_attrs = f'x1="0" y1="0" x2="{self.width}" y2="0" gradientUnits="userSpaceOnUse"'
            if self.direction == "right":
                from_transform = f"-{self.width} 0"
                to_transform = f"{self.width * 2} 0"
            else:  # left
                from_transform = f"{self.width * 2} 0"
                to_transform = f"-{self.width} 0"

        spotlight_percent = self.spotlight_size * 100
        dim_before = (100 - spotlight_percent) / 2
        dim_after = dim_before + spotlight_percent

        return f"""
        <linearGradient id="spotlightGradient" {gradient_attrs}>
          <animateTransform
            attributeName="gradientTransform"
            type="translate"
            from="{from_transform}"
            to="{to_transform}"
            dur="{duration:.2f}s"
            repeatCount="indefinite"/>
          <stop offset="0%" stop-color="{self.color}" stop-opacity="{self.dim_opacity}"/>
          <stop offset="{dim_before:.1f}%" stop-color="{self.color}" stop-opacity="{self.dim_opacity}"/>
          <stop offset="50%" stop-color="{self.color}" stop-opacity="1"/>
          <stop offset="{dim_after:.1f}%" stop-color="{self.color}" stop-opacity="{self.dim_opacity}"/>
          <stop offset="100%" stop-color="{self.color}" stop-opacity="{self.dim_opacity}"/>
        </linearGradient>
        """

    def _generate_arrow_elements(self) -> str:
        elements = []

        spacing = self._calculate_arrow_spacing()

        for i in range(self.num_arrows):
            position = self._calculate_arrow_position(i, spacing)
            elements.append(f'    <g class="arrow">\n      <polyline points="{position}"/>\n    </g>')

        return "\n    \n".join(elements)

    def _calculate_arrow_spacing(self) -> int:
        if self.direction in ["up", "down"]:
            available_space = self.height - 2 * (self.height // 5)
            return available_space // (self.num_arrows + 1)
        else:
            available_space = self.width - 2 * (self.width // 5)
            return available_space // (self.num_arrows + 1)

    def _calculate_arrow_position(self, index: int, spacing: int) -> str:
        base_points = self._get_arrow_points()

        if self.direction in ["up", "down"]:
            margin = self.height // 5
            offset_y = margin + (index + 1) * spacing - self.height // 2
            return self._offset_points(base_points, 0, offset_y)
        else:
            margin = self.width // 5
            offset_x = margin + (index + 1) * spacing - self.width // 2
            return self._offset_points(base_points, offset_x, 0)

    def _offset_points(self, points: str, offset_x: int, offset_y: int) -> str:
        point_pairs = points.split()
        offset_pairs = []

        for pair in point_pairs:
            x, y = map(int, pair.split(','))
            offset_pairs.append(f"{x + offset_x},{y + offset_y}")

        return " ".join(offset_pairs)

    def _get_clip_bounds(self) -> dict[str, int]:
        if self.direction in ["up", "down"]:
            margin_y = self.height // 5
            return {
                "x": 0,
                "y": margin_y,
                "width": self.width,
                "height": self.height - 2 * margin_y
            }
        else:
            margin_x = self.width // 5
            return {
                "x": margin_x,
                "y": 0,
                "width": self.width - 2 * margin_x,
                "height": self.height
            }

    def _get_arrow_points(self) -> str:
        center_x = self.width // 2
        center_y = self.height // 2
        offset_x = self.width // 8
        offset_y = self.height // 8

        if self.direction == "down":
            return f"{center_x - offset_x},{center_y - offset_y // 2} {center_x},{center_y + offset_y // 2} {center_x + offset_x},{center_y - offset_y // 2}"
        elif self.direction == "up":
            return f"{center_x - offset_x},{center_y + offset_y // 2} {center_x},{center_y - offset_y // 2} {center_x + offset_x},{center_y + offset_y // 2}"
        elif self.direction == "right":
            return f"{center_x - offset_x // 2},{center_y - offset_y} {center_x + offset_x // 2},{center_y} {center_x - offset_x // 2},{center_y + offset_y}"
        elif self.direction == "left":
            return f"{center_x + offset_x // 2},{center_y - offset_y} {center_x - offset_x // 2},{center_y} {center_x + offset_x // 2},{center_y + offset_y}"
        else:
            raise ValueError(f"Invalid direction: {self.direction}. Use 'up', 'down', 'left', or 'right'.")

    def _calculate_animation_duration(self) -> float:
        if self.direction in ["up", "down"]:
            total_distance = self.height
        else:
            total_distance = self.width
        return total_distance / self.speed_pixels_per_second

    def _generate_animations(self) -> str:
        distance = max(self.width, self.height)

        if self.direction == "down":
            start_transform = f"translateY(-{distance}px)"
            end_transform = f"translateY({distance}px)"
        elif self.direction == "up":
            start_transform = f"translateY({distance}px)"
            end_transform = f"translateY(-{distance}px)"
        elif self.direction == "right":
            start_transform = f"translateX(-{distance}px)"
            end_transform = f"translateX({distance}px)"
        elif self.direction == "left":
            start_transform = f"translateX({distance}px)"
            end_transform = f"translateX(-{distance}px)"

        return f"""
        @keyframes spotlight {{
          0% {{
            transform: {start_transform};
          }}
          100% {{
            transform: {end_transform};
          }}
        }}
        """

    def save_to_file(self, file_path: str) -> None:
        svg_content = self.generate_svg()
        with open(file_path, 'w') as file:
            file.write(svg_content)


if __name__ == "__main__":
    generator = SpotlightFlowArrowGenerator()

    print("Generated SVG for green downward arrows with moving spotlight:")
    print(generator.generate_svg())

    generator.save_to_file("_tmp/spotlight_arrows.svg")

    configurations = [
        {"direction": "right", "color": "#3b82f6", "num_arrows": 2, "width": 200, "height": 80,
         "speed_pixels_per_second": 100.0, "spotlight_size": 0.3},
        {"direction": "up", "color": "#ef4444", "num_arrows": 3, "width": 100, "height": 150,
         "speed_pixels_per_second": 60.0, "spotlight_size": 0.5, "dim_opacity": 0.1},
        {"direction": "left", "color": "#10b981", "num_arrows": 4, "width": 180, "height": 60,
         "speed_pixels_per_second": 120.0, "spotlight_size": 0.25},
    ]

    for config in configurations:
        gen = SpotlightFlowArrowGenerator(**config)
        file = f"_tmp/spotlight_arrow_{config['num_arrows']}.svg"
        gen.save_to_file(file)
        print(f"Created {file} - {config['direction']} spotlight with {config['num_arrows']} arrows")
