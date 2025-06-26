from typing import Literal


class MovingFlowArrowGenerator:
    def __init__(
            self,
            color: str = "#2563eb",
            stroke_width: int = 15,
            width: int = 100,
            height: int = 100,
            speed: float = 20.0,  # pixels_per_second
            direction: Literal["right", "left", "up", "down"] = "right",
            num_arrows: int = 4,
            animation: Literal["linear", "ease", "ease-in", "ease-out", "ease-in-out"] = "ease-in-out",
    ):
        self.color = color
        self.width = width
        self.height = height
        self.speed = speed
        self.direction = direction.lower()
        self.num_arrows = max(1, num_arrows)
        self.stroke_width = max(2, stroke_width)
        self.animation = animation

    def generate_svg(self) -> str:
        clip_bounds = self._get_clip_bounds()
        animations = self._generate_animations()
        arrow_elements = self._generate_arrow_elements()

        return f"""
        <svg width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <clipPath id="arrowClip">
              <rect x="{clip_bounds['x']}" y="{clip_bounds['y']}" width="{clip_bounds['width']}" height="{clip_bounds['height']}"/>
            </clipPath>
          </defs>
        
          <style>
            .arrow {{
              stroke: {self.color};
              stroke-width: {self.stroke_width};
              stroke-linecap: round;
              stroke-linejoin: round;
              fill: none;
            }}
        
        {self._generate_arrow_classes()}
        
            {animations}
          </style>
        
          <g clip-path="url(#arrowClip)">
        {arrow_elements}
          </g>
        </svg>
        """

    def _generate_arrow_classes(self) -> str:
        duration = self._calculate_animation_duration()
        classes = []
        for i in range(1, self.num_arrows + 1):
            classes.append(
                f"    .arrow{i} {{\n      animation: flow{i} {duration:.2f}s {self.animation} infinite;\n    }}")
        return "\n    \n".join(classes)

    def _generate_arrow_elements(self) -> str:
        arrow_points = self._get_arrow_points()
        duration = self._calculate_animation_duration()
        elements = []

        for i in range(1, self.num_arrows + 1):
            if i == 1:
                delay = ""
            else:
                delay_value = -((i - 1) * duration / self.num_arrows)
                delay = f' style="animation-delay: {delay_value:.2f}s;"'

            elements.append(
                f'    <g class="arrow arrow{i}"{delay}>\n      <polyline points="{arrow_points}"/>\n    </g>')

        return "\n    \n".join(elements)

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
        offset_x = self.width // 4
        offset_y = self.height // 4

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

    def _get_transform_distance(self) -> int:
        if self.direction in ["up", "down"]:
            return self.height // 2
        else:
            return self.width // 2

    def _calculate_animation_duration(self) -> float:
        transform_distance = self._get_transform_distance()
        total_distance = 2 * transform_distance
        return total_distance / self.speed

    def _generate_animations(self) -> str:
        distance = self._get_transform_distance()

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
        else:
            raise ValueError(f"Direction {self.direction} not accepted")

        animation_template = """
        @keyframes {animation_name} {{
          0% {{
            transform: {start_transform};
            opacity: 0;
          }}
          20% {{
            opacity: 1;
          }}
          80% {{
            opacity: 1;
          }}
          100% {{
            transform: {end_transform};
            opacity: 0;
          }}
        }}
        """

        animations = []
        for i in range(1, self.num_arrows + 1):
            animations.append(animation_template.format(
                animation_name=f"flow{i}",
                start_transform=start_transform,
                end_transform=end_transform
            ))

        return "\n    ".join(animations)

    def save_to_file(self, filename: str) -> None:
        svg_content = self.generate_svg()
        with open(filename, 'w') as file:
            file.write(svg_content)


if __name__ == "__main__":
    generator = MovingFlowArrowGenerator()

    print("Generated SVG for green right-pointing arrow with 5 arrows (200x100):")
    print(generator.generate_svg())

    print(f"\nSaving red upward arrow with 2 arrows (100x100) to file...")
    generator.save_to_file("_tmp/arrow_default.svg")

    configurations = [
        {"direction": "down", "color": "#3b82f6", "num_arrows": 3, "width": 60, "height": 120, "animation": "linear"},
        {"direction": "up", "color": "#ef4444", "num_arrows": 4, "width": 80, "height": 80, "animation": "ease-in"},
        {"direction": "left", "color": "#10b981", "num_arrows": 2, "stroke_width": 15, "width": 150, "height": 60},
        {"direction": "right", "color": "#f59e0b", "num_arrows": 6, "stroke_width": 5, "speed": 15.0, "width": 120, "height": 40}
    ]

    for config in configurations:
        gen = MovingFlowArrowGenerator(**config)
        file = f"_tmp/arrow_{config['num_arrows']}_{config['direction']}.svg"
        gen.save_to_file(file)
        print(f"Created {file} with {config['num_arrows']} arrows ({config['width']}x{config['height']})")