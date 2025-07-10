import math
from abc import ABC, abstractmethod

import folium


class IconMap(ABC):
    """Abstract base class for icon mapping functions."""

    @abstractmethod
    def __call__(self, *args, **kwargs) -> folium.DivIcon:
        """Map values to a folium DivIcon.

        Returns:
            folium.DivIcon object ready to use in markers
        """
        pass


class BasicCircleIconMap(IconMap):
    """Simple circle icon with fixed diameter and color."""

    def __init__(self, diameter: float = 20.0, color: str = '#FF0000'):
        self.diameter = diameter
        self.color = color

    def __call__(self, *args, **kwargs) -> folium.DivIcon:
        """Create circle icon."""
        radius = self.diameter / 2
        svg_html = f'''
            <svg width="{self.diameter}" height="{self.diameter}" xmlns="http://www.w3.org/2000/svg">
                <circle cx="{radius}" cy="{radius}" r="{radius - 1}" 
                        fill="{self.color}" stroke="white" stroke-width="1"/>
            </svg>
        '''

        return folium.DivIcon(
            html=svg_html,
            icon_size=(self.diameter, self.diameter),
            icon_anchor=(radius, radius)
        )


class ArrowIconMapBase(IconMap, ABC):

    @abstractmethod
    def __call__(self, angle: float, *args, **kwargs):
        pass


class BasicArrowIconMap(ArrowIconMapBase):
    """Simple arrow icon with fixed size, color and configurable angle."""

    def __call__(self, angle: float, size: float = 20.0, color: str = '#FF0000', *args, **kwargs) -> folium.DivIcon:
        """Create arrow icon pointing in specified direction.

        Args:
            angle: Direction in degrees (0° = North, 90° = East, etc.)
        """
        svg_html = self._create_arrow_svg(angle)

        return folium.DivIcon(
            html=svg_html,
            icon_size=(size, size),
            icon_anchor=(size / 2, size / 2)
        )

    @staticmethod
    def _create_arrow_svg(angle_degrees: float, size: float, color: str) -> str:
        """Create SVG arrow pointing in specified direction."""
        center = size / 2

        # Convert angle to radians (SVG 0° is east, we want 0° to be north)
        angle_rad = math.radians(angle_degrees - 90)

        # Arrow dimensions
        arrow_length = size * 0.4
        arrow_width = size * 0.2

        # Main arrow line start and end
        start_x = center - arrow_length * math.cos(angle_rad) / 2
        start_y = center - arrow_length * math.sin(angle_rad) / 2
        end_x = center + arrow_length * math.cos(angle_rad) / 2
        end_y = center + arrow_length * math.sin(angle_rad) / 2

        # Arrow head points
        head_angle1 = angle_rad + math.pi * 0.75
        head_angle2 = angle_rad - math.pi * 0.75
        head_length = arrow_width

        head1_x = end_x + head_length * math.cos(head_angle1)
        head1_y = end_y + head_length * math.sin(head_angle1)
        head2_x = end_x + head_length * math.cos(head_angle2)
        head2_y = end_y + head_length * math.sin(head_angle2)

        return f'''
            <svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">
                <line x1="{start_x}" y1="{start_y}" x2="{end_x}" y2="{end_y}" 
                      stroke="{color}" stroke-width="2" stroke-linecap="round"/>
                <polygon points="{end_x},{end_y} {head1_x},{head1_y} {head2_x},{head2_y}" 
                         fill="{color}"/>
            </svg>
        '''


class BasicAnimatedArrowIconMap(ArrowIconMapBase):
    def __call__(self, angle: float, size: float = 10, *args, **kwargs) -> folium.DivIcon:
        from captain_arro import MovingFlowArrowGenerator
        arrow_generator = MovingFlowArrowGenerator()
        svg = arrow_generator.generate_svg()
        width = arrow_generator.width
        height = arrow_generator.height

        icon_html = f"""
        <div style="
            position: relative;
            width: {width}px;
            height: {height}px;
            transform: translate(-50%, -50%) rotate({round((angle-90) % 360, 2)}deg);
            transform-origin: center center;
        ">
            {svg}
        </div>
        """

        return folium.DivIcon(
            html=icon_html,
            icon_size=(size, size),
            icon_anchor=(size / 2, size / 2)
        )
