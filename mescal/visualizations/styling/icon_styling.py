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


class BasicArrowIconMap(IconMap):
    """Simple arrow icon with fixed size, color and configurable angle."""

    def __init__(self, size: float = 20.0, color: str = '#FF0000'):
        self.size = size
        self.color = color

    def __call__(self, angle: float = 0.0, *args, **kwargs) -> folium.DivIcon:
        """Create arrow icon pointing in specified direction.

        Args:
            angle: Direction in degrees (0° = North, 90° = East, etc.)
        """
        svg_html = self._create_arrow_svg(angle)

        return folium.DivIcon(
            html=svg_html,
            icon_size=(self.size, self.size),
            icon_anchor=(self.size / 2, self.size / 2)
        )

    def _create_arrow_svg(self, angle_degrees: float) -> str:
        """Create SVG arrow pointing in specified direction."""
        center = self.size / 2

        # Convert angle to radians (SVG 0° is east, we want 0° to be north)
        angle_rad = math.radians(angle_degrees - 90)

        # Arrow dimensions
        arrow_length = self.size * 0.4
        arrow_width = self.size * 0.2

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
            <svg width="{self.size}" height="{self.size}" xmlns="http://www.w3.org/2000/svg">
                <line x1="{start_x}" y1="{start_y}" x2="{end_x}" y2="{end_y}" 
                      stroke="{self.color}" stroke-width="2" stroke-linecap="round"/>
                <polygon points="{end_x},{end_y} {head1_x},{head1_y} {head2_x},{head2_y}" 
                         fill="{self.color}"/>
            </svg>
        '''
