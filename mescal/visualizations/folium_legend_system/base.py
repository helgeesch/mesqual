from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from branca.element import MacroElement, Template

from mescal.visualizations.value_mapping_system import (
    BaseMapping,
    DiscreteInputMapping,
    SegmentedContinuousInputMappingBase,
    SegmentedContinuousColorscale,
    SegmentedContinuousLineWidthMapping,
    DiscreteColorMapping,
    DiscreteLineDashPatternMapping,
    DiscreteLineWidthMapping,
    DiscreteIconMap
)


class BaseLegend(MacroElement, ABC):
    """Base class for all mapping legends with common functionality"""

    _base_template = Template("""
    {% macro header(this, kwargs) %}
        <style>
            #{{ this.get_name() }} {
                position: fixed;
                z-index: 1000;
                background: {{ this.background_color }};
                padding: {{ this.padding }}px;
                border-radius: {{ this.border_radius }}px;
                box-shadow: 0 0 5px rgba(0,0,0,0.2);
                font-family: {{ this.font_family }};
                {% for key, value in this.position.items() %}
                {{ key }}: {{ value }};
                {% endfor %}
            }
            #{{ this.get_name() }} .legend-title {
                margin-bottom: {{ this.title_margin_bottom }}px;
                font-size: {{ this.title_font_size }}px;
                font-weight: bold;
                color: {{ this.title_color }};
            }
            #{{ this.get_name() }} .legend-content {
                position: relative;
                width: {{ this.width }}px;
            }
            {{ this.additional_styles() }}
        </style>
    {% endmacro %}

    {% macro html(this, kwargs) %}
        <div id="{{ this.get_name() }}">
            {% if this.title %}
            <div class="legend-title">{{ this.title }}</div>
            {% endif %}
            <div class="legend-content">
                {{ this.render_content() }}
            </div>
        </div>
    {% endmacro %}
    """)

    def __init__(
            self,
            mapping: BaseMapping,
            title: str = "",
            background_color: str = "white",
            title_color: str = "#333333",
            title_font_size: int = 14,
            title_margin_bottom: int = 8,
            font_family: str = "Arial, sans-serif",
            width: int = 200,
            padding: int = 10,
            border_radius: int = 5,
            position: dict[str, int | float | str] | None = None
    ):
        super().__init__()
        self._name = f"{self.__class__.__name__}_{id(self)}"
        self.mapping = mapping
        self.title = title
        self.background_color = background_color
        self.title_color = title_color
        self.title_font_size = title_font_size
        self.title_margin_bottom = title_margin_bottom
        self.font_family = font_family
        self.width = width
        self.padding = padding
        self.border_radius = border_radius
        self.position = self._process_position(position or {"bottom": "20px", "right": "20px"})

        self._template = self._base_template

    def _process_position(self, position: dict[str, int | float | str]) -> dict[str, str]:
        return {k: f"{v}px" if isinstance(v, (int, float)) else str(v)
                for k, v in position.items()}

    @abstractmethod
    def additional_styles(self) -> str:
        """Return additional CSS styles specific to this legend type"""
        pass

    @abstractmethod
    def render_content(self) -> str:
        """Return the HTML content for the legend"""
        pass

    def _format_value(self, value: float | int | str) -> str:
        """Format a value for display"""
        if isinstance(value, (int, float)):
            if isinstance(value, float) and value % 1 == 0:
                return f"{int(value)}"
            return f"{value:.2f}".rstrip('0').rstrip('.')
        return str(value)


class DiscreteLegendBase(BaseLegend):
    """Base class for discrete mapping legends with two-column layout"""

    def __init__(
            self,
            mapping: DiscreteInputMapping,
            visual_column_width: int = 60,
            column_spacing: int = 10,
            row_spacing: int = 8,
            label_font_size: int = 12,
            **kwargs
    ):
        super().__init__(mapping, **kwargs)
        self.visual_column_width = visual_column_width
        self.column_spacing = column_spacing
        self.row_spacing = row_spacing
        self.label_font_size = label_font_size

    def additional_styles(self) -> str:
        id_selector = f"#{self.get_name()}"
        return f"""
            {id_selector} .discrete-item {{
                display: flex;
                align-items: center;
                margin-bottom: {self.row_spacing}px;
            }}
            {id_selector} .visual-column {{
                width: {self.visual_column_width}px;
                margin-right: {self.column_spacing}px;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            {id_selector} .label-column {{
                flex: 1;
                font-size: {self.label_font_size}px;
                color: {self.title_color};
            }}
            {self.specific_visual_styles()}
        """

    @abstractmethod
    def specific_visual_styles(self) -> str:
        """Return CSS styles specific to the visual representation"""
        pass

    @abstractmethod
    def create_visual_element(self, value: any) -> str:
        """Create the visual representation for a given value"""
        pass

    def render_content(self) -> str:
        items = []

        # Render mapped items
        for key, value in sorted(self.mapping.mapping.items()):
            visual = self.create_visual_element(value)
            items.append(f"""
                <div class="discrete-item">
                    <div class="visual-column">{visual}</div>
                    <div class="label-column">{self._format_value(key)}</div>
                </div>
            """)

        # Render default if exists
        if hasattr(self.mapping, '_default_output') and self.mapping._default_output is not None:
            visual = self.create_visual_element(self.mapping._default_output)
            items.append(f"""
                <div class="discrete-item">
                    <div class="visual-column">{visual}</div>
                    <div class="label-column">Other</div>
                </div>
            """)

        return ''.join(items)


class ContinuousLegendBase(BaseLegend):
    """Base class for continuous mapping legends with column-based segments"""

    def __init__(
            self,
            mapping: SegmentedContinuousInputMappingBase,
            segment_spacing: int = 0,
            tick_font_size: int = 12,
            tick_color: str | None = None,
            n_ticks_per_segment: int = 2,
            segment_height: int = 30,
            merge_adjacent_ticks: bool = True,
            **kwargs
    ):
        super().__init__(mapping, **kwargs)
        self.segment_spacing = segment_spacing
        self.tick_font_size = tick_font_size
        self.tick_color = tick_color or self.title_color
        self.n_ticks_per_segment = n_ticks_per_segment
        self.segment_height = segment_height
        self.merge_adjacent_ticks = merge_adjacent_ticks

    def additional_styles(self) -> str:
        id_selector = f"#{self.get_name()}"
        return f"""
            {id_selector} .segments-container {{
                display: flex;
                align-items: flex-start;
                gap: {self.segment_spacing}px;
            }}
            {id_selector} .segment-column {{
                flex: 1;
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            {id_selector} .segment-visual {{
                width: 100%;
                height: {self.segment_height}px;
                margin-bottom: 4px;
            }}
            {id_selector} .ticks-container {{
                position: relative;
                width: 100%;
                height: 20px;
                margin-top: 2px;
            }}
            {id_selector} .tick {{
                position: absolute;
                transform: translateX(-50%);
                font-size: {self.tick_font_size}px;
                color: {self.tick_color};
                white-space: nowrap;
            }}
            {self.specific_visual_styles()}
        """

    def render_content(self) -> str:
        segments_html: list[str] = []
        sorted_segments = sorted(self.mapping._segments.items())

        for idx, ((start, end), values) in enumerate(sorted_segments):
            # 1) visual
            visual_html = self.create_segment_visual(start, end, values)

            # 2) ticks for this segment
            raw_ticks = list(np.linspace(start, end, self.n_ticks_per_segment))
            if self.merge_adjacent_ticks and idx > 0:
                raw_ticks = raw_ticks[1:]

            # 3) build tick divs with conditional transform
            tick_divs: list[str] = []
            span = end - start
            for i, t in enumerate(raw_ticks):
                # compute percent along the box [0,100]
                pct = 0.0 if span == 0 else (t - start) / span * 100

                # choose transform so start/end snap to edges
                if not self.merge_adjacent_ticks:
                    if i == 0:
                        transform = "translateX(0)"         # pin left edge
                    elif i == len(raw_ticks) - 1:
                        transform = "translateX(-100%)"    # pin right edge
                    else:
                        transform = "translateX(-50%)"     # center
                else:
                    # on merge, always center
                    transform = "translateX(-50%)"

                tick_divs.append(
                    f'<div class="tick" style="left:{pct:.2f}%; transform:{transform};">'
                    f'{self._format_value(t)}</div>'
                )

            ticks_html = f'<div class="ticks-container">{"".join(tick_divs)}</div>'

            # 4) assemble segment column
            segments_html.append(
                f'<div class="segment-column">{visual_html}{ticks_html}</div>'
            )

        return f'<div class="segments-container">{"".join(segments_html)}</div>'

    def _render_ticks(self) -> str:
        # Gather all per-segment tick values
        all_ticks = []
        for (start, end), _ in sorted(self.mapping.segments.items()):
            all_ticks.extend(np.linspace(start, end, self.n_ticks_per_segment))

        # Optionally merge duplicates at boundaries
        if self.merge_adjacent_ticks:
            seen = set()
            ticks = []
            for t in all_ticks:
                key = round(t, 9)
                if key not in seen:
                    seen.add(key)
                    ticks.append(t)
        else:
            ticks = all_ticks

        # Position each tick across the full width
        n_segments = len(self.mapping.segments)
        seg_pct = 100.0 / n_segments
        html = []
        for t in ticks:
            # find its segment index
            for idx, ((s, e), _) in enumerate(sorted(self.mapping.segments.items())):
                if s - 1e-8 <= t <= e + 1e-8:
                    if e == s:
                        left_pct = idx * seg_pct
                    else:
                        left_pct = idx * seg_pct + ((t - s) / (e - s)) * seg_pct
                    html.append(
                        f'<div class="tick" style="left:{left_pct}%;">{self._format_value(t)}</div>'
                    )
                    break
        return ''.join(html)

    @abstractmethod
    def specific_visual_styles(self) -> str:
        """Return CSS styles specific to the visual representation"""
        pass

    @abstractmethod
    def create_segment_visual(self, start: float, end: float, values: any) -> str:
        """Create the visual representation for a segment"""
        pass


class DiscreteColorLegend(DiscreteLegendBase):
    """Legend for discrete color mappings"""

    def __init__(
            self,
            mapping: DiscreteColorMapping,
            swatch_size: int = 20,
            **kwargs
    ):
        super().__init__(mapping, **kwargs)
        self.swatch_size = swatch_size

    def specific_visual_styles(self) -> str:
        id_selector = f"#{self.get_name()}"
        return f"""
            {id_selector} .color-swatch {{
                width: {self.swatch_size}px;
                height: {self.swatch_size}px;
                border: 1px solid #ccc;
                border-radius: 2px;
            }}
        """

    def create_visual_element(self, color: str) -> str:
        return f'<div class="color-swatch" style="background-color: {color};"></div>'


class DiscreteLineDashLegend(DiscreteLegendBase):
    """Legend for line dash pattern mappings"""

    def __init__(
            self,
            mapping: DiscreteLineDashPatternMapping,
            line_color: str = "#000000",
            line_width: int = 2,
            **kwargs
    ):
        super().__init__(mapping, **kwargs)
        self.line_color = line_color
        self.line_width = line_width

    def specific_visual_styles(self) -> str:
        id_selector = f"#{self.get_name()}"
        return f"""
            {id_selector} svg {{
                /* you can put shared SVG styling here if needed */
            }}
        """

    def create_visual_element(self, pattern: str) -> str:
        dash_array = f'stroke-dasharray="{pattern}"' if pattern else ""
        return f"""
            <svg width="{self.visual_column_width}" height="{self.line_width * 2}">
                <line x1="0" y1="{self.line_width}" x2="{self.visual_column_width}" y2="{self.line_width}"
                      stroke="{self.line_color}" stroke-width="{self.line_width}" {dash_array}/>
            </svg>
        """


class DiscreteLineWidthLegend(DiscreteLegendBase):
    """Legend for discrete line width mappings"""

    def __init__(
            self,
            mapping: DiscreteLineWidthMapping,
            line_color: str = "#000000",
            show_pixel_values: bool = True,
            **kwargs
    ):
        super().__init__(mapping, **kwargs)
        self.line_color = line_color
        self.show_pixel_values = show_pixel_values

    def specific_visual_styles(self) -> str:
        id_selector = f"#{self.get_name()}"
        return f"""
            {id_selector} .pixel-value {{
                font-size: {self.label_font_size - 2}px;
                color: {self.title_color};
                opacity: 0.7;
                margin-left: 5px;
            }}
        """

    def create_visual_element(self, width: float) -> str:
        height = max(10, width * 2)
        return f"""
            <svg width="{self.visual_column_width}" height="{height}">
                <line x1="0" y1="{height / 2}" x2="{self.visual_column_width}" y2="{height / 2}"
                      stroke="{self.line_color}" stroke-width="{width}"/>
            </svg>
        """

    def render_content(self) -> str:
        items = []

        # Override to add pixel values
        for key, width in sorted(self.mapping.mapping.items()):
            visual = self.create_visual_element(width)
            label = self._format_value(key)
            if self.show_pixel_values:
                label += f'<span class="pixel-value">({width}px)</span>'

            items.append(f"""
                <div class="discrete-item">
                    <div class="visual-column">{visual}</div>
                    <div class="label-column">{label}</div>
                </div>
            """)

        if hasattr(self.mapping, '_default_output') and self.mapping._default_output is not None:
            width = self.mapping._default_output
            visual = self.create_visual_element(width)
            label = "Other"
            if self.show_pixel_values:
                label += f'<span class="pixel-value">({width}px)</span>'

            items.append(f"""
                <div class="discrete-item">
                    <div class="visual-column">{visual}</div>
                    <div class="label-column">{label}</div>
                </div>
            """)

        return ''.join(items)


class ColorMapLegend(ContinuousLegendBase):
    """Legend for continuous color mappings"""

    def specific_visual_styles(self) -> str:
        # no extra CSS beyond base
        return ""

    def create_segment_visual(self, start: float, end: float, colors: Any) -> str:
        if isinstance(colors, list):
            if len(colors) == 1:
                gradient = f"background: {colors[0]};"
            else:
                stops = [
                    f"{color} {100 * i / (len(colors) - 1):.1f}%"
                    for i, color in enumerate(colors)
                ]
                gradient = f"background: linear-gradient(to right, {', '.join(stops)});"
        else:
            gradient = f"background: {colors};"
        return f'<div class="segment-visual" style="{gradient}"></div>'


class LineWidthMapLegend(ContinuousLegendBase):
    def __init__(
        self,
        mapping: SegmentedContinuousLineWidthMapping,
        line_color: str = "#000000",
        show_pixel_values: bool = True,
        merge_adjacent_ticks: bool = False,
        **kwargs
    ):
        super().__init__(mapping, **kwargs)
        self.line_color = line_color
        self.show_pixel_values = show_pixel_values
        self.merge_adjacent_ticks = merge_adjacent_ticks

    def specific_visual_styles(self) -> str:
        id_selector = f"#{self.get_name()}"
        return f"""
            {id_selector} .pixel-value {{
                font-size: {self.tick_font_size - 2}px;
                color: {self.tick_color};
                margin-bottom: 4px;
            }}
        """

    def create_segment_visual(
            self,
            start: float,
            end: float,
            width_data: float | list[float]
    ) -> str:
        # trapezoid SVG
        if isinstance(width_data, list):
            w0, w1 = width_data[0], width_data[-1]
        else:
            w0 = w1 = width_data

        svg = f"""
            <svg width="100%" height="{self.segment_height}" preserveAspectRatio="none">
                <path d="
                    M0,{self.segment_height / 2 - w0 / 2}
                    L100,{self.segment_height / 2 - w1 / 2}
                    L100,{self.segment_height / 2 + w1 / 2}
                    L0,{self.segment_height / 2 + w0 / 2}
                    Z
                " fill="{self.line_color}" />
            </svg>
        """
        # pixel-value above ticks
        if self.show_pixel_values:
            label = f"{int(w0)}px" if w0 == w1 else f"{int(w0)}-{int(w1)}px"
            svg += f'<div class="pixel-value">{label}</div>'

        return svg


if __name__ == '__main__':
    import folium

    # Example: Continuous color scale
    color_segments = {
        (-20, 0): ['#0000FF', '#00FFFF'],
        (0, 5): ['#00FF00'],
        (5, 10): ['#FF0000', '#FFFF00'],
        (10, 20): ['#FFFFFF']
    }
    color_mapping = SegmentedContinuousColorscale(segments=color_segments)
    color_legend = ColorMapLegend(
        mapping=color_mapping,
        title="Temperature (°C)",
        background_color="#f0f0f0",
        n_ticks_per_segment=3,
        segment_spacing=0,
        position={"bottom": 50, "left": 50},
        width=400
    )

    # Example: Line width mapping
    width_segments = {
        # (0, 10): 1.0,
        (10, 20): 3.0,
        (20, 30): [3.0, 10.0],
        (30, 50): 10.0
    }
    width_mapping = SegmentedContinuousLineWidthMapping(segments=width_segments)
    width_legend = LineWidthMapLegend(
        mapping=width_mapping,
        title="Flow Volume (MW)",
        line_color="#2c3e50",
        segment_spacing=15,
        position={"bottom": 50, "right": 50},
        merge_adjacent_ticks=False,
        show_pixel_values=False,
    )

    # Example: Discrete color mapping
    discrete_color_mapping = DiscreteColorMapping(
        mapping={"Category A": "#ff0000", "Category B": "#00ff00", "Category C": "#0000ff"},
        default_output="#cccccc"
    )
    discrete_color_legend = DiscreteColorLegend(
        mapping=discrete_color_mapping,
        title="Discrete Color Categories",
        swatch_size=50,
        position={"top": 50, "right": 50},
        background_color='#ABABAB'
    )

    # Example: Line dash mapping
    dash_mapping = DiscreteLineDashPatternMapping(
        mapping={"Type 1": "", "Type 2": "5,5", "Type 3": "10,5,5,5"}
    )
    dash_legend = DiscreteLineDashLegend(
        mapping=dash_mapping,
        title="Line Types",
        visual_column_width=80,
        position={"top": 50, "left": 50},
        width=400
    )

    # Example: Discrete line width
    width_mapping_discrete = DiscreteLineWidthMapping(
        mapping={"Small": 1.0, "Medium": 3.0, "Large": 6.0},
        default_output=2.0
    )
    width_legend_discrete = DiscreteLineWidthLegend(
        mapping=width_mapping_discrete,
        title="Line Sizes",
        show_pixel_values=True,
        position={"center": 50, "right": 50}
    )

    # Create map and add legends
    m = folium.Map(location=[48.85, 2.35], zoom_start=10)
    color_legend.add_to(m)
    width_legend.add_to(m)
    discrete_color_legend.add_to(m)
    dash_legend.add_to(m)
    width_legend_discrete.add_to(m)

    m.save("_tmp/test_value_mapping_legends.html")