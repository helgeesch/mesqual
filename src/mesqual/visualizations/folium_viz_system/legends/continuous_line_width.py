from typing import Any

from mesqual.visualizations.folium_viz_system.legends.base_continuous import ContinuousLegendBase
from mesqual.visualizations.value_mapping_system.continuous import SegmentedContinuousLineWidthMapping


class ContinuousLineWidthMapLegend(ContinuousLegendBase[SegmentedContinuousLineWidthMapping]):
    """Legend for continuous line width mappings.

    Args:
        mapping: Segmented continuous line width mapping.
        line_color: CSS color of the line shapes.
        show_pixel_values: If True, show the width in px below the line shape of each segment.
        **kwargs: Passed to `ContinuousLegendBase` and `BaseLegend` (title, position,
            value_formatter, ...).
    """

    def __init__(
        self,
        mapping: SegmentedContinuousLineWidthMapping,
        line_color: str = '#000000',
        show_pixel_values: bool = False,
        **kwargs: Any,
    ):
        super().__init__(mapping, **kwargs)
        self.line_color = line_color
        self.show_pixel_values = show_pixel_values

    def specific_visual_styles(self) -> str:
        id_selector = f'#{self.get_name()}'
        return f"""
            {id_selector} .pixel-value {{
                font-size: {self.tick_font_size - 2}px;
                color: {self.tick_color};
                margin-bottom: 4px;
            }}
        """

    def create_segment_visual(self, start: float, end: float, width_data: float | list[float]) -> str:
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
            label = f'{int(w0)}px' if w0 == w1 else f'{int(w0)}-{int(w1)}px'
            svg += f'<div class="pixel-value">{label}</div>'

        return svg
