from typing import Any

from mesqual.visualizations.folium_viz_system.legends.base_discrete import DiscreteLegendBase
from mesqual.visualizations.value_mapping_system.discrete import DiscreteLineWidthMapping


class DiscreteLineWidthLegend(DiscreteLegendBase[DiscreteLineWidthMapping]):
    """Legend for discrete line width mappings.

    Args:
        mapping: Discrete line width mapping.
        line_color: CSS color of the lines.
        show_pixel_values: If True, show the width in px behind each label.
        **kwargs: Passed to `DiscreteLegendBase` (default_label, title, position,
            value_formatter, ...).
    """

    def __init__(
        self,
        mapping: DiscreteLineWidthMapping,
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

    def _item_label(self, key: Any, value: float) -> str:
        return self._with_pixel_value(self._format_label(key), value)

    def _default_item_label(self, value: float) -> str:
        return self._with_pixel_value(super()._default_item_label(value), value)

    def _with_pixel_value(self, label: str, width: float) -> str:
        if self.show_pixel_values:
            label += f'<span class="pixel-value">({width}px)</span>'
        return label
