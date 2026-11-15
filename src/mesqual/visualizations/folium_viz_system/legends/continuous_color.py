from typing import Any

from mesqual.visualizations.folium_viz_system.legends.base_continuous import ContinuousLegendBase
from mesqual.visualizations.value_mapping_system.continuous import SegmentedContinuousColorscale


class ContinuousColorscaleLegend(ContinuousLegendBase[SegmentedContinuousColorscale]):
    """Legend for continuous color mappings.

    Args:
        mapping: Segmented continuous color scale.
        **kwargs: Passed to `ContinuousLegendBase` and `BaseLegend` (title, position,
            n_ticks_per_segment, value_formatter, ...).
    """

    def specific_visual_styles(self) -> str:
        # no extra CSS beyond base
        return ''

    def create_segment_visual(self, start: float, end: float, colors: Any) -> str:
        if isinstance(colors, list):
            if len(colors) == 1:
                gradient = f'background: {colors[0]};'
            else:
                stops = [f'{color} {100 * i / (len(colors) - 1):.1f}%' for i, color in enumerate(colors)]
                gradient = f'background: linear-gradient(to right, {", ".join(stops)});'
        else:
            gradient = f'background: {colors};'
        return f'<div class="segment-visual" style="{gradient}"></div>'
