from typing import Any

from mesqual.visualizations.folium_viz_system.legends.base_discrete import DiscreteLegendBase
from mesqual.visualizations.value_mapping_system.discrete import DiscreteColorMapping


class DiscreteColorLegend(DiscreteLegendBase[DiscreteColorMapping]):
    """Legend for discrete color mappings.

    Args:
        mapping: Discrete color mapping.
        swatch_size: Size of the color swatch in px.
        **kwargs: Passed to `DiscreteLegendBase` (default_label, title, position,
            value_formatter, ...).
    """

    def __init__(self, mapping: DiscreteColorMapping, swatch_size: int = 20, **kwargs: Any):
        super().__init__(mapping, **kwargs)
        self.swatch_size = swatch_size

    def specific_visual_styles(self) -> str:
        id_selector = f'#{self.get_name()}'
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
