"""Base class of legends for continuous value mappings."""

from abc import abstractmethod
from typing import Any, Generic

import numpy as np

from mesqual.typevars import ContinuousMappingType
from mesqual.visualizations.folium_viz_system.legends.base import BaseLegend


class ContinuousLegendBase(Generic[ContinuousMappingType], BaseLegend[ContinuousMappingType]):
    """Base class for continuous mapping legends with column-based segments.

    Args:
        mapping: Continuous value mapping that the legend describes.
        segment_spacing: Space between segments in px.
        tick_font_size: Tick label font size in px.
        tick_color: CSS color of the tick labels. Default is the title color.
        n_ticks_per_segment: Number of ticks per segment.
        segment_height: Height of the segment visual in px.
        merge_adjacent_ticks: If True, the shared boundary of two segments gets one tick.
        **kwargs: Passed to `BaseLegend` (title, position, value_formatter, ...).
    """

    def __init__(
        self,
        mapping: ContinuousMappingType,
        segment_spacing: int = 0,
        tick_font_size: int = 12,
        tick_color: str | None = None,
        n_ticks_per_segment: int = 2,
        segment_height: int = 30,
        merge_adjacent_ticks: bool = True,
        **kwargs: Any,
    ):
        super().__init__(mapping, **kwargs)
        self.segment_spacing = segment_spacing
        self.tick_font_size = tick_font_size
        self.tick_color = tick_color or self.title_color
        self.n_ticks_per_segment = n_ticks_per_segment
        self.segment_height = segment_height
        self.merge_adjacent_ticks = merge_adjacent_ticks

    def additional_styles(self) -> str:
        """Return the CSS of the segment columns and ticks, followed by `specific_visual_styles()`."""
        id_selector = f'#{self.get_name()}'
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
        """Return the HTML of the segments, sorted by their bounds.

        Each segment has the visual of `create_segment_visual` and
        `n_ticks_per_segment` ticks, spaced evenly from its start to its end.
        With `merge_adjacent_ticks`, each segment after the first leaves out
        its first tick.
        """
        segments_html: list[str] = []
        sorted_segments = sorted(self.mapping.segments.items())

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
                        transform = 'translateX(0)'  # pin left edge
                    elif i == len(raw_ticks) - 1:
                        transform = 'translateX(-100%)'  # pin right edge
                    else:
                        transform = 'translateX(-50%)'  # center
                else:
                    # on merge, always center
                    transform = 'translateX(-50%)'

                tick_divs.append(
                    f'<div class="tick" style="left:{pct:.2f}%; transform:{transform};">'
                    f'{self._format_label(float(t))}</div>'
                )

            ticks_html = f'<div class="ticks-container">{"".join(tick_divs)}</div>'

            # 4) assemble segment column
            segments_html.append(f'<div class="segment-column">{visual_html}{ticks_html}</div>')

        return f'<div class="segments-container">{"".join(segments_html)}</div>'

    @abstractmethod
    def specific_visual_styles(self) -> str:
        """Return CSS styles specific to the visual representation"""
        pass

    @abstractmethod
    def create_segment_visual(self, start: float, end: float, values: any) -> str:
        """Create the visual representation for a segment"""
        pass
