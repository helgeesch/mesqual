"""Base class of legends for discrete value mappings."""

from abc import abstractmethod
from html import escape
from typing import Any, Generic

from mesqual.typevars import DiscreteMappingType
from mesqual.visualizations.folium_viz_system.legends.base import BaseLegend


class DiscreteLegendBase(BaseLegend[DiscreteMappingType], Generic[DiscreteMappingType]):
    """Base class for discrete mapping legends with two-column layout.

    The legend has one row per entry of `mapping.mapping`, in its order, read
    when the map renders. A last row with the label `default_label` shows
    `default_output` when `mapping.default_used` is True.

    Args:
        mapping: Discrete value mapping that the legend describes.
        visual_column_width: Width of the visual column in px.
        column_spacing: Space between the visual and the label column in px.
        row_spacing: Space between rows in px.
        label_font_size: Label font size in px.
        default_label: Label of the row of `default_output`, as text. The
            legend escapes it and does not pass it to `value_formatter`.
        **kwargs: Passed to `BaseLegend` (title, position, value_formatter, ...).

    Examples:
        The default row appears after the mapping has used `default_output`:

        >>> from mesqual.visualizations.folium_viz_system.legends import DiscreteColorLegend
        >>> from mesqual.visualizations.value_mapping_system import DiscreteColorMapping
        >>> colors = DiscreteColorMapping(mapping={'wind': '#1f77b4'}, default_output='#cccccc')
        >>> colors('coal')
        '#cccccc'
        >>> legend = DiscreteColorLegend(colors, title='Technology', default_label='Not mapped')
        >>> 'Not mapped' in legend.render_content()
        True
    """

    def __init__(
        self,
        mapping: DiscreteMappingType,
        visual_column_width: int = 60,
        column_spacing: int = 10,
        row_spacing: int = 8,
        label_font_size: int = 12,
        default_label: str = 'Other',
        **kwargs: Any,
    ):
        super().__init__(mapping, **kwargs)
        self.visual_column_width = visual_column_width
        self.column_spacing = column_spacing
        self.row_spacing = row_spacing
        self.label_font_size = label_font_size
        self.default_label = default_label

    def additional_styles(self) -> str:
        """Return the CSS of the rows and their two columns, followed by `specific_visual_styles()`."""
        id_selector = f'#{self.get_name()}'
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

    def _item_label(self, key: Any, value: Any) -> str:
        """Return the label of a mapped item."""
        return self._format_label(key)

    def _default_item_label(self, value: Any) -> str:
        """Return the escaped `default_label`, the label of the default output."""
        return escape(str(self.default_label))

    def _render_item(self, visual: str, label: str) -> str:
        return f"""
                <div class="discrete-item">
                    <div class="visual-column">{visual}</div>
                    <div class="label-column">{label}</div>
                </div>
            """

    def render_content(self) -> str:
        """Return the HTML of the rows.

        One row per entry of `mapping.mapping`, then the row of
        `default_output` if `mapping.default_used` is True and
        `default_output` is not None.
        """
        items = [
            self._render_item(self.create_visual_element(value), self._item_label(key, value))
            for key, value in self.mapping.mapping.items()
        ]
        default = self.mapping.default_output
        if self.mapping.default_used and default is not None:
            items.append(self._render_item(self.create_visual_element(default), self._default_item_label(default)))
        return ''.join(items)
