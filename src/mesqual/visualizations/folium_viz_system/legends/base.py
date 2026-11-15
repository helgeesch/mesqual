"""Base class of all legends of value mappings."""

from abc import ABC, abstractmethod
from html import escape
from typing import Any, Callable, Generic

from branca.element import MacroElement, Template

from mesqual.typevars import ValueMappingType


class BaseLegend(MacroElement, ABC, Generic[ValueMappingType]):
    """Base class for all mapping legends with common functionality.

    The title and the labels are text: the legend HTML-escapes them, so a value
    such as `A<B` shows as written.

    Args:
        mapping: Value mapping that the legend describes.
        title: Legend title, as text.
        background_color: CSS background color of the legend box.
        title_color: CSS color of the title.
        title_font_size: Title font size in px.
        title_margin_bottom: Space below the title in px.
        font_family: CSS font family.
        width: Width of the legend content in px.
        padding: Padding of the legend box in px.
        border_radius: Corner radius of the legend box in px.
        position: CSS position of the legend, for example `{"bottom": 20, "right": 20}`.
        value_formatter: Function that returns the label text of every tick of a
            continuous legend and of every key of a discrete legend. Continuous
            legends pass the tick value as a float. Discrete legends pass the
            mapping key. The legend escapes the returned text. `None` keeps the
            default labels: integers and whole floats without decimals, other
            floats with up to 2 decimals, other values with `str`. The label of
            the default output of a discrete legend (`default_label`) is not
            formatted.

    Examples:
        Show values of a scale with the suffixes `B` and `M`:

        >>> from mesqual.visualizations.folium_viz_system.legends import ContinuousColorscaleLegend
        >>> from mesqual.visualizations.value_mapping_system import SegmentedContinuousColorscale
        >>> def to_suffix(value: float) -> str:
        ...     if abs(value) >= 1e9:
        ...         return f"{value / 1e9:g}B"
        ...     if abs(value) >= 1e6:
        ...         return f"{value / 1e6:g}M"
        ...     return f"{value:g}"
        >>> mapping = SegmentedContinuousColorscale(segments={(0, 2e9): ["#0000ff", "#ff0000"]})
        >>> legend = ContinuousColorscaleLegend(mapping, value_formatter=to_suffix)
        >>> "2B" in legend.render_content() and "2000000000" not in legend.render_content()
        True
    """

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
            <div class="legend-title">{{ this.title|e }}</div>
            {% endif %}
            <div class="legend-content">
                {{ this.render_content() }}
            </div>
        </div>
    {% endmacro %}
    """)

    def __init__(
        self,
        mapping: ValueMappingType,
        title: str = '',
        background_color: str = 'white',
        title_color: str = '#333333',
        title_font_size: int = 14,
        title_margin_bottom: int = 8,
        font_family: str = 'Arial, sans-serif',
        width: int = 200,
        padding: int = 10,
        border_radius: int = 5,
        position: dict[str, int | float | str] | None = None,
        value_formatter: Callable[[Any], str] | None = None,
    ):
        super().__init__()
        self._name = f'{self.__class__.__name__}_{id(self)}'
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
        self.position = self._process_position(position or {'bottom': '20px', 'right': '20px'})

        self.value_formatter = value_formatter

        self._template = self._base_template

    def __call__(self, value) -> Any:
        return self.mapping(value)

    def _process_position(self, position: dict[str, int | float | str]) -> dict[str, str]:
        return {k: f'{v}px' if isinstance(v, (int, float)) else str(v) for k, v in position.items()}

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
                return f'{int(value)}'
            return f'{value:.2f}'.rstrip('0').rstrip('.')
        return str(value)

    def _format_label(self, value: float | int | str) -> str:
        """Return the escaped label of a tick or key, using `value_formatter` if set."""
        formatter = self.value_formatter or self._format_value
        return escape(str(formatter(value)))
