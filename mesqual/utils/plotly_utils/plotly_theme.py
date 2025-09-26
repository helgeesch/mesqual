"""Custom Plotly theme and color palette definitions for consistent visualization styling.

This module provides a comprehensive theming system for Plotly figures, including
predefined color palettes, theme configuration, and template application. It enables
consistent styling across all visualizations in the MESQUAL framework.
"""

from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


class ConstantsIterable:
    """Base class for creating iterable constant collections.

    Provides dictionary-like interface methods (items, values, keys) for
    accessing class attributes as constants.
    """
    @classmethod
    def items(cls):
        """Yield (name, value) pairs for all non-private, non-callable attributes.

        Yields:
            Tuple of (attribute_name, attribute_value) for class constants.
        """
        for attr_name in dir(cls):
            if not attr_name.startswith('__') and not callable(getattr(cls, attr_name)):
                yield attr_name, getattr(cls, attr_name)

    @classmethod
    def values(cls):
        """Yield values for all non-private, non-callable attributes.

        Yields:
            Attribute values for class constants.
        """
        for attr_name in dir(cls):
            if not attr_name.startswith('__') and not callable(getattr(cls, attr_name)):
                yield getattr(cls, attr_name)

    @classmethod
    def keys(cls):
        """Yield names for all non-private, non-callable attributes.

        Yields:
            Attribute names for class constants.
        """
        for attr_name in dir(cls):
            if not attr_name.startswith('__') and not callable(getattr(cls, attr_name)):
                yield attr_name


class PrimaryColors(ConstantsIterable):
    """Primary color palette with vibrant, distinct colors.

    Provides a curated set of primary colors suitable for categorical data
    visualization and general plotting needs.
    """
    mint = '#00b894'
    cyan = '#00cec9'
    blue = '#0984e3'
    red = '#d63031'
    pink = '#e84393'
    green_light = '#badc58'
    green_bold = '#6ab04c'
    orange_light = '#fdcb6e'
    orange_bold = '#e17055'
    purple_light = '#a29bfe'
    purple_bold = '#6c5ce7'


class SequentialColors(ConstantsIterable):
    """Sequential color palettes for ordered data visualization.

    Contains multi-hue and single-hue sequential palettes appropriate for
    displaying ordered data such as numerical ranges or intensity maps.
    """
    mint_blue_red = ['#00b894', '#0984e3', '#d63031']
    blue_cyan_pink = ['#0984e3', '#00cec9', '#e84393']
    shades_of_mint = ['#e6fff7', '#55efc4', '#00b894', '#009677', '#006b54']
    shades_of_cyan = ['#e6ffff', '#8ee8e7', '#00cec9', '#00a29a', '#00756e']
    shades_of_blue = ['#e6f4ff', '#74b9ff', '#0984e3', '#0063b1', '#004680']
    shades_of_red = ['#ffe6e6', '#ff7675', '#d63031', '#b02525', '#801b1b']
    shades_of_pink = ['#ffe6f3', '#fd79a8', '#e84393', '#c13584', '#962264']
    default = mint_blue_red


class DivergingColors(ConstantsIterable):
    """Diverging color palettes for data with meaningful midpoint.

    Provides color schemes that emphasize deviations from a central value,
    suitable for correlation matrices, anomaly detection, and comparative analysis.
    """
    blue_mint = SequentialColors.shades_of_blue[::-1] + SequentialColors.shades_of_mint
    red_mint = SequentialColors.shades_of_red[::-1] + SequentialColors.shades_of_mint
    pink_blue = SequentialColors.shades_of_pink[::-1] + SequentialColors.shades_of_blue
    pink_cyan = SequentialColors.shades_of_pink[::-1] + SequentialColors.shades_of_cyan
    default = blue_mint


class CyclicalColors(ConstantsIterable):
    """Cyclical color palettes for periodic data.

    Reserved for future implementation of color schemes appropriate for
    cyclical data such as seasonal patterns or angular measurements.
    """
    default = None


class QualitativeColors(ConstantsIterable):
    """Qualitative color palettes for categorical data.

    Provides distinct, visually separable colors for categorical variables
    without inherent ordering.
    """
    default = [
        PrimaryColors.blue,
        PrimaryColors.mint,
        PrimaryColors.cyan,
        PrimaryColors.red,
        PrimaryColors.pink,
        PrimaryColors.green_light,
        PrimaryColors.green_bold,
        PrimaryColors.orange_light,
        PrimaryColors.orange_bold,
        PrimaryColors.purple_light,
        PrimaryColors.purple_bold,
    ]


class ColorPalette:
    """Central access point for all color palette categories.

    Organizes color palettes by type (primary, sequential, diverging, etc.)
    for easy access and consistent usage across the framework.
    """
    primary = PrimaryColors
    sequential = SequentialColors
    diverging = DivergingColors
    cyclical = CyclicalColors
    qualitative = QualitativeColors


colors = ColorPalette
"""Global color palette instance for convenient access to all color schemes.

Example:

    >>> colors.primary.blue
    '#0984e3'
    >>> colors.sequential.default
    ['#00b894', '#0984e3', '#d63031']
"""


@dataclass
class PlotlyTheme:
    """Configurable Plotly theme for consistent figure styling.

    This dataclass encapsulates all theme settings and provides methods to
    apply them to Plotly's global template system. It supports customization
    of colors, fonts, backgrounds, axis styling, and optional watermarking.

    Attributes:
        default_colorway: Default color sequence for traces.
        font: Font configuration dictionary.
        paper_color: Background color outside the plot area.
        background_color: Background color of the plot area.
        xaxis: X-axis styling configuration.
        yaxis: Y-axis styling configuration.
        legend: Legend styling configuration.
        watermark_text: Optional watermark text to display.
        watermark_position: (x, y) position for watermark (paper coordinates).
        watermark_opacity: Opacity level for watermark (0.0 to 1.0).

    Example:

        >>> theme = PlotlyTheme(
        ...     default_colorway=colors.qualitative.default,
        ...     watermark_text="MESQUAL"
        ... )
        >>> theme.apply()
    """
    default_colorway: list[str] = field(default_factory=list)
    font: dict = field(default_factory=dict)
    paper_color: str = '#ffffff'
    background_color: str = '#F2F2F2'
    xaxis: dict = field(default_factory=dict)
    yaxis: dict = field(default_factory=dict)
    legend: dict = field(default_factory=dict)
    watermark_text: str = None
    watermark_position: tuple[float, float] = (0.99, 0.01)
    watermark_opacity: float = 0.1

    def apply(self) -> None:
        """Apply the theme settings to Plotly's global template system.

        Creates a custom template with all specified settings and sets it as
        the default template for all subsequent figure creation. The template
        includes styling for layout, axes, legends, and optionally watermarks.

        Note:
            This method modifies Plotly's global state and affects all figures
            created after calling this method.

        Example:

            >>> theme = PlotlyTheme(watermark_text="My Project")
            >>> theme.apply()
            >>> fig = go.Figure()  # Will use the custom theme
        """
        template = go.layout.Template()

        template.layout.colorway = self.default_colorway
        template.layout.font = self.font
        template.layout.paper_bgcolor = self.paper_color
        template.layout.plot_bgcolor = self.background_color
        template.layout.xaxis.update(self.xaxis)
        template.layout.yaxis.update(self.yaxis)
        template.layout.legend.update(self.legend)

        template.layout.title = dict(x=0.5)

        template.data.bar = [
            go.Bar(marker=dict(line=dict(width=0)))
        ]

        if self.watermark_text:
            template.layout.annotations = [
                dict(
                    name='watermark',
                    text=self.watermark_text,
                    xref="paper",
                    yref="paper",
                    x=self.watermark_position[0],
                    y=self.watermark_position[1],
                    showarrow=False,
                    font=dict(size=50, color="black"),
                    opacity=self.watermark_opacity,
                    textangle=0,
                )
            ]

        pio.templates["custom"] = template
        pio.templates.default = "plotly+custom"


if __name__ == "__main__":
    theme = PlotlyTheme(watermark_text="mesqual")

    theme.apply()

    # Example plot
    x = np.linspace(0, 10, 100)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=np.sin(x), name="Sin"))
    fig.add_trace(go.Scatter(x=x, y=np.cos(x), name="Cos"))
    fig.update_layout(title='Sine and Cosine in Template Style')

    fig.show(renderer='browser')
