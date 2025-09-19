from mescal.utils.plotly_utils.plotly_theme import ConstantsIterable, PlotlyTheme


class PrimaryColors(ConstantsIterable):
    sky_blue = '#0ea5e9'
    teal = '#14b8a6'
    amber = '#f59e0b'
    violet = '#8b5cf6'
    rose = '#dc2626'
    green = '#22c55e'


class SequentialColors(ConstantsIterable):
    sky_blue = ['#bae6fd', '#7dd3fc', '#38bdf8', '#0ea5e9', '#0284c7', '#0369a1', '#075985']
    teal = ['#99f6e4', '#5eead4', '#2dd4bf', '#14b8a6', '#0d9488', '#0f766e', '#115e59']
    amber = ['#fde68a', '#fcd34d', '#fbbf24', '#f59e0b', '#d97706', '#b45309', '#92400e']
    violet = ['#ddd6fe', '#c4b5fd', '#a78bfa', '#8b5cf6', '#7c3aed', '#6d28d9', '#5b21b6']
    rose = ['#fecaca', '#f87171', '#ef4444', '#dc2626', '#b91c1c', '#991b1b', '#7f1d1d']
    green = ['#bbf7d0', '#86efac', '#4ade80', '#22c55e', '#16a34a', '#15803d', '#166534']
    default = sky_blue


class DarkGreyColors(ConstantsIterable):
    background = '#111827'
    gridlines = '#374151'
    borders = '#4b5563'
    text = '#d1d5db'


class LightGreyColors(ConstantsIterable):
    background = '#f8f9fa'
    paper = '#ffffff'
    gridlines = '#e9ecef'
    borders = '#ced4da'
    text = '#343a40'


class DivergingColors(ConstantsIterable):
    teal_amber = SequentialColors.teal[::-1] + SequentialColors.amber
    violet_green = SequentialColors.violet[::-1] + SequentialColors.green
    default = teal_amber


class CyclicalColors(ConstantsIterable):
    default = None


class QualitativeColors(ConstantsIterable):
    default = [
        PrimaryColors.sky_blue,
        PrimaryColors.teal,
        PrimaryColors.amber,
        PrimaryColors.violet,
        PrimaryColors.green,
        PrimaryColors.rose,
    ]


class ColorPalette:
    primary = PrimaryColors
    greys_light = LightGreyColors
    greys_dark = DarkGreyColors
    sequential = SequentialColors
    diverging = DivergingColors
    cyclical = CyclicalColors
    qualitative = QualitativeColors


colors = ColorPalette


plotly_theme_dark = PlotlyTheme(
    default_colorway=QualitativeColors.default,
    font=dict(family='Inter, sans-serif', size=12, color=DarkGreyColors.text),
    paper_color=DarkGreyColors.background,
    background_color=DarkGreyColors.background,
    xaxis=dict(
        gridcolor=DarkGreyColors.gridlines,
        linecolor=DarkGreyColors.borders,
        zerolinecolor=DarkGreyColors.borders
    ),
    yaxis=dict(
        gridcolor=DarkGreyColors.gridlines,
        linecolor=DarkGreyColors.borders,
        zerolinecolor=DarkGreyColors.borders
    ),
    legend=dict(
        bgcolor='rgba(0,0,0,0)', # Transparent background
    )
)

plotly_theme_light = PlotlyTheme(
    default_colorway=QualitativeColors.default,
    font=dict(family='Arial, sans-serif', size=12, color=LightGreyColors.text),
    paper_color=LightGreyColors.paper,
    background_color=LightGreyColors.background,
    xaxis=dict(
        gridcolor=LightGreyColors.gridlines,
        linecolor=LightGreyColors.borders,
        zerolinecolor=LightGreyColors.borders
    ),
    yaxis=dict(
        gridcolor=LightGreyColors.gridlines,
        linecolor=LightGreyColors.borders,
        zerolinecolor=LightGreyColors.borders
    ),
    legend=dict(
        bgcolor='rgba(255,255,255,0.6)', # Semi-transparent white
    )
)
