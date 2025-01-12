from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import folium

from mescal.enums import VisualizationTypeEnum


@dataclass
class LegendEntry:
    label: str
    value: Any
    style: dict


class MescalMapLegend(ABC):
    def __init__(self, title: str):
        self.title = title

    @abstractmethod
    def create_folium_element(self) -> folium.Element:
        pass

    @abstractmethod
    def get_style_for_value(self, value: float) -> dict:
        pass


class ContinuousColorMapLegend(MescalMapLegend):
    def __init__(self, title: str, vmin: float, vmax: float, colormap: str = "RdYlBu", num_steps: int = 8):
        super().__init__(title)
        self.vmin = vmin
        self.vmax = vmax
        self.colormap = colormap
        self.num_steps = num_steps

    def create_folium_element(self) -> folium.Element:
        colorscale = self._create_colorscale()
        return folium.ColorMap(
            colors=list(colorscale.values()),
            vmin=self.vmin,
            vmax=self.vmax,
            caption=self.title
        )

    def get_style_for_value(self, value: float) -> dict:
        return {"fillColor": self._get_color_for_value(value)}

    def _get_color_for_value(self, value: float) -> str:
        import matplotlib.pyplot as plt
        from matplotlib.colors import rgb2hex

        norm = plt.Normalize(vmin=self.vmin, vmax=self.vmax)
        cmap = plt.get_cmap(self.colormap)
        return rgb2hex(cmap(norm(value)))

    def _create_colorscale(self) -> Dict[float, str]:
        values = np.linspace(self.vmin, self.vmax, self.num_steps)
        return {val: self._get_color_for_value(val) for val in values}


class DiscreteColorMapLegend(MescalMapLegend):
    def __init__(self, title: str, entries: List[LegendEntry]):
        super().__init__(title)
        self.entries = entries

    def create_folium_element(self) -> folium.Element:
        # Create a custom HTML legend for discrete colors
        pass

    def get_style_for_value(self, value: float) -> dict:
        for entry in self.entries:
            if entry.value == value:
                return entry.style
        return {}


class SymbolMapLegend(MescalMapLegend):
    def __init__(self, title: str, entries: List[LegendEntry]):
        super().__init__(title)
        self.entries = entries

    def create_folium_element(self) -> folium.Element:
        # Create a custom HTML legend for symbols
        pass

    def get_style_for_value(self, value: float) -> dict:
        for entry in self.entries:
            if entry.value == value:
                return entry.style
        return {}


class LegendFactory:
    @staticmethod
    def create_legend(
            title: str,
            visualization_type: VisualizationTypeEnum,
            values: List[float],
            is_discrete: bool = False
    ) -> MescalMapLegend:
        if visualization_type == VisualizationTypeEnum.Border:
            # Create symbol legend for border visualizations
            entries = [
                LegendEntry("Positive", 1, {"icon": "plus", "color": "green"}),
                LegendEntry("Negative", -1, {"icon": "minus", "color": "red"}),
            ]
            return SymbolMapLegend(title, entries)

        elif is_discrete:
            unique_values = sorted(set(values))
            entries = [
                LegendEntry(str(value), value, {"fillColor": f"#{hash(value) % 0xFFFFFF:06x}"})
                for value in unique_values
            ]
            return DiscreteColorMapLegend(title, entries)

        else:
            from mescal.utils.pretty_scaling import get_pretty_min_max
            vmin, vmax = get_pretty_min_max(values)
            return ContinuousColorMapLegend(title, vmin, vmax)


if __name__ == "__main__":
    # Example usage
    values = [10, 20, 30, 40, 50]

    # Continuous color legend
    continuous_legend = LegendFactory.create_legend(
        "Power Flow",
        VisualizationTypeEnum.Area,
        values
    )

    # Discrete color legend
    discrete_legend = LegendFactory.create_legend(
        "Status",
        VisualizationTypeEnum.Point,
        [0, 1, 2],
        is_discrete=True
    )

    # Symbol legend for borders
    symbol_legend = LegendFactory.create_legend(
        "Flow Direction",
        VisualizationTypeEnum.Border,
        [-1, 1]
    )