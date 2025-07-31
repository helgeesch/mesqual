import os
from typing import List, Union, Dict, TYPE_CHECKING

import plotly.graph_objects as go

from mescal.utils.logging import get_logger

if TYPE_CHECKING:
    import folium
    from mescal.visualizations.html_table import HTMLTable

logger = get_logger(__name__)


class HTMLDashboardElement:
    def __init__(
            self,
            element: Union[go.Figure, str],
            height: str = '100%',
            name: str = None,
    ):
        self.element = element
        self.height = height
        self.name = name or str(id(self))
        if name is None:
            logger.info(f'No name passed for {type(self).__name__}. Automatically generated name: {self.name}')


class HTMLDashboard:
    def __init__(self, name: str = None, font_family: str = "Arial, sans-serif"):
        self.name = name if name else 'HTML Dashboard'
        self.content: Dict[str, HTMLDashboardElement] = dict()
        self.font_family = font_family

    def add_plotly_figure(self, fig: go.Figure, height: str = '100%', name: str = None):
        element = HTMLDashboardElement(fig, height, name)
        self.content[element.name] = element

    def add_html(self, html_string: str, name: str = None):
        element = HTMLDashboardElement(html_string, name=name)
        self.content[element.name] = element

    def add_folium_map(
            self,
            folium_map: 'folium.Map',
            name: str = None,
    ):
        map_html = folium_map._repr_html_()

        if name is None:
            name = f"folium_map_{len([k for k in self.content.keys() if 'folium_map' in k])}"

        wrapped_map_html = f'<div>{map_html}</div>'

        self.add_html(wrapped_map_html, name=name)

        return name

    def add_table(
            self,
            table: 'HTMLTable',
            name: str = None,
            include_dependencies: bool = True
    ) -> str:
        try:
            table_html = table.get_html(include_dependencies=include_dependencies)

            if name is None:
                name = f"table_{table.title.lower().replace(' ', '_')}" if table.title else table.table_id

            self.add_html(table_html, name=name)
            return name

        except Exception as e:
            raise ValueError(f"Failed to add table to dashboard: {str(e)}") from e

    def add_section_divider(
            self,
            title: str,
            subtitle: str = None,
            name: str = None,
            background_color: str = "#f9f9f9",
            title_color: str = "#333",
            subtitle_color: str = "#666",
            padding: str = "20px",
            margin: str = "20px 0",
            text_align: str = "center",
            title_font_size: str = "24px",
            subtitle_font_size: str = "16px",
            border_radius: str = "0px",
            border: str = "none",
            **kwargs
    ) -> str:
        """
        Add a section divider with a title and optional subtitle.
        """
        base_style = f"background-color: {background_color}; padding: {padding}; margin: {margin}; text-align: {text_align}; border-radius: {border_radius}; border: {border};"

        for key, value in kwargs.items():
            css_key = ''.join(word.capitalize() if i > 0 else word for i, word in enumerate(key.split('_')))
            base_style += f" {css_key}: {value};"

        html = f'<div style="{base_style}">\n'
        html += f'    <h2 style="color: {title_color}; font-size: {title_font_size};">{title}</h2>\n'

        if subtitle:
            html += f'    <p style="color: {subtitle_color}; font-size: {subtitle_font_size};">{subtitle}</p>\n'

        html += '</div>'

        if name is None:
            name = f"section_{title.lower().replace(' ', '_')}"

        self.add_html(html, name=name)

        return name

    def save(self, save_to_path, content_order=None):
        if not os.path.exists(os.path.dirname(save_to_path)):
            os.makedirs(os.path.dirname(save_to_path))

        if content_order is None:
            content_order = list(self.content.keys())
        else:
            unrecognized = [k for k in content_order if k not in self.content.keys()]
            if unrecognized:
                raise KeyError(f'Unrecognized content names: {unrecognized}. Allowed: {self.content.keys()}')

        content = []
        plotly_js_included = False
        for key in content_order:
            v = self.content[key]
            if isinstance(v.element, go.Figure):
                html_text = v.element.to_html(
                    include_plotlyjs=True if not plotly_js_included else False,
                    full_html=False,
                    default_height=v.height
                )
                content.append(html_text)
                plotly_js_included = True
            elif isinstance(v.element, str):
                content.append(v.element)
            else:
                TypeError(f'Unexpected element type: {type(v.element)}')

        with open(save_to_path, 'w', encoding='utf-8') as dashboard:
            dashboard.write("<html><head>\n")
            dashboard.write("<meta charset='UTF-8'>\n")
            dashboard.write(f"<title>{self.name}</title>\n")
            dashboard.write(f"<style>\n  body, * {{ font-family: {self.font_family}; }}\n</style>\n")
            dashboard.write("</head><body>\n")

            for item in content:
                dashboard.write(item + "\n")

            dashboard.write("</body></html>\n")

    def show(self, width: str = "100%", height: str = "600"):
        """
        Display the dashboard inline as an interactive HTML iframe.
        """
        import tempfile
        from pathlib import Path
        from IPython.display import IFrame, display

        tmp_dir = Path(tempfile.mkdtemp())
        html_path = tmp_dir / "dashboard.html"
        self.save(html_path)
        display(IFrame(src=str(html_path.resolve()), width=width, height=height))


if __name__ == '__main__':
    from mescal.visualizations.html_table import HTMLTable
    import plotly.express as px

    data = px.data.iris()
    figs = [
        px.scatter(data, x="sepal_width", y="sepal_length", color="species"),
        px.scatter(data, x="petal_width", y="petal_length", color="species"),
        px.scatter(data, x="sepal_width", y="petal_length", color="species"),
    ]

    dashboard = HTMLDashboard(name='Figure Dashboard')

    dashboard.add_section_divider('Your Custom Dashboard', 'your subtitle')

    for f in figs:
        dashboard.add_plotly_figure(f)

    dashboard.add_section_divider('Now on to the TabulatorTable integration')
    dashboard.add_table(HTMLTable(data))


    dashboard.add_section_divider('Now some custom HTML content')
    dashboard.add_html("<h1>Custom HTML Content</h1>")
    dashboard.save('_tmp/figure_dashboard.html')
