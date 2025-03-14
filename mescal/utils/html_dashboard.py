import os
from typing import List, Union, Dict

import plotly.graph_objects as go

from mescal.utils.logging import get_logger

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
    def __init__(self, name: str = None):
        self.name = name if name else 'HTML Dashboard'
        self.content: Dict[str, HTMLDashboardElement] = dict()

    def add_plotly_figure(self, fig: go.Figure, height: str = '100%', name: str = None):
        element = HTMLDashboardElement(fig, height, name)
        self.content[element.name] = element

    def add_html(self, html_string: str, name: str = None):
        element = HTMLDashboardElement(html_string, name=name)
        self.content[element.name] = element

    def save(self, save_to_path: str, content_order: List[str] = None):
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
            dashboard.write("</head><body>\n")

            for item in content:
                dashboard.write(item + "\n")

            dashboard.write("</body></html>\n")


if __name__ == '__main__':
    from eda.visualizations.data_table import DataTable
    import plotly.express as px

    data = px.data.iris()
    figs = [
        px.scatter(data, x="sepal_width", y="sepal_length", color="species"),
        px.scatter(data, x="petal_width", y="petal_length", color="species"),
        px.scatter(data, x="sepal_width", y="petal_length", color="species"),
    ]

    dashboard = HTMLDashboard(name='Figure Dashboard')
    for f in figs:
        dashboard.add_plotly_figure(f)

    dashboard.add_html(DataTable().get_html_content(data))

    dashboard.add_html("<h1>Custom HTML Content</h1>")
    dashboard.save('_temp/figure_dashboard.html')
