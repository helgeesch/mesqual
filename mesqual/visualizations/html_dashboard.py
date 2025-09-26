import os
from typing import List, Union, Dict, TYPE_CHECKING

import plotly.graph_objects as go

from mesqual.utils.logging import get_logger

if TYPE_CHECKING:
    import folium
    from mesqual.visualizations.html_table import HTMLTable

logger = get_logger(__name__)


class HTMLDashboardElement:
    """A wrapper for dashboard elements that can be HTML strings or Plotly figures.

    This class serves as a container for individual dashboard components, storing
    the element content along with display properties like height and a unique name.

    Args:
        element: The dashboard element content, either a Plotly figure or HTML string.
        height: CSS height specification for the element. Defaults to '100%'.
        name: Unique identifier for the element. If None, auto-generates using object id.

    Attributes:
        element: The stored dashboard element (figure or HTML string).
        height: The CSS height specification.
        name: The unique identifier for this element.
    """
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
    """A dashboard builder for creating HTML reports with multiple visualizations.

    This class provides a flexible way to combine Plotly figures, Folium maps,
    HTML tables, custom HTML content, and section dividers into a single HTML
    dashboard file. Elements are stored with unique names and can be ordered
    when saving the final dashboard.

    Args:
        name: The dashboard title. Defaults to 'HTML Dashboard'.
        font_family: CSS font family specification for the dashboard.
            Defaults to "Arial, sans-serif".

    Attributes:
        name: The dashboard title.
        content: Dictionary mapping element names to HTMLDashboardElement objects.
        font_family: The CSS font family specification.

    Example:

        >>> import plotly.express as px
        >>> dashboard = HTMLDashboard(name="My Analysis")
        >>> fig = px.scatter(px.data.iris(), x="sepal_width", y="sepal_length")
        >>> dashboard.add_plotly_figure(fig, name="iris_scatter")
        >>> dashboard.save("analysis.html")
    """
    def __init__(self, name: str = None, font_family: str = "Arial, sans-serif"):
        self.name = name if name else 'HTML Dashboard'
        self.content: Dict[str, HTMLDashboardElement] = dict()
        self.font_family = font_family

    def add_plotly_figure(self, fig: go.Figure, height: str = '100%', name: str = None):
        """Add a Plotly figure to the dashboard.

        Args:
            fig: The Plotly figure to add.
            height: CSS height specification for the figure. Defaults to '100%'.
            name: Unique identifier for the figure. If None, auto-generates.

        Example:

            >>> import plotly.express as px
            >>> fig = px.bar(x=["A", "B", "C"], y=[1, 3, 2])
            >>> dashboard.add_plotly_figure(fig, height="400px", name="my_bar_chart")
        """
        element = HTMLDashboardElement(fig, height, name)
        self.content[element.name] = element

    def add_html(self, html_string: str, name: str = None):
        """Add custom HTML content to the dashboard.

        Args:
            html_string: The HTML content to add.
            name: Unique identifier for the HTML content. If None, auto-generates.

        Example:

            >>> html = "<div><h2>Custom Section</h2><p>Some content here.</p></div>"
            >>> dashboard.add_html(html, name="custom_section")
        """
        element = HTMLDashboardElement(html_string, name=name)
        self.content[element.name] = element

    def add_folium_map(
            self,
            folium_map: 'folium.Map',
            name: str = None,
    ):
        """Add a Folium map to the dashboard.

        Args:
            folium_map: The Folium map object to add.
            name: Unique identifier for the map. If None, auto-generates
                as "folium_map_{index}".

        Returns:
            str: The name assigned to the map element.

        Example:

            >>> import folium
            >>> m = folium.Map(location=[45.5236, -122.6750], zoom_start=13)
            >>> map_name = dashboard.add_folium_map(m, name="portland_map")
        """
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
        """Add an HTML table to the dashboard.

        Args:
            table: The HTMLTable object to add.
            name: Unique identifier for the table. If None, derives from table title
                or uses table_id.
            include_dependencies: Whether to include CSS/JS dependencies in the
                table HTML. Defaults to True.

        Returns:
            str: The name assigned to the table element.

        Raises:
            ValueError: If the table cannot be converted to HTML.

        Example:

            >>> from mesqual.visualizations.html_table import HTMLTable
            >>> import pandas as pd
            >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            >>> table = HTMLTable(df, title="Sample Data")
            >>> table_name = dashboard.add_table(table, name="sample_table")
        """
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
        """Add a styled section divider with title and optional subtitle.

        Creates a formatted section header that can be used to organize dashboard
        content into logical groups. Supports extensive CSS customization through
        parameters and keyword arguments.

        Args:
            title: The main section title.
            subtitle: Optional subtitle text.
            name: Unique identifier for the divider. If None, derives from title.
            background_color: CSS background color. Defaults to "#f9f9f9".
            title_color: CSS color for the title text. Defaults to "#333".
            subtitle_color: CSS color for the subtitle text. Defaults to "#666".
            padding: CSS padding specification. Defaults to "20px".
            margin: CSS margin specification. Defaults to "20px 0".
            text_align: CSS text alignment. Defaults to "center".
            title_font_size: CSS font size for title. Defaults to "24px".
            subtitle_font_size: CSS font size for subtitle. Defaults to "16px".
            border_radius: CSS border radius. Defaults to "0px".
            border: CSS border specification. Defaults to "none".
            **kwargs: Additional CSS properties. Underscores in keys are converted
                to camelCase (e.g., box_shadow becomes boxShadow).

        Returns:
            str: The name assigned to the section divider element.

        Example:

            >>> divider_name = dashboard.add_section_divider(
            ...     title="Data Analysis Results",
            ...     subtitle="Generated on 2024-01-01",
            ...     background_color="#e3f2fd",
            ...     border="1px solid #2196f3"
            ... )
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
        """Save the dashboard as an HTML file.

        Generates a complete HTML document containing all dashboard elements.
        Automatically handles Plotly.js inclusion (only includes once for efficiency)
        and creates output directories as needed.

        Args:
            save_to_path: File path where the HTML dashboard will be saved.
            content_order: Optional list specifying the order of elements in the
                dashboard. If None, uses the order elements were added. Must
                contain only valid element names.

        Raises:
            KeyError: If content_order contains names not found in the dashboard.
            TypeError: If an element has an unexpected type (internal error).

        Example:

            >>> dashboard.save("my_dashboard.html")
            >>> # Custom ordering
            >>> dashboard.save("ordered_dashboard.html",
            ...               content_order=["intro_section", "chart1", "table1"])
        """
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
        """Display the dashboard inline in a Jupyter notebook.

        Creates a temporary HTML file and displays it using an IPython IFrame.
        This method is designed for use within Jupyter notebooks to provide
        inline dashboard previews.

        Args:
            width: CSS width specification for the iframe. Defaults to "100%".
            height: CSS height specification for the iframe. Defaults to "600".

        Note:
            This method requires IPython and is intended for Jupyter notebook use.
            The temporary file is created in the system temp directory.

        Example:

            >>> # In a Jupyter notebook cell
            >>> dashboard.show(width="100%", height="800px")
        """
        import tempfile
        from pathlib import Path
        from IPython.display import IFrame, display

        tmp_dir = Path(tempfile.mkdtemp())
        html_path = tmp_dir / "dashboard.html"
        self.save(html_path)
        display(IFrame(src=str(html_path.resolve()), width=width, height=height))


if __name__ == '__main__':
    from mesqual.visualizations.html_table import HTMLTable
    import plotly.express as px
    import plotly.graph_objects as go

    # Load sample data
    data = px.data.iris()

    # Create dashboard
    dashboard = HTMLDashboard(
        name='MESQUAL Visualization Dashboard',
        font_family="'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
    )

    # Add main header
    dashboard.add_section_divider(
        title='MESQUAL Dashboard Example',
        subtitle='Demonstrating multiple visualization types',
        background_color="#2c3e50",
        title_color="white",
        subtitle_color="#ecf0f1",
        border_radius="8px"
    )

    # Add plotly figures with different types
    scatter_fig = px.scatter(
        data,
        x="sepal_width",
        y="sepal_length",
        color="species",
        title="Sepal Dimensions by Species"
    )
    dashboard.add_plotly_figure(
        scatter_fig,
        height="450px",
        name="sepal_scatter"
    )

    # Box plot example
    box_fig = px.box(
        data,
        x="species",
        y="petal_length",
        title="Petal Length Distribution"
    )
    dashboard.add_plotly_figure(
        box_fig,
        height="400px",
        name="petal_boxplot"
    )

    # Custom plotly figure with annotations
    custom_fig = go.Figure()
    custom_fig.add_scatter(
        x=data["sepal_width"],
        y=data["petal_length"],
        mode='markers',
        marker=dict(size=8, color='lightblue', line=dict(width=1, color='navy')),
        name='All Species'
    )
    custom_fig.update_layout(
        title="Custom Scatter Plot with Styling",
        xaxis_title="Sepal Width (cm)",
        yaxis_title="Petal Length (cm)"
    )
    dashboard.add_plotly_figure(custom_fig, name="custom_scatter")

    # Add data table section
    dashboard.add_section_divider(
        'Data Table Integration',
        'Interactive table with the underlying data'
    )

    # Create and add table
    sample_data = data.head(10)  # First 10 rows for demo
    table = HTMLTable(sample_data, title="Iris Dataset Sample")
    dashboard.add_table(table, name="iris_sample_table")

    # Add custom HTML content
    dashboard.add_section_divider('Custom HTML Content')
    custom_html = """
    <div style="background-color: #f8f9fa; padding: 20px; border-left: 4px solid #007bff;">
        <h3 style="color: #007bff; margin-top: 0;">Analysis Summary</h3>
        <ul>
            <li><strong>Dataset:</strong> Iris flower measurements</li>
            <li><strong>Records:</strong> 150 samples</li>
            <li><strong>Species:</strong> Setosa, Versicolor, Virginica</li>
            <li><strong>Variables:</strong> Sepal/Petal length and width</li>
        </ul>
        <p><em>This dashboard demonstrates the integration of multiple visualization types
        using the MESQUAL HTMLDashboard class.</em></p>
    </div>
    """
    dashboard.add_html(custom_html, name="analysis_summary")

    # Save dashboard with custom content order
    content_order = [
        "section_mesqual_dashboard_example",
        "sepal_scatter",
        "petal_boxplot",
        "custom_scatter",
        "section_data_table_integration",
        "iris_sample_table",
        "section_custom_html_content",
        "analysis_summary"
    ]

    dashboard.save('_tmp/figure_dashboard.html', content_order=content_order)
    print("Dashboard saved to '_tmp/figure_dashboard.html'")
    print(f"Dashboard contains {len(dashboard.content)} elements:")
    for name, element in dashboard.content.items():
        element_type = "Plotly Figure" if isinstance(element.element, go.Figure) else "HTML Content"
        print(f"  - {name}: {element_type}")
