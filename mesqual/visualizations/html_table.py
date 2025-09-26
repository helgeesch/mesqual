from typing import List, Dict, Any, Optional, Union
import os
import json
import uuid
import pandas as pd


class HTMLTable:
    """A class to create interactive Tabulator tables from pandas DataFrames.

    Tabulator (http://tabulator.info/) is a feature-rich interactive table library
    that provides sorting, filtering, formatting, and editing capabilities.

    This class automatically detects column types and applies appropriate formatters,
    sorters, and filters. It supports customization through column configuration
    and provides methods to display tables in Jupyter notebooks or save as HTML files.

    Attributes:
        df (pd.DataFrame): The pandas DataFrame to display.
        title (Optional[str]): Optional title for the table.
        height (str): Height of the table container.
        layout (str): Tabulator layout mode.
        theme (str): Visual theme for the table.
        pagination (Union[str, bool]): Pagination settings.
        page_size (int): Number of rows per page when pagination is enabled.
        movable_columns (bool): Whether columns can be moved.
        resizable_columns (bool): Whether columns can be resized.
        column_config (Optional[Dict[str, Dict[str, Any]]]): Custom column configurations.
        responsive (bool): Whether table should be responsive.
        selectable (bool): Whether rows can be selected.
        table_id (Optional[str]): Unique identifier for the table.
        container_style (Optional[Dict[str, str]]): Custom CSS styles for container.
        default_float_precision (int | None): Default decimal places for float columns.
    """

    def __init__(
            self,
            df: pd.DataFrame,
            title: Optional[str] = None,
            height: str = "400px",
            layout: str = "fitColumns",  # fitColumns, fitData, fitDataFill
            theme: str = "simple",  # simple, bootstrap, midnight, modern, etc.
            pagination: Union[str, bool] = "local",  # local, remote, or False
            page_size: int = 10,
            movable_columns: bool = True,
            resizable_columns: bool = True,
            column_config: Optional[Dict[str, Dict[str, Any]]] = None,
            responsive: bool = True,
            selectable: bool = False,
            table_id: Optional[str] = None,
            container_style: Optional[Dict[str, str]] = None,
            default_float_precision: int | None = 2,
    ):
        """Initialize the HTMLTable with configuration options.

        Args:
            df: The pandas DataFrame to display in the table.
            title: Optional title to display above the table.
            height: CSS height value for the table container (default: "400px").
            layout: Tabulator layout mode - "fitColumns", "fitData", or "fitDataFill".
            theme: Visual theme - "simple", "bootstrap", "midnight", "modern", etc.
            pagination: Pagination mode - "local", "remote", or False to disable.
            page_size: Number of rows to display per page when pagination is enabled.
            movable_columns: Whether users can drag columns to reorder them.
            resizable_columns: Whether users can resize column widths.
            column_config: Dictionary mapping column names to Tabulator column definitions
                for custom formatting, validation, or behavior.
            responsive: Whether the table should adapt to different screen sizes.
            selectable: Whether users can select table rows.
            table_id: Custom HTML ID for the table element. If None, generates unique ID.
            container_style: CSS styles to apply to the table container as key-value pairs.
            default_float_precision: Number of decimal places for float columns.
                Set to None to disable automatic formatting.

        Example:

            >>> import pandas as pd
            >>> df = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Score': [95.5, 87.2]})
            >>> table = HTMLTable(df, title="Student Scores", height="300px")
        """
        self.df = df
        self.title = title
        self.height = height
        self.layout = layout
        self.theme = theme
        self.pagination = pagination
        self.page_size = page_size
        self.movable_columns = movable_columns
        self.resizable_columns = resizable_columns
        self.column_config = column_config or {}
        self.responsive = responsive
        self.selectable = selectable
        self.table_id = table_id or f"tabulator_{str(uuid.uuid4()).replace('-', '')[:8]}"
        self.container_style = container_style or {}
        self.default_float_precision = default_float_precision

    def _get_column_definitions(self) -> List[Dict[str, Any]]:
        """Generate column definitions for Tabulator based on DataFrame columns and types.

        Automatically detects column data types and applies appropriate Tabulator
        configurations including sorters, formatters, and filters. Custom configurations
        from column_config override automatic detection.

        Returns:
            List of column definition dictionaries compatible with Tabulator.
        """
        column_defs = []

        for col in self.df.columns:
            # Start with default configuration
            col_def = {
                "title": str(col),
                "field": str(col),
                "headerFilter": "input",
            }

            if col in self.column_config:
                col_def.update(self.column_config[col])
            else:
                sample_data = self.df[col].dropna().iloc[0] if not self.df[col].dropna().empty else None

                if pd.api.types.is_numeric_dtype(self.df[col]):
                    col_def["sorter"] = "number"
                    col_def["headerFilter"] = "number"
                    col_def["hozAlign"] = "right"

                    if pd.api.types.is_float_dtype(self.df[col]) and (self.default_float_precision is not None):
                        if col not in self.column_config or "formatter" not in self.column_config[col]:
                            col_def["formatter"] = "money"
                            col_def["formatterParams"] = {
                                "symbol": "",
                                "precision": str(self.default_float_precision),
                                "thousand": ",",
                                "decimal": "."
                            }
                elif pd.api.types.is_datetime64_dtype(self.df[col]):
                    col_def["sorter"] = "date"
                    col_def["headerFilter"] = "input"
                    col_def["formatter"] = "datetime"
                    col_def["formatterParams"] = {"outputFormat": "YYYY-MM-DD HH:mm:ss"}
                elif isinstance(sample_data, bool) or pd.api.types.is_bool_dtype(self.df[col]):
                    col_def["sorter"] = "boolean"
                    col_def["formatter"] = "tickCross"
                    col_def["headerFilter"] = "tickCross"
                    col_def["hozAlign"] = "center"
                else:
                    col_def["sorter"] = "string"

            column_defs.append(col_def)

        return column_defs

    def _process_dataframe(self) -> List[Dict[str, Any]]:
        """Process DataFrame to handle NaN values and convert to JSON records.

        Converts pandas DataFrame to a format suitable for Tabulator by handling
        NaN values, ensuring numeric columns are properly typed, and converting
        to JSON-serializable records.

        Returns:
            List of dictionaries representing table rows.
        """
        processed_df = self.df.copy()

        for col in processed_df.columns:
            if pd.api.types.is_numeric_dtype(processed_df[col]):
                processed_df[col] = processed_df[col].astype(float)
            processed_df[col] = processed_df[col].where(~pd.isna(processed_df[col]), None)

        records = processed_df.to_dict(orient='records')
        return records

    def _get_container_style_string(self) -> str:
        """Convert container style dictionary to CSS string.

        Transforms Python-style CSS property names (snake_case) to CSS format
        (camelCase) and combines with base styling.

        Returns:
            CSS style string for the table container.
        """
        base_style = "padding: 10px; margin: 15px 0;"

        for key, value in self.container_style.items():
            css_key = ''.join(word.capitalize() if i > 0 else word
                              for i, word in enumerate(key.split('_')))
            base_style += f" {css_key}: {value};"

        return base_style

    def get_html(self, include_dependencies: bool = True) -> str:
        """Generate HTML representation of the interactive table.

        Creates the complete HTML markup including JavaScript initialization
        for the Tabulator table with all configured options.

        Args:
            include_dependencies: Whether to include Tabulator CSS/JS dependencies
                in the output. Set to False when embedding in documents that already
                include these dependencies.

        Returns:
            Complete HTML string ready for display or embedding.

        Example:

            >>> table = HTMLTable(df, title="My Data")
            >>> html_output = table.get_html()
            >>> # Save to file or display in web application
        """
        processed_data = self._process_dataframe()
        column_defs = self._get_column_definitions()
        container_style = self._get_container_style_string()

        html_parts = []

        if include_dependencies:
            html_parts.append(
                """
                <link href="https://unpkg.com/tabulator-tables@5.4.4/dist/css/tabulator.min.css" rel="stylesheet">
                <script type="text/javascript" src="https://unpkg.com/tabulator-tables@5.4.4/dist/js/tabulator.min.js"></script>
                """
            )

        html_parts.append(f'<div style="{container_style}">')

        if self.title:
            html_parts.append(f'<h3 style="margin-bottom: 10px;">{self.title}</h3>')

        html_parts.append(f'<div id="{self.table_id}" style="height: {self.height};"></div>')

        html_parts.append(
            f"""
            <script type="text/javascript">
                document.addEventListener('DOMContentLoaded', function() {{
                    var tabledata = {json.dumps(processed_data)};
            
                    var table = new Tabulator("#{self.table_id}", {{
                        data: tabledata,
                        columns: {json.dumps(column_defs)},
                        layout: "{self.layout}",
                        responsiveLayout: {json.dumps(self.responsive)},
                        movableColumns: {json.dumps(self.movable_columns)},
                        resizableColumns: {json.dumps(self.resizable_columns)},
                        selectable: {json.dumps(self.selectable)},
                    """
        )

        if self.pagination:
            html_parts.append(f"""
            pagination: "{self.pagination}",
            paginationSize: {self.page_size},
            paginationSizeSelector: [10, 25, 50, 100, true],
            """)

        if self.theme:
            html_parts.append(f'    theme: "{self.theme}",')

        html_parts.append(
            """
                    });
                });
            </script>
            """
        )
        html_parts.append("</div>")

        return "\n".join(html_parts)

    def save_html(self, filepath: str, title: str = "Tabulator Table") -> str:
        """Save the table as a standalone HTML file.

        Creates a complete HTML document with embedded Tabulator dependencies
        and saves it to the specified path. The resulting file can be opened
        directly in any web browser.

        Args:
            filepath: Path where to save the HTML file. Parent directories
                will be created if they don't exist.
            title: Title for the HTML document head section.

        Returns:
            The filepath of the saved file (same as input parameter).

        Example:

            >>> table = HTMLTable(df, title="Sales Report")
            >>> saved_path = table.save_html("reports/sales_table.html")
            >>> print(f"Table saved to: {saved_path}")
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        html = f"""<!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <link href="https://unpkg.com/tabulator-tables@5.4.4/dist/css/tabulator.min.css" rel="stylesheet">
            <script type="text/javascript" src="https://unpkg.com/tabulator-tables@5.4.4/dist/js/tabulator.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                {self.get_html(include_dependencies=False)}
            </div>
        </body>
        </html>
        """

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        return filepath

    def show(self, width: str = "100%", height: str = "600"):
        """Display the interactive Tabulator table inline in a Jupyter notebook.

        Creates a temporary HTML file and displays it using an IPython IFrame.
        This method is designed for use within Jupyter notebook environments.

        Args:
            width: Width of the display frame (CSS units or percentage).
            height: Height of the display frame in pixels (as string).

        Note:
            This method requires IPython to be available and will only work
            in Jupyter notebook environments.

        Example:

            >>> # In a Jupyter notebook cell:
            >>> table = HTMLTable(df, title="Interactive Data")
            >>> table.show(width="800px", height="400")
        """
        import tempfile
        from pathlib import Path
        from IPython.display import IFrame, display

        tmp_dir = Path(tempfile.mkdtemp())
        html_path = tmp_dir / f"{self.table_id}.html"
        self.save_html(str(html_path))
        display(IFrame(src=str(html_path.resolve()), width=width, height=height))


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    # Create sample data with different column types
    np.random.seed(42)
    dates = [datetime.now() - timedelta(days=i) for i in range(10)]

    sample_data = {
        'Date': dates,
        'Product': ['Widget A', 'Widget B', 'Gadget X', 'Tool Y', 'Device Z'] * 2,
        'Sales': np.random.uniform(1000, 5000, 10).round(2),
        'Units': np.random.randint(10, 100, 10),
        'Active': np.random.choice([True, False], 10),
        'Rating': np.random.uniform(1, 5, 10).round(1)
    }

    df = pd.DataFrame(sample_data)

    # Basic table example
    print("Creating basic HTMLTable...")
    table = HTMLTable(
        df=df,
        title="Sales Dashboard",
        height="500px",
        theme="simple",
        pagination="local",
        page_size=5
    )

    # Save to HTML file
    output_file = "sample_table.html"
    saved_path = table.save_html(output_file, "Sample Sales Data")
    print(f"Table saved to: {saved_path}")

    # Example with custom column configuration
    print("\nCreating table with custom column formatting...")
    column_config = {
        'Sales': {
            'formatter': 'money',
            'formatterParams': {'symbol': '$', 'precision': '2'}
        },
        'Rating': {
            'formatter': 'star',
            'formatterParams': {'stars': 5}
        }
    }

    custom_table = HTMLTable(
        df=df,
        title="Enhanced Sales Dashboard",
        column_config=column_config,
        theme="bootstrap",
        selectable=True
    )

    custom_output = "enhanced_table.html"
    custom_table.save_html(custom_output, "Enhanced Sales Data")
    print(f"Enhanced table saved to: {custom_output}")
