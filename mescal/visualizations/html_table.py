from typing import List, Dict, Any, Optional, Union
import os
import json
import uuid
import pandas as pd


class HTMLTable:
    """
    A class to create interactive Tabulator tables from pandas DataFrames.

    Tabulator (http://tabulator.info/) is a feature-rich interactive table library
    that provides sorting, filtering, formatting, and editing capabilities.
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
        """Generate column definitions for Tabulator based on DataFrame columns and types."""
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
        """Process DataFrame to handle NaN values and convert to JSON records."""
        processed_df = self.df.copy()

        for col in processed_df.columns:
            if pd.api.types.is_numeric_dtype(processed_df[col]):
                processed_df[col] = processed_df[col].astype(float)
            processed_df[col] = processed_df[col].where(~pd.isna(processed_df[col]), None)

        records = processed_df.to_dict(orient='records')
        return records

    def _get_container_style_string(self) -> str:
        """Convert container style dictionary to CSS string."""
        base_style = "padding: 10px; margin: 15px 0;"

        for key, value in self.container_style.items():
            css_key = ''.join(word.capitalize() if i > 0 else word
                              for i, word in enumerate(key.split('_')))
            base_style += f" {css_key}: {value};"

        return base_style

    def get_html(self, include_dependencies: bool = True) -> str:
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
        """
        Save the table as a standalone HTML file.

        Args:
            filepath: Path where to save the HTML file
            title: Title for the HTML document

        Returns:
            The filepath of the saved file
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

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        return filepath
