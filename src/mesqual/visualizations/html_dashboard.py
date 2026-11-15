import logging
import os
from html import escape
from typing import Any, Literal, Union, Dict, Tuple, TYPE_CHECKING
from collections import OrderedDict

import plotly.graph_objects as go


if TYPE_CHECKING:
    import folium
    from mesqual.visualizations.html_table import HTMLTable

logger = logging.getLogger(__name__)


class HTMLDashboardElement:
    """A wrapper for dashboard elements that can be HTML strings or Plotly figures.

    This class serves as a container for individual dashboard components, storing
    the element content along with display properties like height and a unique name.

    Args:
        element: The dashboard element content, either a Plotly figure or HTML string.
        height: CSS height specification for the element. Defaults to '100%'.
        name: Unique identifier for the element. If None, auto-generates using object id.
        tab: Optional tab path. A string places the element in a top-level tab.
            A tuple of strings places it in nested sub-tabs of arbitrary depth,
            e.g. `("Market Results", "Volumes", "Import")`. If None, the element is ungrouped
            (rendered at the top level in scroll mode, or placed in a default tab
            in tabbed mode).

    Attributes:
        element: The stored dashboard element (figure or HTML string).
        height: The CSS height specification.
        name: The unique identifier for this element.
        tab: The tab path, normalised to a tuple or None.
    """

    def __init__(
        self,
        element: Union[go.Figure, str],
        height: str = '100%',
        name: str = None,
        tab: Union[str, Tuple[str, ...], None] = None,
        force_height: bool = False,
    ):
        self.element = element
        self.height = height
        self.force_height = force_height
        self.name = name or str(id(self))
        if name is None:
            logger.info(f'No name passed for {type(self).__name__}. Automatically generated name: {self.name}')

        # Normalise tab to tuple or None
        if isinstance(tab, str):
            self.tab = (tab,)
        elif isinstance(tab, tuple):
            if len(tab) == 0:
                raise ValueError('tab tuple must have at least one element')
            self.tab = tab
        else:
            self.tab = None


class HTMLDashboard:
    """A dashboard builder for creating HTML reports with multiple visualizations.

    This class combines Plotly figures, Folium maps,
    HTML tables, custom HTML content, and section dividers into a single HTML
    dashboard file. Elements are stored with unique names and can be ordered
    when saving the final dashboard.

    An auto-generated name that already exists gets a counter suffix (`_2`,
    `_3`, ...). An explicit name that already exists replaces the element and
    logs a warning. The page title is HTML-escaped; tab labels and element
    content are written as raw HTML.

    Args:
        name: The dashboard title. Defaults to 'HTML Dashboard'.
        font_family: CSS font family specification for the dashboard.
            Defaults to "Arial, sans-serif".
        plotly_js: How the page gets plotly.js. `'embed'` (default) writes
            the full library into the page, so the page works offline.
            `'cdn'` writes one `<script>` tag that loads plotly.js from
            the plotly CDN, so the page is much smaller but needs
            internet access to show figures.

    Attributes:
        name: The dashboard title.
        content: Dictionary mapping element names to HTMLDashboardElement objects.
        font_family: The CSS font family specification.
        plotly_js: `'embed'` or `'cdn'`. Can be changed between saves.

    Raises:
        ValueError: If `plotly_js` is not `'embed'` or `'cdn'`.

    Examples:
        >>> import plotly.express as px
        >>> from mesqual.visualizations import HTMLTable
        >>> prices = study.scen.fetch('Country.Results.price')
        >>> dashboard = HTMLDashboard(name='Prices')
        >>> dashboard.add_section_divider('Scenarios', subtitle='Mean price per country')
        'section_scenarios'
        >>> dashboard.add_table(HTMLTable(prices.mean().unstack('Country').round(2), title='Mean price'))
        'table_mean_price'
        >>> dashboard.add_plotly_figure(px.line(prices['base']), name='base_prices')
        >>> list(dashboard.content)
        ['section_scenarios', 'table_mean_price', 'base_prices']
        >>> dashboard.save('prices.html')
    """

    _PLOTLY_JS_OPTIONS = {'embed': True, 'cdn': 'cdn'}

    def __init__(
        self,
        name: str = None,
        font_family: str = 'Arial, sans-serif',
        plotly_js: Literal['embed', 'cdn'] = 'embed',
    ):
        self.name = name if name else 'HTML Dashboard'
        self.content: Dict[str, HTMLDashboardElement] = dict()
        self.font_family = font_family
        self.plotly_js = plotly_js
        self._include_plotlyjs()
        self._plotly_js_for_next_figure: bool | str = False

    def _include_plotlyjs(self) -> bool | str:
        if not isinstance(self.plotly_js, str) or self.plotly_js not in self._PLOTLY_JS_OPTIONS:
            raise ValueError(f'plotly_js must be one of {list(self._PLOTLY_JS_OPTIONS)}, got {self.plotly_js!r}.')
        return self._PLOTLY_JS_OPTIONS[self.plotly_js]

    def add_plotly_figure(
        self,
        fig: go.Figure,
        height: str = '100%',
        name: str = None,
        tab: Union[str, Tuple[str, ...], None] = None,
        force_height: bool = False,
    ):
        """Add a Plotly figure to the dashboard.

        Args:
            fig: The Plotly figure to add.
            height: CSS height specification for the figure. Defaults to '100%'.
                Only used as the rendered container height when `force_height`
                is True or when the figure has no explicit `layout.height`.
            name: Unique identifier for the figure. If None, auto-generates.
            tab: Optional tab path (see `HTMLDashboardElement`).
            force_height: When True, the dashboard's `height` argument wins
                and the figure's `layout.height`/`layout.width` are cleared
                so the plot adapts to the container (e.g. `height='100%'`
                fits the tab, `height='2000px'` produces a fixed-pixel
                container, parent's `overflow: auto` then handles scroll).
                When False (default), the figure's intrinsic
                `layout.height`/`layout.width` win and `height` is only
                used as a fallback if those are unset.
                In both cases the figure is responsive: a size that the layout
                does not fix follows the container when the window is resized.
        """
        element = HTMLDashboardElement(fig, height, name, tab=tab, force_height=force_height)
        self._store(element)

    def add_html(self, html_string: str, name: str = None, tab: Union[str, Tuple[str, ...], None] = None):
        """Add custom HTML content to the dashboard.

        Args:
            html_string: The HTML content to add.
            name: Unique identifier for the HTML content. If None, auto-generates.
            tab: Optional tab path (see `HTMLDashboardElement`).
        """
        element = HTMLDashboardElement(html_string, name=name, tab=tab)
        self._store(element)

    def add_folium_map(
        self,
        folium_map: 'folium.Map',
        name: str = None,
        tab: Union[str, Tuple[str, ...], None] = None,
    ) -> str:
        """Add a Folium map to the dashboard.

        Args:
            folium_map: The Folium map object to add.
            name: Unique identifier for the map. If None, uses "folium_map",
                with a counter suffix if the name exists.
            tab: Optional tab path (see `HTMLDashboardElement`).

        Returns:
            The name assigned to the map element.

        Examples:
            >>> import folium
            >>> dashboard = HTMLDashboard()
            >>> dashboard.add_folium_map(folium.Map(location=[48, 9], zoom_start=5))
            'folium_map'
            >>> dashboard.add_folium_map(folium.Map(location=[48, 9], zoom_start=7))
            'folium_map_2'
        """
        map_html = folium_map._repr_html_()

        if name is None:
            name = self._unique_name('folium_map')

        wrapped_map_html = f'<div>{map_html}</div>'

        self.add_html(wrapped_map_html, name=name, tab=tab)

        return name

    def add_table(
        self,
        table: 'HTMLTable',
        name: str = None,
        include_dependencies: bool = True,
        tab: Union[str, Tuple[str, ...], None] = None,
    ) -> str:
        """Add an HTML table to the dashboard.

        Args:
            table: The HTMLTable object to add.
            name: Unique identifier for the table. If None, derives from table title
                or uses table_id, with a counter suffix if the name exists.
            include_dependencies: Whether to include CSS/JS dependencies in the
                table HTML. Defaults to True.
            tab: Optional tab path (see `HTMLDashboardElement`).

        Returns:
            The name assigned to the table element.

        Raises:
            ValueError: If the table cannot be converted to HTML.
        """
        try:
            table_html = table.get_html(include_dependencies=include_dependencies)

            if name is None:
                name = f'table_{table.title.lower().replace(" ", "_")}' if table.title else table.table_id
                name = self._unique_name(name)

            self.add_html(table_html, name=name, tab=tab)
            return name

        except Exception as e:
            raise ValueError(f'Failed to add table to dashboard: {str(e)}') from e

    def add_section_divider(
        self,
        title: str,
        subtitle: str = None,
        name: str = None,
        tab: Union[str, Tuple[str, ...], None] = None,
        background_color: str = '#f9f9f9',
        title_color: str = '#333',
        subtitle_color: str = '#666',
        padding: str = '20px',
        margin: str = '20px 0',
        text_align: str = 'center',
        title_font_size: str = '24px',
        subtitle_font_size: str = '16px',
        border_radius: str = '0px',
        border: str = 'none',
        **kwargs: Any,
    ) -> str:
        """Add a styled section divider with title and optional subtitle.

        Creates a formatted section header that can be used to organize dashboard
        content into logical groups. Supports extensive CSS customization through
        parameters and keyword arguments.

        Args:
            title: The main section title.
            subtitle: Optional subtitle text.
            name: Unique identifier for the divider. If None, derives from title,
                with a counter suffix if the name exists.
            tab: Optional tab path (see `HTMLDashboardElement`).
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
            The name assigned to the section divider element.
        """
        base_style = f'background-color: {background_color}; padding: {padding}; margin: {margin}; text-align: {text_align}; border-radius: {border_radius}; border: {border};'

        for key, value in kwargs.items():
            css_key = ''.join(word.capitalize() if i > 0 else word for i, word in enumerate(key.split('_')))
            base_style += f' {css_key}: {value};'

        html = f'<div style="{base_style}">\n'
        html += f'    <h2 style="color: {title_color}; font-size: {title_font_size};">{title}</h2>\n'

        if subtitle:
            html += f'    <p style="color: {subtitle_color}; font-size: {subtitle_font_size};">{subtitle}</p>\n'

        html += '</div>'

        if name is None:
            name = self._unique_name(f'section_{title.lower().replace(" ", "_")}')

        self.add_html(html, name=name, tab=tab)

        return name

    # -- internal helpers ------------------------------------------------

    def _store(self, element: HTMLDashboardElement):
        if element.name in self.content:
            logger.warning(f'Dashboard element "{element.name}" already exists and is replaced.')
        self.content[element.name] = element

    def _unique_name(self, base_name: str) -> str:
        name = base_name
        counter = 2
        while name in self.content:
            name = f'{base_name}_{counter}'
            counter += 1
        return name

    def _has_tabs(self) -> bool:
        return any(el.tab is not None for el in self.content.values())

    def _resolve_content_order(self, content_order):
        if content_order is None:
            return list(self.content.keys())
        unrecognized = [k for k in content_order if k not in self.content.keys()]
        if unrecognized:
            raise KeyError(f'Unrecognized content names: {unrecognized}. Allowed: {self.content.keys()}')
        return content_order

    def _element_to_html(self, element: HTMLDashboardElement) -> str:
        """Render one element. The first figure after `save` starts gets plotly.js."""
        if isinstance(element.element, go.Figure):
            fig = element.element
            if element.force_height:
                # Clone so we don't mutate the caller's figure, then strip
                # explicit layout dimensions so plotly's `default_height`
                # actually takes effect and the plot adapts to the container.
                fig = go.Figure(fig.to_dict())
                fig.layout.height = None
                fig.layout.width = None
                fig.layout.autosize = True
            include_plotlyjs, self._plotly_js_for_next_figure = self._plotly_js_for_next_figure, False
            return fig.to_html(
                include_plotlyjs=include_plotlyjs,
                full_html=False,
                default_height=element.height,
            )
        elif isinstance(element.element, str):
            return element.element
        else:
            raise TypeError(f'Unexpected element type: {type(element.element)}')

    DEFAULT_TAB = '.'

    def _build_tab_structure(self, content_order):
        """Return an OrderedDict representing the tab tree of arbitrary depth.

        Structure::

            {
                "Market Results": {          # top-level tab
                    "_elements": [...],      # elements directly in this tab
                    "Volumes": {             # sub-tab
                        "_elements": [...],
                        "Import": {          # sub-sub-tab
                            "_elements": [...],
                        },
                    },
                },
                ...
            }
        """
        tabs: OrderedDict = OrderedDict()

        for key in content_order:
            el = self.content[key]
            path = el.tab if el.tab is not None else (self.DEFAULT_TAB,)
            node = tabs
            for label in path:
                if label not in node:
                    node[label] = OrderedDict(_elements=[])
                node = node[label]
            node['_elements'].append(key)

        return tabs

    @staticmethod
    def _scroll_css() -> str:
        """Give the body a definite height so `height:100%` figure wrappers fill the window."""
        return """
        html, body { height: 100%; }
        body { margin: 0; padding: 8px; box-sizing: border-box; }
        """

    @staticmethod
    def _tab_css() -> str:
        return """
        html, body { height: 100%; margin: 0; }
        #tabgroup_top { display: flex; flex-direction: column; height: 100vh; }
        .dashboard-tabs { flex: 0 0 auto; display: flex; flex-wrap: wrap; border-bottom: 2px solid #dee2e6; margin: 0; padding: 0; gap: 2px; }
        .dashboard-tabs button {
            padding: 10px 20px; border: 1px solid transparent; border-bottom: none;
            background: #f8f9fa; cursor: pointer; font-size: 14px; color: #495057;
            border-radius: 6px 6px 0 0; transition: background 0.15s, color 0.15s;
        }
        .dashboard-tabs button:hover { background: #e9ecef; }
        .dashboard-tabs button.active { background: #fff; color: #212529; border-color: #dee2e6; font-weight: 600; position: relative; }
        .dashboard-tabs button.active::after { content: ''; position: absolute; bottom: -2px; left: 0; right: 0; height: 2px; background: #fff; }
        .dashboard-tab-content { display: none; flex: 1 1 0; min-height: 0; overflow: auto; padding: 0; }
        .dashboard-tab-content.active { display: flex; flex-direction: column; }
        .dashboard-subtabs { flex: 0 0 auto; display: flex; flex-wrap: wrap; border-bottom: 1px solid #e9ecef; margin: 0; padding: 0; gap: 2px; }
        .dashboard-subtabs button {
            padding: 6px 16px; border: 1px solid transparent; border-bottom: none;
            background: #f8f9fa; cursor: pointer; font-size: 13px; color: #6c757d;
            border-radius: 4px 4px 0 0; transition: background 0.15s, color 0.15s;
        }
        .dashboard-subtabs button:hover { background: #e9ecef; }
        .dashboard-subtabs button.active { background: #fff; color: #212529; border-color: #e9ecef; font-weight: 600; position: relative; }
        .dashboard-subtabs button.active::after { content: ''; position: absolute; bottom: -1px; left: 0; right: 0; height: 1px; background: #fff; }
        .dashboard-subtabgroup .dashboard-subtabgroup .dashboard-subtabs button { font-size: 12px; padding: 4px 12px; }
        .dashboard-subtab-content { display: none; flex: 1 1 0; min-height: 0; overflow: auto; padding: 0; }
        .dashboard-subtabgroup { display: flex; flex-direction: column; flex: 1 1 0; min-height: 0; }
        .dashboard-subtab-content.active { display: flex; flex-direction: column; }
        .dashboard-tab-content .js-plotly-plot,
        .dashboard-subtab-content .js-plotly-plot { width: 100% !important; flex: 1 1 0; min-height: 0; }
        .dashboard-tab-content .plotly-graph-div,
        .dashboard-subtab-content .plotly-graph-div { height: 100%; }
        """

    @staticmethod
    def _tab_js() -> str:
        return """
        function switchTab(groupId, tabId) {
            var group = document.getElementById(groupId);
            group.querySelectorAll(':scope > .dashboard-tab-content, :scope > .dashboard-subtab-content').forEach(function(el) {
                el.classList.remove('active');
            });
            var tabs = group.querySelector(':scope > .dashboard-tabs, :scope > .dashboard-subtabs');
            tabs.querySelectorAll('button').forEach(function(btn) { btn.classList.remove('active'); });
            document.getElementById(tabId).classList.add('active');
            document.querySelector('[onclick*=\"\\'' + tabId + '\\'\"]').classList.add('active');
            // Plotly figures rendered inside hidden tabs have zero dimensions;
            // resize them once the tab becomes visible.
            var target = document.getElementById(tabId);
            var plots = target.querySelectorAll('.plotly-graph-div');
            plots.forEach(function(p) { if (window.Plotly) Plotly.Plots.resize(p); });
        }
        """

    def _render_tabbed(self, content_order) -> str:
        tab_tree = self._build_tab_structure(content_order)
        renderer = _TabTreeRenderer(self, tab_tree)
        return renderer.render()

    # -- public API ----------------------------------------------------

    def save(self, save_to_path: str | os.PathLike, content_order: list[str] | None = None) -> None:
        """Save the dashboard as an HTML file.

        If any element has a `tab` set, the dashboard is rendered with a tabbed
        layout. Otherwise it uses the scrollable layout.
        The page is an HTML5 document with a viewport meta tag. In the
        scrollable layout, a figure with `height='100%'` (the default) and
        either no `layout.height` or `force_height=True` is as tall as the
        window minus 16px of page padding.

        Args:
            save_to_path: File path where the HTML dashboard will be saved.
            content_order: Optional list specifying the order of elements in the
                dashboard. If None, uses the order elements were added. Must
                contain only valid element names.

        Raises:
            KeyError: If content_order contains names not found in the dashboard.
            ValueError: If `plotly_js` is not `'embed'` or `'cdn'`.
            TypeError: If an element has an unexpected type (internal error).

        Examples:
            The elements in a custom order:

            >>> import plotly.express as px
            >>> dashboard = HTMLDashboard()
            >>> dashboard.add_plotly_figure(px.line(dataset.fetch('Country.Results.price')), name='prices')
            >>> dashboard.add_section_divider('Prices')
            'section_prices'
            >>> dashboard.save('prices.html', content_order=['section_prices', 'prices'])
        """
        dir_name = os.path.dirname(save_to_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)

        content_order = self._resolve_content_order(content_order)
        use_tabs = self._has_tabs()
        self._plotly_js_for_next_figure = self._include_plotlyjs()

        with open(save_to_path, 'w', encoding='utf-8') as f:
            f.write('<!doctype html>\n<html lang="en"><head>\n')
            f.write("<meta charset='UTF-8'>\n")
            f.write('<meta name="viewport" content="width=device-width, initial-scale=1">\n')
            f.write(f'<title>{escape(self.name)}</title>\n')
            f.write(f'<style>\n  body, * {{ font-family: {self.font_family}; }}\n')
            f.write(self._tab_css() if use_tabs else self._scroll_css())
            f.write('</style>\n')
            f.write('</head><body>\n')

            if use_tabs:
                f.write(self._render_tabbed(content_order))
                f.write(f'\n<script>\n{self._tab_js()}\n</script>\n')
            else:
                for key in content_order:
                    f.write(self._element_to_html(self.content[key]) + '\n')

            f.write('</body></html>\n')

    def show(self, width: str = '100%', height: str = '600'):
        """Display the dashboard inline in a Jupyter notebook.

        Creates a temporary HTML file and displays it using an IPython IFrame.
        This method is designed for use within Jupyter notebooks to provide
        inline dashboard previews. With `plotly_js='cdn'`, the figures need
        internet access.

        Args:
            width: CSS width specification for the iframe. Defaults to "100%".
            height: CSS height specification for the iframe. Defaults to "600".

        Note:
            This method requires IPython and is intended for Jupyter notebook use.
            The temporary file is created in the system temp directory.
        """
        import tempfile
        from pathlib import Path
        from IPython.display import IFrame, display

        tmp_dir = Path(tempfile.mkdtemp())
        html_path = tmp_dir / 'dashboard.html'
        self.save(html_path)
        display(IFrame(src=str(html_path.resolve()), width=width, height=height))


class _TabTreeRenderer:
    """Renders a nested tab tree (any depth) into HTML. Level 0 uses the top-level
    tab styling, every deeper level reuses the sub-tab styling inside a sub-tab group."""

    def __init__(self, dashboard: HTMLDashboard, tab_tree: OrderedDict):
        self._dashboard = dashboard
        self._tab_tree = tab_tree
        self._counter = 0

    def render(self) -> str:
        return '\n'.join(self._render_group(self._tab_tree, level=0, group_id='tabgroup_top'))

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f'{prefix}_{self._counter}'

    def _render_group(self, node: OrderedDict, level: int, group_id: str) -> list:
        is_top = level == 0
        bar_class = 'dashboard-tabs' if is_top else 'dashboard-subtabs'
        content_class = 'dashboard-tab-content' if is_top else 'dashboard-subtab-content'
        group_class = '' if is_top else ' class="dashboard-subtabgroup"'
        id_prefix = 'tab' if is_top else 'subtab'

        children = OrderedDict((k, v) for k, v in node.items() if k != '_elements')
        direct_elements = node['_elements'] if not is_top else []
        entries = []
        if direct_elements:
            entries.append((self._dashboard.DEFAULT_TAB, OrderedDict(_elements=direct_elements)))
        entries.extend(children.items())

        parts = [f'<div id="{group_id}"{group_class}>', f'<div class="{bar_class}">']
        tab_ids = [self._next_id(id_prefix) for _ in entries]
        for i, ((label, _), tid) in enumerate(zip(entries, tab_ids)):
            active = 'active' if i == 0 else ''
            parts.append(f'<button class="{active}" onclick="switchTab(\'{group_id}\', \'{tid}\')">{label}</button>')
        parts.append('</div>')

        for i, ((_, child), tid) in enumerate(zip(entries, tab_ids)):
            active = ' active' if i == 0 else ''
            parts.append(f'<div id="{tid}" class="{content_class}{active}">')
            parts.extend(self._render_node_content(child, level + 1))
            parts.append('</div>')

        parts.append('</div>')
        return parts

    def _render_node_content(self, node: OrderedDict, level: int) -> list:
        has_children = any(k != '_elements' for k in node)
        if has_children:
            return self._render_group(node, level, self._next_id('subtabgroup'))
        parts = []
        for key in node['_elements']:
            parts.append(self._dashboard._element_to_html(self._dashboard.content[key]))
        return parts
