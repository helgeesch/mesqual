"""Background color of a folium map."""

import folium


def set_background_color_of_map(folium_map: folium.Map, color: str) -> folium.Map:
    """Set the background color of the map, visible where no tile or feature covers it.

    Args:
        folium_map: The map. The function adds a style element to its header.
        color: A CSS color, for example `'#ffffff'` or `'white'`.

    Returns:
        The same map.
    """
    css = f"""
    <style>
        .leaflet-container {{
            background-color: {color} !important;
        }}
    </style>
    """
    folium_map.get_root().header.add_child(folium.Element(css))
    return folium_map
