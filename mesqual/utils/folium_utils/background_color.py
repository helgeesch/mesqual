import folium


def set_background_color_of_map(folium_map: folium.Map, color: str) -> folium.Map:
    css = f"""
    <style>
        .leaflet-container {{
            background-color: {color} !important;
        }}
    </style>
    """
    folium_map.get_root().header.add_child(folium.Element(css))
    return folium_map
