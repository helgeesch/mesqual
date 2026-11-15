"""Popup with the coordinates of a click on a folium map."""

import folium


class CustomLatLngPopup(folium.LatLngPopup):
    """Popup that shows the latitude and longitude of the point that you click on the map.

    The popup also shows the point as WKT text, for example `POINT(9.1234 48.5678)`,
    to copy it into a geometry.

    Examples:
        >>> import folium
        >>> folium_map = folium.Map()
        >>> folium_map = folium_map.add_child(CustomLatLngPopup())
        >>> '"<br><br>POINT("' in folium_map.get_root().render()
        True
    """

    _template = folium.features.Template("""
        {% macro script(this, kwargs) %}
            var {{this.get_name()}} = L.popup();
            function latLngPop(e) {
                {{this.get_name()}}
                    .setLatLng(e.latlng)
                    .setContent("Latitude: " + e.latlng.lat.toFixed(4) +
                                "<br>Longitude: " + e.latlng.lng.toFixed(4) +
                                "<br><br>POINT("+e.latlng.lng.toFixed(4)+" "+e.latlng.lat.toFixed(4)+")")
                    .openOn({{this._parent.get_name()}});
                }
            {{this._parent.get_name()}}.on('click', latLngPop);
        {% endmacro %}
    """)
