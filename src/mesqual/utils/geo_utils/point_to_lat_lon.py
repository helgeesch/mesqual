"""Latitude and longitude of a shapely point."""

from shapely import Point


def get_lat_lon_from_point(point: Point) -> tuple[float, float]:
    """Return latitude and longitude of a point with x as longitude and y as latitude.

    Returns:
        `(lat, lon)`, the order that folium uses.
    """
    coords = list(point.coords)
    lon, lat = coords[0][0], coords[0][1]
    return lat, lon
