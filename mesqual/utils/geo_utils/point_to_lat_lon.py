from shapely import Point


def get_lat_lon_from_point(point: Point) -> tuple[float, float]:
    coords = list(point.coords)
    lon, lat = coords[0][0], coords[0][1]
    return lat, lon
