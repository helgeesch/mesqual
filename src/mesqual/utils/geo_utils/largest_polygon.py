"""The largest polygon of a multipolygon."""

from shapely import MultiPolygon, Polygon


def get_largest_sub_polygon(multi_polygon: MultiPolygon) -> Polygon:
    """Return the polygon with the largest area of a multipolygon.

    Args:
        multi_polygon: A `MultiPolygon`, or a `Polygon`, which is returned as it is.
    """

    if isinstance(multi_polygon, Polygon):
        return multi_polygon

    largest_sub_polygon = max(multi_polygon.geoms, key=lambda polygon: polygon.area)

    return largest_sub_polygon
