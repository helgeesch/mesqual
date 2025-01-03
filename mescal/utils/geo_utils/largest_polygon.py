from shapely import MultiPolygon, Polygon


def get_largest_sub_polygon(multi_polygon: MultiPolygon) -> Polygon:

    if isinstance(multi_polygon, Polygon):
        return multi_polygon

    largest_sub_polygon = max(multi_polygon.geoms, key=lambda polygon: polygon.area)

    return largest_sub_polygon
