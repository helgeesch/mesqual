import pandas as pd
from shapely import wkt
from shapely.geometry import base
from typing import Optional

from mescal.utils.logging import get_logger

logger = get_logger(__name__)


def convert_wkt_series(series: pd.Series) -> pd.Series:
    """
        Convert a pandas Series of WKT (Well-Known Text) strings to Shapely geometry objects.

        Handles None and empty string values gracefully by returning None.
        If a value cannot be parsed, it is left unchanged and a warning is printed if any parsing fails.
    """

    def safe_load(value: Optional[str]) -> Optional[base.BaseGeometry]:
        if value is None or value == "":
            return None
        try:
            return wkt.loads(value)
        except Exception:
            return value

    result = series.apply(safe_load)

    if not result.apply(lambda x: x is None or isinstance(x, base.BaseGeometry)).all():
        logger.warning("Not all values could be converted to geometries.")

    return result
