"""Country shapes on folium maps."""

from importlib.resources import as_file, files

import pandas as pd
import geopandas as gpd
import folium


class MapCountryPlotter:
    """Draws country shapes on folium maps, for example as a background for a map of results.

    A country id is compared exactly, case included, with the values of the
    columns in `GEOJSON_ID_COLUMNS`, in this order. The rows of the first
    column with a match are the country. So an id finds a country only if the
    file has it in one of these columns. In the Natural Earth file some
    countries have `-99` as ISO code: `'FRA'` finds nothing, `'FR'` finds
    France through `FIPS_10_`. The id `'NO'` is looked up as `'NOR'`, the
    sovereign code of Norway.

    Args:
        geojson_file_path: GeoJSON file with the country shapes, coordinates
            in longitude and latitude (EPSG:4326). The CRS of the file is
            overwritten with EPSG:4326, not converted. Default: the Natural
            Earth countries that ship with mesqual.
    """

    GEOJSON_ID_COLUMNS = ['ISO_A2', 'ISO_A3', 'SOV_A3', 'FIPS_10_']
    DEFAULT_STYLE = {'fillColor': '#C2C2C2', 'color': 'white', 'weight': 0.5, 'fillOpacity': 1}
    _AUTO_REPLACEMENTS = {
        'NO': 'NOR',
    }

    def __init__(self, geojson_file_path: str = None):
        if geojson_file_path is None:
            gdf = _load_countries_geojson()
        else:
            gdf = gpd.read_file(geojson_file_path)
        gdf.loc[:, 'geometry'] = gdf.buffer(0)
        gdf.set_crs('EPSG:4326', allow_override=True, inplace=True)
        self._countries_gdf = gdf

    def add_countries_to_feature_group(
        self, fg: folium.FeatureGroup, countries: list[str], style: dict = None
    ) -> folium.FeatureGroup:
        """Add the shapes of the given countries to the feature group, as one `folium.GeoJson`.

        An id without match is skipped. If no id matches, nothing is added.

        Args:
            fg: The feature group.
            countries: Country ids, looked up as described in the class docstring.
            style: Style keys that replace those of `DEFAULT_STYLE`.

        Returns:
            The same feature group.
        """
        countries_gdf = self._find_countries(countries)
        if countries_gdf.empty:
            return fg

        style_dict = self._get_updated_style_dict(style)
        folium.GeoJson(countries_gdf, style_function=lambda x: style_dict).add_to(fg)
        return fg

    def _get_updated_style_dict(self, style: dict = None) -> dict:
        style_dict = self.DEFAULT_STYLE.copy()
        if style:
            style_dict.update(style)
        return style_dict

    def add_all_countries_except(
        self, fg: folium.FeatureGroup, excluded_countries: list[str], style: dict = None
    ) -> folium.FeatureGroup:
        """Add the shapes of all countries of the file except the given ones, as one `folium.GeoJson`.

        Args:
            fg: The feature group.
            excluded_countries: Ids of the countries to leave out. An id without match is skipped.
            style: Style keys that replace those of `DEFAULT_STYLE`.

        Returns:
            The same feature group.
        """
        excluded_gdf = self._find_countries(excluded_countries)
        all_countries = self._countries_gdf.copy()

        if not excluded_gdf.empty:
            all_countries = all_countries[~all_countries.index.isin(excluded_gdf.index)]

        style_dict = self._get_updated_style_dict(style)
        folium.GeoJson(all_countries, style_function=lambda x: style_dict).add_to(fg)
        return fg

    def get_geojson_for_country(self, country_id: str) -> gpd.GeoDataFrame:
        """Return the rows of the file that match a country id, as a `GeoDataFrame`.

        Args:
            country_id: The id, looked up as described in the class docstring.

        Returns:
            The matching rows of the first id column with a match, an empty
                `GeoDataFrame` if no column matches.
        """
        _country_id = self._AUTO_REPLACEMENTS.get(country_id, country_id)

        for column in self.GEOJSON_ID_COLUMNS:
            if column not in self._countries_gdf.columns:
                continue
            match = self._countries_gdf[self._countries_gdf[column] == _country_id]
            if not match.empty:
                return match
        return gpd.GeoDataFrame()

    def _find_countries(self, countries: list[str]) -> gpd.GeoDataFrame:
        matches = gpd.GeoDataFrame()
        for country in countries:
            country_match = self.get_geojson_for_country(country)
            if not country_match.empty:
                matches = pd.concat([matches, country_match])
        return matches


def _load_countries_geojson() -> gpd.GeoDataFrame:
    with as_file(files('mesqual.data') / 'countries.geojson') as path:
        return gpd.read_file(path)
