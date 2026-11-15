"""How many base objects and edges a layer leaves out."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mesqual.flag.layers import AreaLayer, BorderLayer
from mesqual.energy_data_handling.area_accounting import AreaMembership
from mesqual.layer_flags.aggregations import layer_edge_areas

if TYPE_CHECKING:
    from mesqual.datasets.dataset import Dataset


def objects_without_area(dataset: Dataset, layer: AreaLayer) -> int:
    """Return the number of objects of the base model of `layer` that belong to no area.

    An object with an empty value in the layer membership column belongs to
    no area. It fetches the base model only.

    Args:
        dataset: A dataset whose model results have no dataset level, for
            example a platform dataset or a merge collection.
        layer: The area layer.

    Raises:
        TypeError: The base model of `dataset` has a dataset level, for
            example on a concat collection.

    Examples:
        >>> from mesqual.testing import DummyScenario, make_dummy_dataset
        >>> dataset = make_dummy_dataset(DummyScenario(name='base', node_without_country=True))
        >>> objects_without_area(dataset, dataset.flag_index.get_layer('Country.Results.price'))
        1
    """
    _check_no_dataset_level(dataset, layer.base_model)
    areas = dataset.fetch(layer.base_model)[dataset.flag_index.get_mapping_column(layer)]
    return int((~AreaMembership.has_area(areas)).sum())


def edges_on_no_border(dataset: Dataset, layer: BorderLayer) -> int:
    """Return the number of edges of the edge model of `layer` that cross no border.

    An edge with both ends in one area, or with an end without area, is on
    no border. It fetches model flags only.

    Args:
        dataset: A dataset whose model results have no dataset level, for
            example a platform dataset or a merge collection.
        layer: The border layer.

    Raises:
        TypeError: The edge model or the base model of the area layer of
            `dataset` has a dataset level, for example on a concat collection.

    Examples:
        >>> from mesqual.testing import DummyScenario, make_dummy_dataset
        >>> dataset = make_dummy_dataset(DummyScenario(name='base', node_without_country=True))
        >>> edges_on_no_border(dataset, dataset.flag_index.get_layer('CountryBorder.Model'))
        2
    """
    _check_no_dataset_level(dataset, layer.edge_model)
    _check_no_dataset_level(dataset, dataset.flag_index.layers.area_layer_of(layer).base_model)
    areas = layer_edge_areas(dataset, layer, None, {})
    return int((~AreaMembership.crosses_border(areas)).sum())


def _check_no_dataset_level(dataset: Dataset, flag: Any) -> None:
    schema = dataset._result_schema(flag)
    levels = [level.name for level in (*schema.index, *schema.columns) if level.role == 'dataset']
    if levels:
        raise TypeError(
            f'{type(dataset).__name__} {dataset.name!r}: {flag} has the dataset level {levels}. Call it for each child.'
        )
