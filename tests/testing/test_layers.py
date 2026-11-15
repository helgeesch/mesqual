"""Area and border layers of the synthetic platform, and its describe methods against the fetched results."""

import pandas as pd
import pytest

from mesqual.datasets import DatasetConcatCollection
from mesqual.enums import ItemTypeEnum
from mesqual.layer_flags import edges_on_no_border, objects_without_area
from mesqual.testing import DummyDataset, DummyPeriodDataset, DummyScenario, make_dummy_dataset

BASE = DummyScenario(name='base', start='2024-03-30', days=2, resolution_change=('2024-03-31', '15min'))
LINES_INSIDE_ONE_COUNTRY = 2
"""L3 (N3-N1) and L5 (N4-N2) have both ends in one country; L1, L2 and L4 cross the border."""


def test_dataset_registers_the_country_layers():
    [country] = DummyDataset.get_area_layers()
    [border] = DummyDataset.get_border_layers()
    assert (country.name, country.base_model, country.layer_model) == ('Country', 'Node.Model', 'Country.Model')
    assert (border.name, border.area_layer, border.edge_model) == ('CountryBorder', 'Country', 'Line.Model')
    [area_row] = DummyDataset.describe_class_area_layers(as_records=True)
    [border_row] = DummyDataset.describe_class_border_layers(as_records=True)
    assert (area_row['mapping_columns'], area_row['flags']) == (['country'], 3)
    assert (border_row['mapping_columns'], border_row['flags']) == (['node_from', 'node_to'], 4)


def test_country_demand_is_the_sum_of_the_node_demand_per_country():
    dataset = DummyDataset(BASE)
    country_of_node = dataset.fetch('Node.Model')['country']
    expected = dataset.fetch('Node.Results.demand_total').T.groupby(country_of_node).sum().T
    pd.testing.assert_frame_equal(dataset.fetch('Country.Results.demand_total'), expected, check_names=False)


def test_border_capacity_sums_the_line_capacities_in_border_direction():
    """L1 (N1-N2) and L4 (N3-N4) go from C1 to C2, L2 (N2-N3) goes from C2 to C1."""
    dataset = DummyDataset(BASE)
    up, down = dataset.fetch('Line.Constraints.capacity_up'), dataset.fetch('Line.Constraints.capacity_down')
    border_up = dataset.fetch('CountryBorder.Constraints.capacity_up')
    border_down = dataset.fetch('CountryBorder.Constraints.capacity_down')
    pd.testing.assert_series_equal(border_up['C1 - C2'], up['L1'] + up['L4'] + down['L2'], check_names=False)
    pd.testing.assert_series_equal(border_down['C1 - C2'], down['L1'] + down['L4'] + up['L2'], check_names=False)
    pd.testing.assert_series_equal(border_up['C2 - C1'], border_down['C1 - C2'], check_names=False)


def test_price_spread_is_the_price_at_the_end_minus_the_price_at_the_start():
    dataset = DummyDataset(BASE)
    price = dataset.fetch('Country.Results.price')
    spread = dataset.fetch('CountryBorder.Results.price_spread')
    pd.testing.assert_series_equal(spread['C1 - C2'], price['C2'] - price['C1'], check_names=False)
    assert dataset.fetch('CountryBorder.Model')['geo_line_string'].notna().all()


class TestDescribeMatchesFetch:
    """`describe_flags()` and `flag_info()` say what `fetch` returns, for every flag of every spec."""

    @staticmethod
    def flags(dataset) -> list[str]:
        return [flag for spec in dataset.flag_index.specs for flag in spec.expand(dataset)]

    @pytest.mark.parametrize('dataset_type', [DummyDataset, DummyPeriodDataset])
    def test_flag_info_gives_the_levels_unit_and_objects_of_the_fetched_frame(self, dataset_type):
        dataset = dataset_type(BASE)
        for flag in self.flags(dataset):
            data, info = dataset.fetch(flag), dataset.flag_info(flag)
            assert info.available is True, flag
            assert list(data.index.names) == [level.name for level in info.schema.index], flag
            assert list(data.columns.names) == [level.name for level in info.schema.columns], flag
            if info.item_type == ItemTypeEnum.Model:
                continue
            assert info.unit == dataset.flag_index.get_unit(flag)
            assert set(data.columns) <= set(dataset.fetch(info.model_flag).index), flag

    def test_describe_flags_lists_every_spec_with_its_fetched_flags(self):
        dataset = DummyDataset(BASE)
        rows = dataset.describe_flags()
        assert list(rows.index) == [spec.name for spec in dataset.flag_index.specs]
        assert (rows['availability'] == 'available').all()
        for spec in dataset.flag_index.specs:
            assert rows.loc[spec.name, 'flags'] == len(spec.expand(dataset)), spec.name
        records = {row['name']: row for row in dataset.describe_flags(as_records=True)}
        assert records['{area_layer}.Results.price']['layers'] == ['Country']
        assert records['{border_layer}.Results.flow_{side}_{direction}']['object'] == ['CountryBorder.Model']

    def test_describe_flags_of_a_split_scenario_equals_the_unsplit_one(self):
        merged = make_dummy_dataset(BASE.split('2024-03-31')).describe_flags().drop(columns='datasets')
        pd.testing.assert_frame_equal(merged, DummyDataset(BASE).describe_flags())


class TestLayerCoverage:
    """`objects_without_area` and `edges_on_no_border` on the country layers."""

    @staticmethod
    def counts(dataset):
        index = dataset.flag_index
        return (
            objects_without_area(dataset, index.get_layer('Country.Results.price')),
            edges_on_no_border(dataset, index.get_layer('CountryBorder.Model')),
        )

    def test_node_without_country_and_lines_inside_one_country(self):
        assert self.counts(DummyDataset(BASE)) == (0, LINES_INSIDE_ONE_COUNTRY)
        assert self.counts(DummyDataset(BASE.replace(node_without_country=True))) == (1, LINES_INSIDE_ONE_COUNTRY)

    def test_merge_of_parts_counts_like_its_parts(self):
        merged = make_dummy_dataset(BASE.replace(node_without_country=True).split('2024-03-31'))
        assert self.counts(merged) == (1, LINES_INSIDE_ONE_COUNTRY)

    def test_flag_info_has_no_counts(self):
        record = DummyDataset(BASE).flag_info('Country.Results.price').as_record()
        assert 'objects_without_area' not in record and 'edges_on_no_border' not in record

    def test_functions_raise_on_a_dataset_level(self):
        collection = DatasetConcatCollection([DummyDataset(BASE.replace(name='a'))], name='c')
        with pytest.raises(TypeError, match='dataset level'):
            objects_without_area(collection, collection.flag_index.get_layer('Country.Results.price'))
        with pytest.raises(TypeError, match='dataset level'):
            edges_on_no_border(collection, collection.flag_index.get_layer('CountryBorder.Model'))
