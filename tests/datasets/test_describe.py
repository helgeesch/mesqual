import json
from dataclasses import dataclass

import pandas as pd
import pytest

from mesqual import StudyManager
from mesqual.datasets import (
    Availability,
    Dataset,
    DatasetCollection,
    DatasetComparison,
    DatasetConcatCollection,
    DatasetConcatCollectionOfComparisons,
    DatasetLinkCollection,
    DatasetMergeCollection,
    DatasetSumCollection,
    DatasetConfig,
    FlagUnavailableError,
    PlatformDataset,
    SourceInventory,
)
from mesqual.layer_flags import (
    AreaFlagInterpreter,
    AreaModel,
    BorderFlagInterpreter,
    BorderModel,
    PriceSpread,
    WeightedMean,
)
from mesqual.enums import ItemTypeEnum, TopologyTypeEnum
from mesqual.flag import (
    AreaFlagSpec,
    AreaLayer,
    BorderFlagSpec,
    BorderLayer,
    Dimension,
    ModelFlagSpec,
    TimeSeriesFlagSpec,
)
from mesqual.testing import DummyDataset, DummyScenario, DummySource, make_dummy_study
from mesqual.units import Units
from tests.datasets.fixtures import StaticDataset, hourly_frame, series_spec

NODE_MODEL = 'Node.Model'
GENERATOR_MODEL = 'Generator.Model'
LINE_MODEL = 'Line.Model'
CAPACITY = 'Node.Constraints.capacity_{direction}'
PRICE = 'Node.Results.price'
VOLUME = 'Node.Results.volume'
MEAN_PRICE = 'Node.Results.mean_price'
MEAN_VOLUME = 'Node.Results.mean_volume'
SOLUTION = Dimension(
    'solution',
    'Solution of the optimisation to report.',
    {'final': 'final solution', 'first': 'first solution'},
    default='final',
)
CURRENCY = Dimension('currency', 'Currency of the prices.', {'EUR': 'euro', 'USD': 'US dollar'}, default='EUR')


def node_series(name: str, description: str, **fields) -> TimeSeriesFlagSpec:
    return TimeSeriesFlagSpec(
        name=name,
        description=description,
        model_flag=NODE_MODEL,
        unit=Units.EUR_per_MWh,
        **fields,
    )


@dataclass(frozen=True)
class TableInventory(SourceInventory):
    tables: frozenset = frozenset()


NODE_MODEL_SPEC = ModelFlagSpec(
    name=NODE_MODEL,
    description='Nodes.',
    index='Node',
    topology_type=TopologyTypeEnum.Node,
    memberships={'country': 'Country.Model'},
)
LINE_MODEL_SPEC = ModelFlagSpec(
    name=LINE_MODEL,
    description='Lines.',
    index='Line',
    topology_type=TopologyTypeEnum.Edge,
    memberships={'node_from': NODE_MODEL, 'node_to': NODE_MODEL},
    edge_ends=('node_from', 'node_to'),
)
PRICE_SPEC = node_series(PRICE, 'Price at the node.', params=(SOLUTION,))
AREA_MODEL = '{area_layer}.Model'
AREA_PRICE = '{area_layer}.Results.price'
BORDER_MODEL = '{border_layer}.Model'
PRICE_SPREAD = '{border_layer}.Results.price_spread'
AREA_PRICE_SPEC = AreaFlagSpec(
    name=AREA_PRICE,
    base=PRICE_SPEC,
    aggregation=WeightedMean(weight=VOLUME),
    description='Volume-weighted mean of the node prices of the area.',
)
COUNTRY = AreaLayer('Country', 'country', NODE_MODEL, 'Country.Model', 'Countries of the nodes.')
COUNTRY_BORDER = BorderLayer('CountryBorder', 'country_border', 'Country', LINE_MODEL, 'Borders between countries.')


class CountingInterpreter(Dataset):
    """Interpreter base of the test platform. `fetch_log` lists the item type of every `_fetch`."""

    fetch_log: list = []

    def __init__(self, parent_dataset: Dataset = None):
        super().__init__(parent_dataset=parent_dataset)

    @property
    def tables(self) -> frozenset:
        return self.parent_dataset.source_inventory.tables

    def _fetch(self, flag, effective_config, **params):
        CountingInterpreter.fetch_log.append((flag, self.flag_index.get_item_type(flag)))
        return self._data(flag)

    def _data(self, flag):
        raise NotImplementedError


class ModelInterpreter(CountingInterpreter):
    flag_specs = (
        NODE_MODEL_SPEC,
        ModelFlagSpec(
            name=GENERATOR_MODEL,
            description='Generators.',
            index='Generator',
            topology_type=TopologyTypeEnum.NodeConnectedElement,
            memberships={'node': NODE_MODEL},
        ),
        LINE_MODEL_SPEC,
    )

    def _is_available(self, flag):
        return 'nodes' in self.tables

    def _data(self, flag):
        if flag == NODE_MODEL:
            return pd.DataFrame({'country': ['DE', 'FR']}, index=pd.Index(['a', 'b'], name='Node'))
        if flag == LINE_MODEL:
            return pd.DataFrame({'node_from': ['a'], 'node_to': ['b']}, index=pd.Index(['l1'], name='Line'))
        generators = pd.Index(['g1', 'g2'], name='Generator')
        return pd.DataFrame({'node': ['a', 'b'], 'fuel': ['coal', 'gas']}, index=generators)


class PriceInterpreter(CountingInterpreter):
    flag_specs = (PRICE_SPEC,)

    def _is_available(self, flag):
        return 'prices' in self.tables

    def _data(self, flag):
        return hourly_frame(periods=3)


class VolumeInterpreter(CountingInterpreter):
    """No availability check: unknown until the first fetch."""

    flag_specs = (node_series(VOLUME, 'Traded volume at the node.'),)

    def _data(self, flag):
        return hourly_frame(periods=3)


class MeanInterpreter(CountingInterpreter):
    """Layer flags: each follows the availability of its base flag."""

    BASES = {MEAN_PRICE: PRICE, MEAN_VOLUME: VOLUME}
    flag_specs = (node_series(MEAN_PRICE, 'Mean price.'), node_series(MEAN_VOLUME, 'Mean volume.'))

    def _is_available(self, flag):
        return self.parent_dataset.is_flag_available(self.BASES[flag])

    def _data(self, flag):
        return self.parent_dataset.fetch(self.BASES[flag]).mean(axis=1).to_frame('a')


class GenerationInterpreter(CountingInterpreter):
    """A flag family with one flag per fuel of the generator model."""

    FUEL = Dimension('fuel', 'Fuel.', lambda ds: {fuel: fuel for fuel in ds.fetch(GENERATOR_MODEL)['fuel']})
    flag_specs = (node_series('Node.Results.generation_{fuel}', 'Generation per node.', variants=(FUEL,)),)

    def _data(self, flag):
        return hourly_frame(periods=3)


class CapacityInterpreter(CountingInterpreter):
    """Only the 'up' direction is in the source, so the spec is partly available."""

    DIRECTION = Dimension('direction', 'Direction.', {'up': 'up', 'down': 'down'})
    flag_specs = (node_series(CAPACITY, 'Capacity at the node.', variants=(DIRECTION,)),)

    def _is_available(self, flag):
        return flag.endswith('_up')

    def _data(self, flag):
        return hourly_frame(periods=3)


class TablePlatform(PlatformDataset):
    """A platform whose source is a set of table names."""

    fetch_params = (CURRENCY,)

    def __init__(self, tables=('nodes', 'prices'), name: str = 'scenario', **kwargs):
        self.tables = frozenset(tables)
        self.inventory_builds = 0
        super().__init__(name=name, **kwargs)

    def build_source_inventory(self) -> TableInventory:
        self.inventory_builds += 1
        return TableInventory(fingerprint=','.join(sorted(self.tables)), tables=self.tables)

    def _default_attributes(self) -> dict:
        return {'tables': len(self.source_inventory.tables), 'nodes': len(self.fetch(NODE_MODEL))}

    def _fetch(self, flag, effective_config, currency=None, **params):
        return super()._fetch(flag, effective_config, **params)

    @classmethod
    def get_child_dataset_type(cls):
        return CountingInterpreter


class AreaFlags(AreaFlagInterpreter):
    flag_specs = (
        AreaFlagSpec(name=AREA_MODEL, base=NODE_MODEL_SPEC, aggregation=AreaModel(), description='One row per area.'),
        AREA_PRICE_SPEC,
    )


class BorderFlags(BorderFlagInterpreter):
    flag_specs = (
        BorderFlagSpec(
            name=BORDER_MODEL,
            base=LINE_MODEL_SPEC,
            aggregation=BorderModel(geometry=False),
            description='One row per border.',
        ),
        BorderFlagSpec(
            name=PRICE_SPREAD,
            base=AREA_PRICE_SPEC,
            aggregation=PriceSpread(),
            description='Area price at area_to minus the price at area_from.',
        ),
    )


for interpreter in (
    ModelInterpreter,
    PriceInterpreter,
    VolumeInterpreter,
    MeanInterpreter,
    GenerationInterpreter,
    CapacityInterpreter,
    AreaFlags,
    BorderFlags,
):
    TablePlatform.register_interpreter(interpreter)
TablePlatform.register_area_layer(COUNTRY)
TablePlatform.register_border_layer(COUNTRY_BORDER)


class PriceCap(PriceInterpreter):
    """Override of `PriceInterpreter` with its own description."""

    flag_specs = (node_series(PRICE, 'Price at the node, capped at 1000.', params=(SOLUTION,)),)

    def _data(self, flag):
        return super()._data(flag).clip(upper=1000)


class PriceFloor(PriceCap):
    """Override of `PriceCap` on a dataset subclass."""

    def _data(self, flag):
        return super()._data(flag).clip(lower=0)


class CapPlatform(TablePlatform):
    pass


class StudyPlatform(CapPlatform):
    pass


CapPlatform.register_interpreter(PriceCap, override_interpreter=PriceInterpreter, version=1)
StudyPlatform.register_interpreter(PriceFloor, override_interpreter=PriceCap, version=1)


class UnknownPriceCap(PriceCap):
    """Override of `PriceInterpreter` that cannot check its source cheaply."""

    def _is_available(self, flag):
        return Availability(None, 'no cheap check of the cap file')


class MissingPriceCap(PriceCap):
    """Override of `PriceInterpreter` whose cap file is missing."""

    def _is_available(self, flag):
        return Availability(False, 'no cap file')


class UnknownCapPlatform(TablePlatform):
    pass


class MissingCapPlatform(TablePlatform):
    pass


UnknownCapPlatform.register_interpreter(UnknownPriceCap, override_interpreter=PriceInterpreter, version=1)
MissingCapPlatform.register_interpreter(MissingPriceCap, override_interpreter=PriceInterpreter, version=1)


@pytest.fixture(autouse=True)
def clear_fetch_log():
    CountingInterpreter.fetch_log.clear()


def time_series_fetches() -> list:
    return [flag for flag, item_type in CountingInterpreter.fetch_log if item_type is ItemTypeEnum.TimeSeries]


class TestSourceInventory:
    def test_inventory_is_built_once_per_instance(self):
        platform = TablePlatform()
        assert platform.source_inventory is platform.source_inventory
        assert platform.source_inventory.tables == {'nodes', 'prices'}
        assert platform.inventory_builds == 1

    def test_platform_without_build_source_inventory_cannot_be_created(self):
        class NoInventory(PlatformDataset):
            @classmethod
            def get_child_dataset_type(cls):
                return CountingInterpreter

        with pytest.raises(TypeError, match='build_source_inventory'):
            NoInventory()

    def test_interpreter_uses_the_inventory_of_its_parent(self):
        assert TablePlatform(tables=('nodes', 'prices')).is_flag_available(PRICE) is True
        assert TablePlatform(tables=('nodes',)).is_flag_available(PRICE) is False

    def test_dummy_dataset_inventory_has_the_source_fingerprint(self):
        dataset = DummyDataset(DummyScenario(name='base', days=2))
        assert dataset.source_inventory.fingerprint == dataset.source.inventory.fingerprint


class TestDefaultAttributes:
    def test_default_attributes_appear_and_explicit_attributes_win(self):
        assert TablePlatform().attributes == {'tables': 2, 'nodes': 2}
        explicit = TablePlatform(attributes={'nodes': 7, 'year': 2030})
        assert explicit.attributes == {'tables': 2, 'nodes': 7, 'year': 2030}

    def test_default_attributes_fetch_no_time_series_flag(self):
        TablePlatform()
        assert CountingInterpreter.fetch_log and not time_series_fetches()


class TestAvailability:
    def test_default_hook_gives_unknown(self):
        assert TablePlatform().is_flag_available(VOLUME) is None

    def test_override_gives_true_or_false_per_flag(self):
        platform = TablePlatform(tables=('nodes',))
        assert platform.is_flag_available(NODE_MODEL) is True
        assert platform.is_flag_available(PRICE) is False

    @pytest.mark.parametrize('tables, expected', [(('nodes', 'prices'), True), (('nodes',), False)])
    def test_layer_flag_follows_its_base(self, tables, expected):
        assert TablePlatform(tables=tables).is_flag_available(MEAN_PRICE) is expected

    def test_layer_flag_of_unknown_base_is_unknown(self):
        assert TablePlatform().is_flag_available(MEAN_VOLUME) is None

    def test_unknown_flag_counts_as_available_after_first_successful_fetch(self):
        platform = TablePlatform()
        platform.fetch(VOLUME)
        assert platform.is_flag_available(VOLUME) is True
        assert platform.is_flag_available(MEAN_VOLUME) is True
        assert TablePlatform().is_flag_available(VOLUME) is None

    def test_fetch_of_unavailable_flag_raises(self):
        with pytest.raises(ValueError, match='not available'):
            TablePlatform(tables=('nodes',)).fetch(PRICE)

    def test_flag_that_is_not_accepted_is_unavailable(self):
        assert TablePlatform().is_flag_available('Node.Results.unknown') is False

    def test_family_flag_with_a_value_not_in_the_dataset_is_unavailable(self):
        info = TablePlatform().flag_info('Node.Results.generation_wind')
        assert info.available is False
        assert "'wind' is not a value of 'fuel'" in info.availability_reason
        assert TablePlatform().is_flag_available('Node.Results.generation_gas') is None

    def test_hook_may_return_an_availability_whose_reason_reaches_flag_info_and_fetch(self):
        class NoPrices(StaticDataset):
            def _is_available(self, flag):
                return Availability(False, 'no price file')

        dataset = NoPrices({PRICE: hourly_frame()})
        assert dataset.is_flag_available(PRICE) is False
        assert dataset.flag_info(PRICE).availability_reason == 'NoPrices._is_available: no price file'
        with pytest.raises(FlagUnavailableError, match='no price file'):
            dataset.fetch(PRICE)

    def test_layer_flag_of_an_unavailable_input_names_it_in_the_reason(self):
        info = TablePlatform(tables=('nodes',)).flag_info('Country.Results.price')
        assert info.available is False
        assert info.availability_reason == (
            f'AreaFlags._is_available: unavailable input {PRICE} (PriceInterpreter._is_available)'
        )

    @pytest.mark.parametrize(
        'states, expected',
        [
            ((True, True), True),
            ((True, None), None),
            ((None, False), False),
            ((), True),
        ],
    )
    def test_of_inputs_is_unavailable_then_unknown_then_available(self, states, expected):
        inputs = {f'flag_{i}': Availability(state, f'reason {i}') for i, state in enumerate(states)}
        assert Availability.of_inputs(inputs).available is expected

    def test_of_inputs_names_only_the_deciding_inputs(self):
        inputs = {'a': Availability(True, 'x'), 'b': Availability(None, 'y'), 'c': Availability(False, 'z')}
        assert Availability.of_inputs(inputs).reason == 'unavailable input c (z)'

    def test_layer_flag_with_an_unknown_weight_is_unknown_and_names_the_weight(self):
        info = TablePlatform().flag_info('Country.Results.price')
        assert info.available is None
        assert f'unknown input {VOLUME}' in info.availability_reason

    def test_family_whose_values_function_fetches_an_unavailable_flag_has_no_flags(self):
        class NoAttributes(TablePlatform):
            def _default_attributes(self):
                return {}

        platform = NoAttributes(tables=('prices',))
        row = platform.describe_flags(available_only=False).loc['Node.Results.generation_{fuel}']
        assert row['flags'] == 0 and row['availability'] == 'unavailable'
        info = platform.flag_info('Node.Results.generation_coal')
        assert info.available is False
        assert info.availability_reason.startswith("values of 'fuel' need the unavailable flag Generator.Model: ")
        with pytest.raises(FlagUnavailableError) as error:
            platform.fetch('Node.Results.generation_coal')
        assert error.value.flag == 'Node.Results.generation_coal'
        assert error.value.reason.startswith("values of 'fuel' need the unavailable flag Generator.Model: ")

    def test_family_whose_values_function_fails_for_another_reason_raises(self):
        class BrokenFuels(CountingInterpreter):
            FUEL = Dimension('fuel', 'Fuel.', lambda ds: ds.fetch(GENERATOR_MODEL)['no_such_column'])
            flag_specs = (node_series('Node.Results.output_{fuel}', 'Output per node.', variants=(FUEL,)),)

        class BrokenPlatform(TablePlatform):
            pass

        BrokenPlatform.register_interpreter(BrokenFuels)
        with pytest.raises(KeyError, match='no_such_column'):
            BrokenPlatform().describe_flags()

    def test_plain_dataset_has_no_checks(self):
        assert StaticDataset({PRICE: hourly_frame()}).is_flag_available(PRICE) is True

    def test_collection_is_the_union_of_its_children(self):
        collection = DatasetConcatCollection(
            [TablePlatform(name='a'), TablePlatform(tables=('nodes',), name='b')],
            name='study',
        )
        assert collection.is_flag_available(PRICE) is True
        assert collection.flag_info(PRICE).availability_reason == 'available in a; unavailable in b'


class TestDescribeClassFlags:
    def test_class_method_needs_no_source(self):
        specs = TablePlatform.describe_class_flags()
        assert not CountingInterpreter.fetch_log
        assert list(specs.index) == [
            NODE_MODEL,
            GENERATOR_MODEL,
            LINE_MODEL,
            PRICE,
            VOLUME,
            MEAN_PRICE,
            MEAN_VOLUME,
            'Node.Results.generation_{fuel}',
            CAPACITY,
            AREA_MODEL,
            AREA_PRICE,
            BORDER_MODEL,
            PRICE_SPREAD,
        ]
        assert specs.loc['Node.Results.generation_{fuel}', 'variants'] == 'fuel: resolved per dataset'
        assert specs.loc[PRICE, 'object'] == NODE_MODEL
        assert specs.loc[PRICE, 'quantity_type'] == 'INTENSIVE'

    def test_parameters_have_spec_and_dataset_owner(self):
        record = TablePlatform.describe_class_flags(contains='node.results.price', as_records=True)
        params = {p['name']: p for p in record[0]['params']}
        assert params['solution'] == {
            'name': 'solution',
            'owner': 'spec',
            'description': 'Solution of the optimisation to report.',
            'values': [
                {'value': 'final', 'description': 'final solution'},
                {'value': 'first', 'description': 'first solution'},
            ],
            'default': 'final',
            'default_source': 'spec',
        }
        assert params['currency']['owner'] == 'dataset'
        assert params['currency']['default_source'] == 'dataset'

    def test_filters(self):
        assert list(TablePlatform.describe_class_flags(contains='MEAN').index) == [MEAN_PRICE, MEAN_VOLUME]
        models = [NODE_MODEL, GENERATOR_MODEL, LINE_MODEL, AREA_MODEL, BORDER_MODEL]
        assert list(TablePlatform.describe_class_flags(item_type=ItemTypeEnum.Model).index) == models
        assert list(TablePlatform.describe_class_flags(item_type='Model').index) == models
        assert list(TablePlatform.describe_class_flags(layer='Country').index) == [AREA_MODEL, AREA_PRICE]
        assert list(TablePlatform.describe_class_flags(layer='CountryBorder').index) == [BORDER_MODEL, PRICE_SPREAD]

    def test_layer_spec_is_listed_once_with_its_layers(self):
        row = TablePlatform.describe_class_flags().loc[AREA_PRICE]
        assert row['item_type'] == 'TimeSeries'
        assert row['layers'] == 'Country'
        assert row['unit'] == str(Units.EUR_per_MWh)

    @pytest.mark.parametrize(
        'spec, layer_model',
        [
            (AREA_MODEL, 'Country.Model'),
            (AREA_PRICE, 'Country.Model'),
            (BORDER_MODEL, 'CountryBorder.Model'),
            (PRICE_SPREAD, 'CountryBorder.Model'),
        ],
    )
    def test_object_of_a_layer_spec_is_the_layer_model_flag(self, spec, layer_model):
        record = TablePlatform.describe_class_flags(contains=spec, as_records=True)[0]
        assert record['object'] == [layer_model]

    def test_records_are_json_serializable(self):
        json.dumps(TablePlatform.describe_class_flags(as_records=True))


class TestDescribeClassFlagsOnCollections:
    COLLECTION_TYPES = [
        DatasetCollection,
        DatasetMergeCollection,
        DatasetConcatCollection,
        DatasetLinkCollection,
        DatasetSumCollection,
        DatasetComparison,
        DatasetConcatCollectionOfComparisons,
    ]

    @pytest.mark.parametrize('collection_type', COLLECTION_TYPES)
    @pytest.mark.parametrize(
        'method',
        [
            'describe_class_flags',
            'describe_class_area_layers',
            'describe_class_border_layers',
        ],
    )
    def test_class_call_on_a_collection_raises(self, collection_type, method):
        with pytest.raises(TypeError, match=r'Use describe_flags\(\) on the instance'):
            getattr(collection_type, method)()

    def test_call_through_a_collection_instance_raises(self):
        collection = DatasetConcatCollection([TablePlatform(name='a'), TablePlatform(name='b')], name='s')
        with pytest.raises(TypeError, match='DatasetConcatCollection is a collection class'):
            collection.describe_class_flags()

    def test_call_through_a_platform_instance_gives_the_class_specs(self):
        assert TablePlatform().describe_class_flags().equals(TablePlatform.describe_class_flags())


class TestFrameShape:
    def test_class_frame_is_a_subset_of_the_instance_frame(self):
        class_frame = TablePlatform.describe_class_flags()
        instance_frame = TablePlatform().describe_flags(available_only=False)
        assert list(class_frame.index) == list(instance_frame.index)
        assert list(instance_frame.columns[: len(class_frame.columns)]) == list(class_frame.columns)
        assert list(instance_frame.columns[len(class_frame.columns) :]) == ['flags', 'availability']

    def test_layer_specs_have_the_same_object_and_layers_on_class_and_instance(self):
        class_frame = TablePlatform.describe_class_flags()
        instance_frame = TablePlatform().describe_flags(available_only=False)
        layer_flags = [AREA_MODEL, AREA_PRICE, BORDER_MODEL, PRICE_SPREAD]
        columns = ['item_type', 'object', 'layers']
        assert instance_frame.loc[layer_flags, columns].equals(class_frame.loc[layer_flags, columns])


class TestDescribeFlags:
    def test_available_only_drops_unavailable_specs_and_keeps_unknown_ones(self):
        platform = TablePlatform(tables=('nodes',))
        flags = platform.describe_flags()
        assert PRICE not in flags.index and MEAN_PRICE not in flags.index
        assert flags.loc[VOLUME, 'availability'] == 'unknown'
        assert flags.loc[NODE_MODEL, 'availability'] == 'available'
        assert platform.describe_flags(available_only=False).loc[PRICE, 'availability'] == 'unavailable'

    def test_spec_with_some_flags_unavailable_is_partial(self):
        row = TablePlatform().describe_flags().loc[CAPACITY]
        assert row['availability'] == 'partial'
        assert row['flags'] == 1

    def test_flag_family_is_listed_once_with_its_resolved_values(self):
        row = TablePlatform().describe_flags().loc['Node.Results.generation_{fuel}']
        assert row['variants'] == 'fuel: coal, gas'
        assert row['flags'] == 2

    def test_effective_defaults_with_their_source(self):
        platform = TablePlatform(config=DatasetConfig(param_defaults={'solution': 'first'}))
        record = platform.describe_flags(contains=PRICE, as_records=True)[0]
        params = {p['name']: (p['default'], p['default_source']) for p in record['params']}
        assert params == {
            'solution': ('first', 'config'),
            'currency': ('EUR', 'dataset'),
        }

    def test_filters_and_records(self):
        platform = TablePlatform()
        assert list(platform.describe_flags(layer='Country').index) == [AREA_MODEL, AREA_PRICE]
        assert set(platform.describe_flags(item_type='Model').index) == {
            NODE_MODEL,
            GENERATOR_MODEL,
            LINE_MODEL,
            AREA_MODEL,
            BORDER_MODEL,
        }
        assert set(platform.describe_flags(contains='gas').index) == {'Node.Results.generation_{fuel}'}
        json.dumps(platform.describe_flags(as_records=True))

    def test_collection_shows_the_children_that_have_a_flag(self):
        collection = DatasetConcatCollection(
            [TablePlatform(name='a'), TablePlatform(tables=('nodes',), name='b')],
            name='study',
        )
        flags = collection.describe_flags()
        assert flags.loc[PRICE, 'datasets'] == 'a'
        assert flags.loc[PRICE, 'availability'] == 'available'
        assert pd.isna(flags.loc[NODE_MODEL, 'datasets'])

    def test_layer_specs_show_their_layers_and_layer_model_flags(self):
        flags = TablePlatform().describe_flags()
        assert flags.loc[AREA_PRICE, ['object', 'layers', 'flags', 'availability']].tolist() == [
            'Country.Model',
            'Country',
            1,
            'unknown',
        ]
        assert flags.loc[PRICE_SPREAD, ['object', 'layers']].tolist() == ['CountryBorder.Model', 'CountryBorder']

    def test_layer_spec_of_an_unavailable_base_has_no_layers(self):
        row = TablePlatform(tables=('nodes',)).describe_flags(available_only=False).loc[AREA_PRICE]
        assert row['availability'] == 'unavailable'
        assert row['layers'] == '' and row['object'] == ''

    def test_accepted_flag_lookup_uses_the_same_match(self):
        assert TablePlatform().get_accepted_flags_containing_x('MEAN') == {MEAN_PRICE, MEAN_VOLUME}


class TestCollectionFetch:
    @pytest.mark.parametrize('collection_type', [DatasetConcatCollection, DatasetMergeCollection, DatasetSumCollection])
    def test_collections_skip_children_where_the_flag_is_unavailable(self, collection_type):
        collection = collection_type([TablePlatform(name='a'), TablePlatform(tables=('nodes',), name='b')], name='s')
        prices = collection.fetch(PRICE)
        if collection_type is DatasetConcatCollection:
            assert list(prices.columns.get_level_values(0).unique()) == ['a']
        assert len(prices) == 3

    @pytest.mark.parametrize('collection_type', [DatasetConcatCollection, DatasetMergeCollection, DatasetSumCollection])
    def test_flag_unavailable_on_all_accepting_children_raises_value_error(self, collection_type):
        collection = collection_type(
            [TablePlatform(tables=('nodes',), name='a'), TablePlatform(tables=('nodes',), name='b')],
            name='s',
        )
        with pytest.raises(ValueError, match=f'Flag {PRICE} is not available on dataset s: unavailable in a, b'):
            collection.fetch(PRICE)

    def test_flag_that_no_child_accepts_raises_value_error(self):
        class DataOnlyDataset(StaticDataset):
            """Narrows `accepted_flags` to the flags it has data for."""

            @property
            def accepted_flags(self):
                return set(self._data)

        children = [
            DataOnlyDataset({'x': hourly_frame(periods=2)}, name=name, specs=(series_spec('x'), series_spec('y')))
            for name in ('a', 'b')
        ]
        collection = DatasetConcatCollection(children, name='s')
        with pytest.raises(ValueError, match='not accepted'):
            collection.fetch('y')

    def test_link_collection_raises_if_its_child_has_the_flag_unavailable(self):
        link = DatasetLinkCollection([TablePlatform(tables=('nodes',), name='a')], name='link')
        with pytest.raises(ValueError, match=f'Flag {PRICE} is not available on dataset link: unavailable in a'):
            link.fetch(PRICE)

    def test_comparison_needs_the_flag_in_both_children(self):
        complete, without_prices = TablePlatform(name='a'), TablePlatform(tables=('nodes',), name='b')
        comparison = DatasetComparison(complete, without_prices)
        assert comparison.is_flag_available(PRICE) is False
        assert comparison.is_flag_available(VOLUME) is None
        assert comparison.is_flag_available(NODE_MODEL) is True
        assert PRICE not in comparison.describe_flags().index


class TestOverrideChain:
    def test_override_inherits_the_availability_check_of_its_parent_class(self):
        study = StudyPlatform(tables=('nodes',))
        assert study.is_flag_available(PRICE) is False
        assert study.flag_info(PRICE).availability_reason == 'PriceFloor._is_available'
        assert StudyPlatform().is_flag_available(PRICE) is True

    def test_platform_raises_before_the_override_runs(self):
        study = StudyPlatform(tables=('nodes',))
        CountingInterpreter.fetch_log.clear()
        with pytest.raises(ValueError, match='PriceFloor._is_available'):
            study.fetch(PRICE)
        assert not CountingInterpreter.fetch_log

    def test_collection_skips_a_child_whose_override_is_unavailable(self):
        collection = DatasetConcatCollection(
            [StudyPlatform(name='a'), StudyPlatform(tables=('nodes',), name='b')],
            name='s',
        )
        assert list(collection.fetch(PRICE).columns.get_level_values(0).unique()) == ['a']

    def test_override_spec_shows_its_description(self):
        assert StudyPlatform().flag_info(PRICE).description == 'Price at the node, capped at 1000.'

    def test_chain_of_two_links(self):
        assert StudyPlatform().flag_info(PRICE).chain == ('PriceInterpreter', 'PriceCap', 'PriceFloor')

    def test_unknown_availability_of_an_override_decides(self):
        availability = UnknownCapPlatform(tables=('nodes',)).availability(PRICE)
        assert availability.available is None
        assert availability.reason == 'UnknownPriceCap._is_available: no cheap check of the cap file'

    def test_unavailable_override_decides(self):
        platform = MissingCapPlatform()
        assert platform.availability(PRICE) == Availability(False, 'MissingPriceCap._is_available: no cap file')
        with pytest.raises(FlagUnavailableError, match='no cap file'):
            platform.fetch(PRICE)


class TestCollectionParams:
    def test_children_that_agree_show_their_default(self):
        first = DatasetConfig(param_defaults={'solution': 'first'})
        collection = DatasetConcatCollection(
            [TablePlatform(name='a', config=first), TablePlatform(name='b', config=first)],
            name='s',
        )
        assert self.solution(collection) == ('first', 'config')

    def test_children_that_differ_show_per_dataset(self):
        collection = DatasetConcatCollection(
            [
                TablePlatform(name='a', config=DatasetConfig(param_defaults={'solution': 'first'})),
                TablePlatform(name='b'),
            ],
            name='s',
        )
        assert self.solution(collection) == (None, 'per dataset')

    def test_collection_param_defaults_reach_the_children(self):
        collection = DatasetConcatCollection(
            [
                TablePlatform(name='a', config=DatasetConfig(param_defaults={'solution': 'first'})),
                TablePlatform(name='b'),
            ],
            name='s',
            config=DatasetConfig(param_defaults={'solution': 'final'}),
        )
        assert self.solution(collection) == ('final', 'config')

    @staticmethod
    def solution(collection) -> tuple:
        info = next(p for p in collection.flag_info(PRICE).params if p.name == 'solution')
        return info.default, info.default_source


class TestNoDataLoad:
    def test_describe_methods_fetch_no_time_series_flag(self):
        platform = TablePlatform()
        CountingInterpreter.fetch_log.clear()
        platform.describe_flags(available_only=False)
        platform.flag_info('Node.Results.generation_gas')
        platform.flag_info(MEAN_PRICE)
        platform.describe_datasets()
        TablePlatform.describe_class_area_layers()
        TablePlatform.describe_class_border_layers()
        assert not time_series_fetches()

    def test_describe_methods_read_only_model_tables_of_the_dummy_source(self, monkeypatch):
        read_tables = []
        original_read = DummySource.read

        def counting_read(source, table):
            read_tables.append(table)
            return original_read(source, table)

        monkeypatch.setattr(DummySource, 'read', counting_read)
        study = make_dummy_study()
        base = study.scen.get_dataset('base')
        base.describe_flags(available_only=False)
        base.flag_info('Node.Results.generation_gas')
        study.describe_study()
        assert read_tables and set(read_tables) <= set(DummySource.MODEL_TABLES)


class TestFlagInfo:
    def test_fields(self):
        platform = TablePlatform(config=DatasetConfig(param_defaults={'solution': 'first'}))
        info = platform.flag_info(PRICE)
        assert info.spec is PriceInterpreter.flag_specs[0]
        assert info.item_type is ItemTypeEnum.TimeSeries
        assert info.description == 'Price at the node.'
        assert info.unit == Units.EUR_per_MWh and info.unit_source == 'spec'
        assert info.quantity_type_source == 'unit'
        assert info.model_flag == NODE_MODEL
        assert info.available is True and info.availability_reason == 'PriceInterpreter._is_available'
        assert info.chain == ('PriceInterpreter',)
        assert [(p.name, p.owner, p.default, p.default_source) for p in info.params] == [
            ('solution', 'spec', 'first', 'config'),
            ('currency', 'dataset', 'EUR', 'dataset'),
        ]
        assert [level.name for level in info.schema.columns] == ['Node']
        record = info.as_record()
        assert record['schema']['columns'] == [{'name': 'Node', 'role': 'object'}]
        assert record['schema']['index'] == [{'name': 'time', 'role': 'time'}]
        json.dumps(record)

    def test_family_flag_description_and_variant(self):
        info = TablePlatform().flag_info('Node.Results.generation_gas')
        assert info.variant == {'fuel': 'gas'}
        assert info.description == 'Generation per node. Fuel: gas.'

    def test_dependencies_after_first_fetch(self):
        platform = TablePlatform()
        assert platform.flag_info(MEAN_PRICE).dependencies == ()
        platform.fetch(MEAN_PRICE)
        dependencies = platform.flag_info(MEAN_PRICE).dependencies
        assert [d.flag for d in dependencies] == [MEAN_PRICE]
        assert [d.flag for d in dependencies[0].iter_dependencies()][-1] == PRICE

    def test_unknown_availability_reason(self):
        info = TablePlatform().flag_info(VOLUME)
        assert info.available is None
        assert info.availability_reason == 'no availability check in VolumeInterpreter and the flag was not fetched yet'


class TestDescribeClassLayers:
    def test_area_layers(self):
        layers = TablePlatform.describe_class_area_layers()
        assert list(layers.index) == ['Country']
        assert layers.loc['Country', 'refers_to'] == f'{NODE_MODEL} -> Country.Model'
        assert layers.loc['Country', 'mapping_columns'] == 'country'
        assert layers.loc['Country', 'flags'] == 2

    def test_border_layers(self):
        records = TablePlatform.describe_class_border_layers(as_records=True)
        assert records == [
            {
                'name': 'CountryBorder',
                'index': 'country_border',
                'refers_to': f'Country via {LINE_MODEL}',
                'mapping_columns': ['node_from', 'node_to'],
                'flags': 2,
                'description': 'Borders between countries.',
            }
        ]

    def test_call_through_a_platform_instance_gives_the_class_layers(self):
        assert TablePlatform().describe_class_area_layers().equals(TablePlatform.describe_class_area_layers())

    @pytest.mark.parametrize('method', ['describe_class_area_layers', 'describe_class_border_layers'])
    def test_class_without_layers_gives_empty_frame_with_columns(self, method):
        layers = getattr(StaticDataset, method)()
        assert layers.empty
        assert list(layers.columns) == ['index', 'refers_to', 'mapping_columns', 'flags', 'description']


class TestDescribeDatasets:
    def test_nested_collections_with_descriptions_and_attributes(self):
        a = TablePlatform(name='a', attributes={'year': 2030})
        a.description = 'Reference run.'
        inner = DatasetConcatCollection([a, TablePlatform(name='b')], name='runs')
        outer = DatasetConcatCollection([inner], name='study', config=DatasetConfig(use_cache=False))
        tree = outer.describe_datasets()
        assert list(tree.index) == ['study', 'study/runs', 'study/runs/a', 'study/runs/b']
        assert tree.loc['study/runs/a', 'description'] == 'Reference run.'
        assert tree.loc['study/runs/a', 'attributes'] == 'tables: 2; nodes: 2; year: 2030'
        assert tree.loc['study', 'config'] == 'use_cache: False'
        assert tree.loc['study/runs', 'children'] == 'a, b'
        assert tree.loc['study/runs/a', 'class'] == 'TablePlatform'
        json.dumps(outer.describe_datasets(as_records=True))

    @pytest.mark.parametrize(
        'max_depth, paths',
        [
            (None, ['study', 'study/runs', 'study/runs/a', 'study/runs/b']),
            (2, ['study', 'study/runs', 'study/runs/a', 'study/runs/b']),
            (1, ['study', 'study/runs']),
            (0, ['study']),
        ],
    )
    def test_max_depth(self, max_depth, paths):
        outer = self.nested()
        tree = outer.describe_datasets(max_depth=max_depth)
        assert list(tree.index) == paths
        assert tree.loc['study', 'children'] == 'runs'
        assert [r['path'] for r in outer.describe_datasets(max_depth=max_depth, as_records=True)] == paths

    def test_max_depth_one_lists_all_children_of_the_lowest_row(self):
        assert self.nested().describe_datasets(max_depth=1).loc['study/runs', 'children'] == 'a, b'

    def test_negative_max_depth_raises(self):
        with pytest.raises(ValueError, match='max_depth'):
            self.nested().describe_datasets(max_depth=-1)

    @staticmethod
    def nested() -> DatasetConcatCollection:
        inner = DatasetConcatCollection([TablePlatform(name='a'), TablePlatform(name='b')], name='runs')
        return DatasetConcatCollection([inner], name='study')

    def test_describe_config_is_available(self):
        assert TablePlatform().describe_config().loc['use_cache', 'layer'] == 'default'


class TestDescribeStudy:
    def test_four_parts(self):
        description = make_dummy_study().describe_study()
        study = make_dummy_study()
        assert list(description) == ['scenarios', 'comparisons', 'scenario_flags', 'comparison_flags']
        pd.testing.assert_frame_equal(description['scenarios'], study.scen.describe_datasets(max_depth=1))
        pd.testing.assert_frame_equal(description['comparisons'], study.comp.describe_datasets(max_depth=1))
        pd.testing.assert_frame_equal(description['scenario_flags'], study.scen.describe_flags())
        pd.testing.assert_frame_equal(description['comparison_flags'], study.comp.describe_flags())
        assert list(description['scenarios'].index)[0] == 'scenario'
        assert all(path.count('/') <= 1 for path in description['scenarios'].index)

    def test_as_records_passes_to_all_parts(self):
        study = make_dummy_study()
        records = study.describe_study(as_records=True)
        assert all(isinstance(part, list) for part in records.values())
        json.dumps(records)

    def test_scenario_flags_show_the_scenarios_that_have_a_flag(self):
        study = StudyManager.factory_from_scenarios(
            [TablePlatform(name='a'), TablePlatform(tables=('nodes',), name='b')],
            comparisons=[('b', 'a')],
        )
        flags = {r['name']: r for r in study.describe_study(as_records=True)['scenario_flags']}
        assert flags[PRICE]['datasets'] == ['a']
        assert flags[VOLUME]['datasets'] is None
