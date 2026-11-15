from collections import Counter
from itertools import product as cartesian_product

import pandas as pd
import pytest

from mesqual.datasets import (
    INHERIT,
    DatasetComparison,
    DatasetConcatCollection,
    DatasetLinkCollection,
    DatasetMergeCollection,
)
from mesqual.datasets import FlagUnavailableError
from mesqual.enums import ItemTypeEnum, TopologyTypeEnum
from mesqual.flag import AreaFlagSpec, LayerFlagSpec, TimeSeriesFlagSpec
from mesqual.layer_flags import AreaFlagInterpreter, WeightedMean
from mesqual.testing import (
    DummyConfig,
    DummyDataset,
    DummyPeriodDataset,
    DummyScenario,
    make_dummy_dataset,
)
from mesqual.testing.dataset import (
    DummyAreaFlags,
    DummyDerivedInterpreter,
    DummyInterpreter,
    DummySource,
    DummyTimeSeriesInterpreter,
)
from mesqual.units import Units

BASE = DummyScenario(name='base', start='2024-03-30', days=3, resolution_change=('2024-03-31', '15min'))
CONSTANT = DummyScenario(name='constant', profile='constant', tz=None, days=1)
FLOW_FLAGS = [f'Line.Results.flow_{s}_{d}' for s in ('sent', 'received') for d in ('up', 'down', 'net')]


class DoubledVariant(DummyTimeSeriesInterpreter):
    """Override of the time series interpreter that doubles one flag, `DOUBLED`."""

    DOUBLED: str

    def _fetch(self, flag, effective_config, **params):
        result = super()._fetch(flag, effective_config, **params)
        return 2 * result if flag == self.DOUBLED else result


class OverrideDataset(DummyDataset):
    pass


@OverrideDataset.register_interpreter(override_interpreter=DummyTimeSeriesInterpreter, version=1)
class DoubledInFlow(DoubledVariant):
    DOUBLED = 'Line.Results.flow_sent_net'


class DoubledHourlyDemandDataset(DummyDataset):
    pass


@DoubledHourlyDemandDataset.register_interpreter(override_interpreter=DummyTimeSeriesInterpreter, version=1)
class DoubledHourlyDemand(DoubledVariant):
    DOUBLED = 'Demand.Results.demand_hourly'


class DoubledTotalDemandDataset(DummyDataset):
    pass


@DoubledTotalDemandDataset.register_interpreter(override_interpreter=DummyTimeSeriesInterpreter, version=1)
class DoubledTotalDemand(DoubledVariant):
    DOUBLED = 'Demand.Results.demand_total'


class DoubledCapacityDataset(DummyDataset):
    pass


@DoubledCapacityDataset.register_interpreter(override_interpreter=DummyTimeSeriesInterpreter, version=1)
class DoubledCapacity(DoubledVariant):
    DOUBLED = 'Line.Constraints.capacity_up'


def _parameter_combinations(spec) -> list[dict]:
    names = [d.name for d in spec.params]
    return [dict(zip(names, values)) for values in cartesian_product(*(list(d.values) for d in spec.params))]


@pytest.mark.parametrize('dataset_type', [DummyDataset, DummyPeriodDataset])
def test_every_flag_of_every_spec_fetches_with_every_parameter_value(dataset_type):
    dataset = dataset_type(BASE)
    fetched = set()
    for spec in dataset.flag_index.specs:
        for flag in spec.expand(dataset):
            for params in _parameter_combinations(spec):
                data = dataset.fetch(flag, **params)
                assert isinstance(data, pd.DataFrame) and not data.empty, (flag, params)
            fetched.add(flag)
    assert dataset.accepted_flags <= fetched
    assert {'Node.Results.generation_solar', 'Country.Results.generation_solar'} <= fetched
    assert {'Country.Results.price', 'CountryBorder.Model', 'CountryBorder.Results.price_spread'} <= fetched
    assert {'CountryBorder.Constraints.capacity_up', 'CountryBorder.Constraints.capacity_down'} <= fetched


def test_dataset_accepts_the_flags_of_the_synthetic_platform():
    flags = DummyDataset(BASE).accepted_flags
    assert {'Country.Model', 'Node.Model', 'Line.Model', 'Generator.Model', 'Demand.Model'} <= flags
    assert {'Node.Results.price', 'Demand.Results.demand_total', 'Generator.Results.generation'} <= flags
    assert set(FLOW_FLAGS) | {'Line.Constraints.capacity_up', 'Line.Constraints.capacity_down'} <= flags
    assert {'Line.Results.loading', 'Node.Results.demand_total'} <= flags
    assert DummyDataset(BASE).flag_is_accepted('Node.Results.generation_gas')


SOLUTION_FLAGS = [
    'Node.Results.price',
    'Demand.Results.demand_total',
    'Node.Results.demand_hourly',
    'Country.Results.price',
    'Country.Results.demand_total',
    'CountryBorder.Results.price_spread',
]
SOLUTION_SPECS = {
    'Node.Results.price',
    'Demand.Results.demand_{product}',
    'Node.Results.demand_{product}',
    '{area_layer}.Results.price',
    '{area_layer}.Results.demand_{product}',
    '{border_layer}.Results.price_spread',
}


def test_unknown_solution_raises_value_error():
    with pytest.raises(ValueError, match="'best' is not a value of 'solution'"):
        DummyDataset(BASE).fetch('Demand.Results.demand_total', solution='best')


@pytest.mark.parametrize('flag', ['Line.Results.flow_sent_up', 'Generator.Results.generation', 'Line.Results.loading'])
def test_solution_on_a_flag_without_the_parameter_raises(flag):
    with pytest.raises(ValueError, match='solution'):
        DummyDataset(BASE).fetch(flag, solution='first')


@pytest.mark.parametrize('solution', ['first', 'final'])
@pytest.mark.parametrize('level', ['Demand', 'Node', 'Country'])
def test_demand_of_hourly_and_quarter_hourly_orders_adds_up_to_the_total(level, solution):
    dataset = DummyDataset(BASE)
    total = dataset.fetch(f'{level}.Results.demand_total', solution=solution)
    hourly = dataset.fetch(f'{level}.Results.demand_hourly', solution=solution)
    quarter_hourly = dataset.fetch(f'{level}.Results.demand_quarter_hourly', solution=solution)
    pd.testing.assert_frame_equal(hourly + quarter_hourly, total)
    pd.testing.assert_frame_equal(hourly, total * 0.75)


@pytest.mark.parametrize('flag', SOLUTION_FLAGS)
def test_first_solution_differs_from_the_final_one_and_final_is_the_default(flag):
    dataset = DummyDataset(BASE)
    final = dataset.fetch(flag)
    pd.testing.assert_frame_equal(dataset.fetch(flag, solution='final'), final)
    assert not dataset.fetch(flag, solution='first').equals(final)


def test_first_solution_of_a_node_does_not_depend_on_removed_nodes():
    less = DummyDataset(BASE.replace(name='less', removed_objects=('N2',)))
    for solution in ('first', 'final'):
        deltas = (
            less.fetch('Node.Results.price', solution=solution)
            - DummyDataset(BASE).fetch(
                'Node.Results.price',
                solution=solution,
            )[['N1', 'N3', 'N4']]
        )
        assert (deltas == 0).all().all(), solution


def test_first_solution_deviates_by_a_fixed_amount():
    dataset = DummyDataset(BASE)
    price = dataset.fetch('Node.Results.price')
    steps = pd.Series([-1.0, -2.0, -3.0, -4.0], index=price.columns)
    pd.testing.assert_frame_equal(dataset.fetch('Node.Results.price', solution='first'), price + steps)
    country_price = dataset.fetch('Country.Results.price', solution='first')
    spread = dataset.fetch('CountryBorder.Results.price_spread', solution='first')
    pd.testing.assert_series_equal(spread['C1 - C2'], country_price['C2'] - country_price['C1'], check_names=False)
    demand = dataset.fetch('Demand.Results.demand_total')
    pd.testing.assert_frame_equal(dataset.fetch('Demand.Results.demand_total', solution='first'), demand * 0.98)


def test_country_flags_of_the_first_solution_build_on_the_node_flags_of_the_first_solution():
    dataset = DummyDataset(BASE)
    country_of_node = dataset.fetch('Node.Model')['country']
    price = dataset.fetch('Node.Results.price', solution='first')
    demand = dataset.fetch('Node.Results.demand_total', solution='first')
    expected_demand = demand.T.groupby(country_of_node).sum().T
    expected_price = (price * demand).T.groupby(country_of_node).sum().T / expected_demand
    pd.testing.assert_frame_equal(
        dataset.fetch('Country.Results.demand_total', solution='first'),
        expected_demand,
        check_names=False,
    )
    pd.testing.assert_frame_equal(
        dataset.fetch('Country.Results.price', solution='first'),
        expected_price,
        check_names=False,
    )


def test_solution_in_param_defaults_reaches_only_the_declaring_flags():
    first = DummyDataset(BASE, config=DummyConfig(param_defaults={'solution': 'first'}))
    default = DummyDataset(BASE)
    for flag in SOLUTION_FLAGS:
        pd.testing.assert_frame_equal(first.fetch(flag), default.fetch(flag, solution='first'))
    for flag in ('Line.Results.flow_sent_up', 'Generator.Results.generation'):
        pd.testing.assert_frame_equal(first.fetch(flag), default.fetch(flag))


def test_describe_shows_solution_only_on_the_declaring_specs():
    dataset = DummyDataset(BASE)
    params = dataset.describe_flags()['params']
    assert set(params.index[params.str.contains('solution')]) == SOLUTION_SPECS
    for flag in ('Country.Results.price', 'Line.Results.flow_sent_up'):
        names = {p.name for p in dataset.flag_info(flag).params}
        assert ('solution' in names) == (flag == 'Country.Results.price')


def test_line_flow_sides_and_directions_add_up():
    dataset = DummyDataset(BASE)
    up, down, net = (dataset.fetch(f'Line.Results.flow_sent_{d}') for d in ('up', 'down', 'net'))
    pd.testing.assert_frame_equal(net, up - down)
    assert (up >= 0).all().all() and (down >= 0).all().all()
    received_up = dataset.fetch('Line.Results.flow_received_up')
    pd.testing.assert_frame_equal(received_up, up * 0.98)


def test_an_override_of_the_source_flow_reaches_the_derived_flow_variants():
    dataset, overridden = DummyDataset(BASE), OverrideDataset(BASE)
    for flag in ('Line.Results.flow_sent_up', 'Line.Results.flow_received_down', 'Line.Results.flow_received_net'):
        pd.testing.assert_frame_equal(overridden.fetch(flag), 2 * dataset.fetch(flag))


def test_override_of_the_hourly_demand_reaches_only_the_hourly_node_demand():
    overridden, base = DoubledHourlyDemandDataset(BASE), DummyDataset(BASE)
    hourly = 'Node.Results.demand_hourly'
    pd.testing.assert_frame_equal(overridden.fetch(hourly), 2 * base.fetch(hourly))
    total = 'Node.Results.demand_total'
    pd.testing.assert_frame_equal(overridden.fetch(total), base.fetch(total))


def test_override_of_the_total_demand_reaches_the_hourly_demand():
    overridden, base = DoubledTotalDemandDataset(BASE), DummyDataset(BASE)
    hourly = 'Demand.Results.demand_hourly'
    pd.testing.assert_frame_equal(overridden.fetch(hourly), 2 * base.fetch(hourly))


def test_line_capacity_down_follows_an_override_of_capacity_up():
    pd.testing.assert_frame_equal(
        DoubledCapacityDataset(BASE).fetch('Line.Constraints.capacity_down'),
        2 * DummyDataset(BASE).fetch('Line.Constraints.capacity_down'),
    )


def test_interpreter_with_a_spec_in_both_maps_or_in_neither_raises():
    spec = DummyTimeSeriesInterpreter.PRICE_SPEC
    with pytest.raises(ValueError, match='in SPEC_TABLE and in INPUT_FLAGS'):
        type(
            'Both',
            (DummyInterpreter,),
            {
                'flag_specs': (spec,),
                'SPEC_TABLE': {spec.name: 'node_price'},
                'INPUT_FLAGS': {spec.name: ()},
            },
        )
    with pytest.raises(ValueError, match='in neither'):
        type('Neither', (DummyInterpreter,), {'flag_specs': (spec,)})


def test_input_flag_with_a_placeholder_that_is_no_variant_raises():
    spec = DummyDerivedInterpreter.NODE_DEMAND_SPEC
    with pytest.raises(ValueError, match=r"placeholders \['period'\] that are no variants"):
        type(
            'Unknown',
            (DummyInterpreter,),
            {
                'flag_specs': (spec,),
                'INPUT_FLAGS': {spec.name: ('Demand.Results.demand_{period}',)},
            },
        )


def test_line_loading_is_flow_over_capacity():
    dataset = DummyDataset(CONSTANT)
    assert (dataset.fetch('Line.Results.loading') == 0.5).all().all()


def test_line_loading_of_a_negative_flow_uses_the_down_capacity():
    dataset = DummyDataset(BASE)
    flow = dataset.fetch('Line.Results.flow_sent_net')
    capacity_up = dataset.fetch('Line.Constraints.capacity_up')
    capacity_down = dataset.fetch('Line.Constraints.capacity_down')
    loading = dataset.fetch('Line.Results.loading')
    pd.testing.assert_frame_equal(capacity_down, capacity_up * 0.9)
    negative = flow < 0
    assert negative.any().any()
    pd.testing.assert_frame_equal(loading[negative], (-flow / capacity_down)[negative])
    pd.testing.assert_frame_equal(loading[~negative], (flow / capacity_up)[~negative])


def test_line_loading_uses_an_override_of_the_line_flow():
    loading = OverrideDataset(CONSTANT).fetch('Line.Results.loading')
    assert (loading == 1.0).all().all()


def test_generation_family_groups_generators_per_node_and_fuel():
    dataset = DummyDataset(BASE.replace(n_generators=6))
    generators = dataset.fetch('Generator.Model')
    generation = dataset.fetch('Generator.Results.generation')
    gas = dataset.fetch('Node.Results.generation_gas')
    gas_generators = generators.index[generators.fuel == 'gas']
    expected = generation[gas_generators].T.groupby(generators.loc[gas_generators, 'node']).sum().T
    pd.testing.assert_frame_equal(gas, expected.rename_axis(columns='Node'))
    family = DummyDerivedInterpreter.GENERATION_BY_FUEL_SPEC.expand(dataset)
    assert sorted(family) == sorted(f'Node.Results.generation_{f}' for f in generators.fuel.unique())


def test_node_demand_is_the_sum_of_the_demands_at_the_node():
    dataset = DummyDataset(BASE)
    demands = dataset.fetch('Demand.Model')
    demand = dataset.fetch('Demand.Results.demand_total')
    node_demand = dataset.fetch('Node.Results.demand_total')
    assert node_demand.columns.tolist() == ['N1', 'N2', 'N3', 'N4']
    for node in node_demand.columns:
        expected = demand[demands.index[demands.node == node]].sum(axis=1)
        pd.testing.assert_series_equal(node_demand[node], expected, check_names=False)


def test_node_demand_of_a_constant_scenario_is_the_sum_of_the_base_demands():
    dataset = DummyDataset(CONSTANT)
    base_demand = dataset.fetch('Demand.Model').groupby('node')['base_demand'].sum()
    assert (dataset.fetch('Node.Results.demand_total') == base_demand).all().all()


class ClassConfigDataset(DummyDataset):
    pass


def test_price_offset_of_the_config_subclass_applies_on_every_layer():
    ClassConfigDataset.set_class_config(DummyConfig(price_offset=5.0))
    base_price = DummyDataset(CONSTANT).fetch('Node.Model')['base_price']
    assert (ClassConfigDataset(CONSTANT).fetch('Node.Results.price') == base_price + 5.0).all().all()
    dataset = ClassConfigDataset(CONSTANT, config=DummyConfig(price_offset=10.0))
    assert (dataset.fetch('Node.Results.price') == base_price + 10.0).all().all()
    assert (dataset.fetch('Node.Results.price', config={'price_offset': 1.0}) == base_price + 1.0).all().all()
    merged = DummyConfig(price_offset=3.0).merge(DummyConfig(use_cache=False))
    assert isinstance(merged, DummyConfig) and merged.price_offset == 3.0


class MarkupStudyDataset(DummyDataset):
    pass


@MarkupStudyDataset.register_interpreter
class PriceMarkupInterpreter(DummyInterpreter):
    """A study flag that fetches its input with the config of the call, as the docs and the skill teach."""

    MARKUP_SPEC = TimeSeriesFlagSpec(
        name='Node.Results.price_with_markup',
        description='Node price with a markup of 10 percent.',
        unit=Units.EUR_per_MWh,
        model_flag='Node.Model',
    )
    flag_specs = (MARKUP_SPEC,)
    INPUT_FLAGS = {MARKUP_SPEC.name: ('Node.Results.price',)}

    def _fetch(self, flag, effective_config, **params):
        return 1.1 * self.parent_dataset.fetch('Node.Results.price', effective_config)


def test_study_flag_derived_from_the_price_follows_a_price_offset_of_the_call():
    dataset = MarkupStudyDataset(CONSTANT)
    offset = {'price_offset': 100.0}
    with_offset = dataset.fetch('Node.Results.price_with_markup', config=offset)
    pd.testing.assert_frame_equal(with_offset, 1.1 * dataset.fetch('Node.Results.price', config=offset))
    pd.testing.assert_frame_equal(with_offset, dataset.fetch('Node.Results.price_with_markup') + 110.0)


def test_study_flag_that_passes_its_config_records_the_inputs_as_inherited_and_not_the_price_offset():
    dataset = MarkupStudyDataset(CONSTANT)
    dataset.fetch('Node.Results.price_with_markup', config={'price_offset': 100.0})
    record = dataset.get_fetch_record('Node.Results.price_with_markup', config={'price_offset': 100.0})
    (interpreter_record,) = record.dependencies
    (price_record,) = interpreter_record.dependencies
    assert price_record.call_config is INHERIT
    assert 'price_offset' not in record.config_reads
    assert 'price_offset' not in interpreter_record.config_reads
    assert [r.config_reads['price_offset'] for r in price_record.iter_dependencies()] == [100.0]


class PriceWeightedDemandDataset(DummyDataset):
    pass


@PriceWeightedDemandDataset.register_interpreter
class PriceWeightedDemandFlags(AreaFlagInterpreter):
    """A layer flag whose aggregation fetches an input that depends on the config: the price as weight."""

    flag_specs = (
        AreaFlagSpec(
            name='{area_layer}.Results.price_weighted_demand_{product}',
            base=DummyDerivedInterpreter.NODE_DEMAND_SPEC,
            aggregation=WeightedMean(weight='Node.Results.price'),
            description='Mean of the node demand of the area, weighted by the node price.',
        ),
    )


def test_weight_that_the_aggregation_fetches_follows_a_price_offset_of_the_call():
    dataset = PriceWeightedDemandDataset(CONSTANT)
    offset = {'price_offset': 100.0}
    result = dataset.fetch('Country.Results.price_weighted_demand_total', config=offset)
    demand = dataset.fetch('Node.Results.demand_total')
    weights = dataset.fetch('Node.Results.price', config=offset)
    country = dataset.fetch('Node.Model')['country']
    expected = (demand * weights).T.groupby(country).sum().T / weights.T.groupby(country).sum().T
    pd.testing.assert_frame_equal(result, expected, check_names=False)
    assert not result.equals(dataset.fetch('Country.Results.price_weighted_demand_total'))


def layer_flags_with_static_variants() -> list[str]:
    """The layer flags of the synthetic platform, without the fuel family.

    The fuel dimension reads its values from `Generator.Model` with the
    config of the dataset, not with the config of the call.
    """
    dataset = DummyDataset(CONSTANT)
    specs = [s for s in dataset.flag_index.specs if isinstance(s, LayerFlagSpec)]
    return sorted(f for s in specs if s.name != DummyAreaFlags.GENERATION_BY_FUEL_SPEC.name for f in s.expand(dataset))


@pytest.mark.parametrize('flag', layer_flags_with_static_variants())
def test_every_nested_fetch_of_a_layer_flag_inherits_the_config_of_the_call(flag):
    dataset = DummyDataset(CONSTANT)
    config = {'param_defaults': {'solution': 'first'}, 'price_offset': 100.0}
    dataset.fetch(flag, config=config)
    nested = list(dataset.get_fetch_record(flag, config=config).iter_dependencies())
    assert nested and all(record.call_config is INHERIT for record in nested)


@pytest.mark.parametrize(
    'table, unavailable, available',
    [
        (
            'line_capacity',
            ['Line.Constraints.capacity_up', 'Line.Results.loading', 'CountryBorder.Constraints.capacity_up'],
            ['Line.Results.flow_sent_up', 'CountryBorder.Results.flow_sent_up'],
        ),
        (
            'generator_generation',
            ['Generator.Results.generation', 'Node.Results.generation_gas', 'Country.Results.generation_gas'],
            ['Generator.Model'],
        ),
        (
            'demand_load',
            ['Demand.Results.demand_total', 'Node.Results.demand_total', 'Country.Results.demand_total'],
            ['Demand.Model'],
        ),
        (
            'line_flow',
            ['Line.Results.flow_sent_up', 'Line.Results.loading', 'CountryBorder.Results.flow_sent_up'],
            ['CountryBorder.Model'],
        ),
    ],
)
def test_missing_table_makes_its_flags_and_the_flags_derived_from_them_unavailable(table, unavailable, available):
    dataset = DummyDataset(BASE.replace(missing_tables=(table,)))
    for flag in unavailable:
        assert dataset.flag_is_accepted(flag)
        assert dataset.is_flag_available(flag) is False, flag
        with pytest.raises(ValueError, match='not available'):
            dataset.fetch(flag)
    for flag in available:
        assert dataset.is_flag_available(flag) is True, flag


def test_missing_generator_model_gives_a_generation_family_without_flags():
    dataset = DummyDataset(BASE.replace(missing_tables=('generators',)))
    rows = dataset.describe_flags(available_only=False)
    assert rows.loc['Node.Results.generation_{fuel}', 'flags'] == 0
    assert rows.loc['{area_layer}.Results.generation_{fuel}', 'flags'] == 0
    assert 'Node.Results.generation_{fuel}' not in dataset.describe_flags().index
    assert dataset.is_flag_available('Node.Results.generation_gas') is False
    assert dataset.flag_info('Country.Results.generation_gas').available is False
    assert 'Node.Model' in dataset.accepted_flags


BORDER_FLAGS = sorted(f for f in DummyDataset(BASE).accepted_flags if f.startswith('CountryBorder.'))


@pytest.mark.parametrize(
    'table, flag, missing_input',
    [
        ('demands', 'Country.Results.price', 'Node.Results.demand_total'),
        ('demands', 'CountryBorder.Results.price_spread', 'Country.Results.price'),
        ('nodes', 'Country.Results.price', 'Node.Model'),
        ('nodes', 'CountryBorder.Model', 'Node.Model'),
        ('nodes', 'CountryBorder.Results.flow_sent_up', 'Node.Model'),
        ('lines', 'CountryBorder.Model', 'Line.Model'),
        ('lines', 'CountryBorder.Results.flow_sent_up', 'Line.Model'),
    ],
)
def test_layer_flag_is_unavailable_when_an_input_of_its_fetch_is_unavailable(table, flag, missing_input):
    dataset = DummyDataset(BASE.replace(missing_tables=(table,)))
    info = dataset.flag_info(flag)
    assert info.available is False
    assert f'unavailable input {missing_input} (' in info.availability_reason
    with pytest.raises(FlagUnavailableError):
        dataset.fetch(flag)


def test_reason_names_the_missing_table():
    reason = (
        DummyDataset(BASE.replace(missing_tables=('demands',))).flag_info('Country.Results.price').availability_reason
    )
    assert "table 'demands' not in the source" in reason


def test_area_flags_need_no_layer_model():
    dataset = DummyDataset(BASE.replace(missing_tables=('countries',)))
    assert dataset.is_flag_available('Country.Results.price') is True
    assert not dataset.fetch('Country.Results.price').empty


def test_border_flags_without_the_layer_model_are_available_and_the_border_geometry_is_nan(caplog):
    dataset = DummyDataset(BASE.replace(days=1, missing_tables=('countries',)))
    assert all(dataset.is_flag_available(flag) is True for flag in BORDER_FLAGS)
    with caplog.at_level('INFO', logger='mesqual.layer_flags.aggregations'):
        model = dataset.fetch('CountryBorder.Model')
        assert not dataset.fetch('CountryBorder.Results.flow_sent_up').empty
    assert set(model.index) == {'C1 - C2', 'C2 - C1'}
    assert model[['projection_point', 'azimuth_angle', 'geo_line_string', 'areas_touch']].isna().all().all()
    messages = [record.getMessage() for record in caplog.records if record.levelname == 'INFO']
    assert messages == ["Border model of area layer 'Country' without geometry: Country.Model is unavailable."]


def test_fetch_of_a_family_flag_without_values_raises_flag_unavailable_error_with_the_cause():
    dataset = DummyDataset(BASE.replace(missing_tables=('generators',)))
    reason = dataset.availability('Node.Results.generation_gas').reason
    assert reason.startswith("values of 'fuel' need the unavailable flag Generator.Model: ")
    with pytest.raises(FlagUnavailableError) as error:
        dataset.fetch('Node.Results.generation_gas')
    assert (error.value.flag, error.value.dataset) == ('Node.Results.generation_gas', 'base')
    assert "table 'generators' not in the source" in error.value.reason


@pytest.mark.parametrize(
    'flag, max_generator_reads',
    [
        ('Node.Results.generation_gas', 3),
        ('Country.Results.generation_gas', 7),
    ],
)
def test_fetch_of_a_family_flag_reads_the_generator_table_at_most_as_often_as_before(
    monkeypatch,
    flag,
    max_generator_reads,
):
    reads = Counter()
    read = DummySource.read

    def counting_read(source, table):
        reads[table] += 1
        return read(source, table)

    monkeypatch.setattr(DummySource, 'read', counting_read)
    DummyDataset(BASE).fetch(flag)
    assert reads['generators'] <= max_generator_reads


@pytest.mark.parametrize('table', DummySource.TABLES)
def test_describe_flags_and_flag_info_match_the_fetch_for_every_layer_flag(table):
    dataset = DummyDataset(BASE.replace(days=1, missing_tables=(table,)))
    rows = dataset.describe_flags(available_only=False)
    for spec in dataset.flag_index.specs:
        if not isinstance(spec, LayerFlagSpec):
            continue
        fetched = 0
        for flag in spec.expand(dataset):
            available = dataset.flag_info(flag).available
            assert available is dataset.is_flag_available(flag), flag
            if available:
                dataset.fetch(flag)
                fetched += 1
            else:
                with pytest.raises(ValueError):
                    dataset.fetch(flag)
        assert rows.loc[spec.name, 'flags'] == fetched, spec.name


class UnknownWeightDataset(DummyDataset):
    pass


@UnknownWeightDataset.register_interpreter
class UncheckedWeight(DummyInterpreter):
    """A weight flag without an availability check."""

    WEIGHT_SPEC = TimeSeriesFlagSpec(
        name='Node.Results.weight',
        description='Weight.',
        model_flag='Node.Model',
        unit=Units.MW,
    )
    flag_specs = (WEIGHT_SPEC,)
    INPUT_FLAGS = {WEIGHT_SPEC.name: ('Node.Results.demand_total',)}

    def _is_available(self, flag):
        return None

    def _fetch(self, flag, effective_config, **params):
        return self._fetch_inputs(flag, effective_config, params)[0]


@UnknownWeightDataset.register_interpreter
class UnknownWeightAreaFlags(AreaFlagInterpreter):
    flag_specs = (
        AreaFlagSpec(
            name='{area_layer}.Results.weighted_price',
            base=DummyTimeSeriesInterpreter.PRICE_SPEC,
            aggregation=WeightedMean(weight='Node.Results.weight'),
            description='Price weighted by the unchecked weight.',
        ),
    )


def test_layer_flag_with_an_unknown_input_is_unknown_until_the_input_is_fetched():
    dataset = UnknownWeightDataset(BASE)
    info = dataset.flag_info('Country.Results.weighted_price')
    assert info.available is None
    assert 'unknown input Node.Results.weight' in info.availability_reason
    dataset.fetch('Node.Results.weight')
    assert dataset.is_flag_available('Country.Results.weighted_price') is True


def test_missing_table_hides_its_specs_in_describe_flags():
    dataset = DummyDataset(BASE.replace(missing_tables=('line_capacity',)))
    assert 'Line.Constraints.capacity_{direction}' not in dataset.describe_flags().index
    all_rows = dataset.describe_flags(available_only=False)
    assert all_rows.loc['Line.Constraints.capacity_{direction}', 'availability'] == 'unavailable'
    assert all_rows.loc['Line.Results.loading', 'availability'] == 'unavailable'
    assert all_rows.loc['Line.Results.flow_{side}_{direction}', 'availability'] == 'available'


def test_default_attributes_come_from_the_scenario_and_explicit_attributes_win():
    assert DummyDataset(BASE).attributes == {'first_date': '2024-03-30', 'last_date': '2024-04-01', 'days': 3}
    assert DummyDataset(BASE, attributes={'days': 5}).attributes['days'] == 5


def test_duplicate_rows_are_removed_by_the_fetch_post_processing():
    dataset = DummyDataset(BASE.replace(duplicate_rows=3))
    prices = dataset.fetch('Node.Results.price')
    assert not prices.index.duplicated().any()
    assert prices.index.is_monotonic_increasing
    all_rows = dataset.fetch('Node.Results.price', config={'keep_duplicate_indices': 'all'})
    assert all_rows.index.duplicated().sum() == 3


def test_split_scenario_merges_into_the_unsplit_series():
    merged = make_dummy_dataset(BASE.split('2024-03-31'))
    assert isinstance(merged, DatasetMergeCollection)
    assert merged.name == 'base'
    for flag in ('Node.Results.price', 'Generator.Results.generation', 'Line.Results.flow_received_down', 'Line.Model'):
        pd.testing.assert_frame_equal(merged.fetch(flag), DummyDataset(BASE).fetch(flag))


def test_period_dataset_of_a_split_scenario_merges_into_the_unsplit_series():
    parts = [DummyPeriodDataset(part) for part in BASE.split('2024-03-31')]
    merged = DatasetMergeCollection(parts, name='base')
    for flag in ('Node.Results.price', 'Line.Results.loading'):
        pd.testing.assert_frame_equal(merged.fetch(flag), DummyPeriodDataset(BASE).fetch(flag))


def test_make_dummy_dataset_names_parts_without_suffix_and_raises_without_common_name():
    first, second = DummyScenario(name='alpha.beta', days=2).split('2024-01-02')
    assert make_dummy_dataset((first, second)).name == 'alpha.beta'
    with pytest.raises(ValueError):
        make_dummy_dataset((first, second.replace(name='other.2')))
    assert make_dummy_dataset((first, second.replace(name='other.2')), name='mixed').name == 'mixed'


def test_merged_split_dataset_has_the_attributes_of_the_whole():
    merged = make_dummy_dataset(BASE.split('2024-03-31'))
    assert merged.attributes == DummyDataset(BASE).attributes


def test_make_dummy_dataset_without_parts_raises():
    with pytest.raises(ValueError, match='at least one part'):
        make_dummy_dataset(())


def test_make_dummy_dataset_of_one_scenario_is_a_dummy_dataset():
    dataset = make_dummy_dataset(BASE, name='other')
    assert isinstance(dataset, DummyDataset) and dataset.name == 'other'


def test_period_dataset_has_a_tuple_time_level():
    prices = DummyPeriodDataset(BASE).fetch('Node.Results.price')
    assert prices.index[0] == (20240330, 1)
    assert len(prices) == len(DummyDataset(BASE).fetch('Node.Results.price'))
    assert not DummyPeriodDataset(BASE).fetch('Line.Results.loading').empty


def test_flag_index_reads_metadata_from_the_specs():
    index = DummyDataset(BASE).flag_index
    assert index.get_item_type('Node.Model') == ItemTypeEnum.Model
    assert index.get_item_type('Line.Results.flow_sent_up') == ItemTypeEnum.TimeSeries
    assert index.get_linked_model_flag('Line.Results.flow_sent_up') == 'Line.Model'
    assert index.get_topology_type('Line.Model') == TopologyTypeEnum.Edge
    assert index.get_unit('Node.Results.price') == Units.EUR_per_MWh
    assert index.get_unit('Generator.Results.generation') == Units.MWh
    assert index.get_memberships('Line.Model') == {'node_from': 'Node.Model', 'node_to': 'Node.Model'}
    assert index.get_spec('Line.Model').edge_ends == ('node_from', 'node_to')
    assert index.get_memberships('Node.Model') == {'country': 'Country.Model'}
    assert index.get_referencing_columns('Country.Model') == [('Node.Model', 'country')]
    assert index.get_spec('Node.Results.price').fill_value is None
    assert index.get_spec('Generator.Results.generation').fill_value == 0
    assert index.get_topology_type('Demand.Model') == TopologyTypeEnum.NodeConnectedElement
    assert index.get_unit('Demand.Results.demand_total') == Units.MW


def _compare(variation: DummyScenario, reference: DummyScenario, flag: str) -> pd.DataFrame:
    return DatasetComparison(DummyDataset(variation), DummyDataset(reference)).fetch(flag)


def test_comparison_with_a_removed_object_has_zero_deltas_on_the_others():
    less = BASE.replace(name='less', removed_objects=('G2', 'N2'))
    deltas = _compare(less, BASE, 'Generator.Results.generation')
    assert (deltas['G1'] == 0).all()
    removed = ['G2', 'G3', 'G4']
    pd.testing.assert_frame_equal(deltas[removed], -DummyDataset(BASE).fetch('Generator.Results.generation')[removed])
    line_deltas = _compare(BASE.replace(name='less', removed_objects=('N4',)), BASE, 'Line.Results.flow_sent_net')
    assert (line_deltas[['L1', 'L2', 'L3']] == 0).all().all()


def test_comparison_with_a_removed_node_drops_its_demands():
    less = BASE.replace(name='less', removed_objects=('N2',))
    assert DummyDataset(less).fetch('Demand.Model').index.tolist() == ['D1', 'D2', 'D5', 'D6', 'D7', 'D8']
    deltas = _compare(less, BASE, 'Demand.Results.demand_total')
    removed = DummyDataset(BASE).fetch('Demand.Results.demand_total')[['D3', 'D4']]
    pd.testing.assert_frame_equal(deltas[['D3', 'D4']], -removed)
    assert (deltas[['D1', 'D2', 'D5', 'D6', 'D7', 'D8']] == 0).all().all()
    node_deltas = _compare(less, BASE, 'Node.Results.demand_total')
    pd.testing.assert_series_equal(node_deltas['N2'], -DummyDataset(BASE).fetch('Node.Results.demand_total')['N2'])
    assert (node_deltas[['N1', 'N3', 'N4']] == 0).all().all()


def test_comparison_with_added_objects_has_zero_deltas_on_the_existing_ones():
    more = BASE.replace(name='more', n_nodes=5, n_lines=7, n_generators=6, n_demands=10)
    for flag, existing in [
        ('Node.Results.price', ['N1', 'N2', 'N3', 'N4']),
        ('Node.Results.demand_total', ['N1', 'N2', 'N3', 'N4']),
        ('Line.Results.loading', ['L1', 'L2', 'L3', 'L4', 'L5']),
        ('Generator.Results.generation', ['G1', 'G2', 'G3', 'G4', 'G5']),
        ('Demand.Results.demand_total', [f'D{i}' for i in range(1, 9)]),
    ]:
        deltas = _compare(more, BASE, flag)
        assert (deltas[existing] == 0).all().all(), flag
    model = _compare(more, BASE, 'Generator.Model')
    assert model.loc['G6', 'fuel'] == 'nuclear (new)'
    assert model.loc['G1', 'fuel'] == 'nuclear'
    demand_model = _compare(more, BASE, 'Demand.Model')
    assert demand_model.loc['D10', 'sector'] == 'industry (new)'
    assert demand_model.loc['D1', 'sector'] == 'industry'


def test_comparison_with_a_changed_text_property_shows_the_change():
    changed = BASE.replace(name='changed', text_changes=(('generators', 'G2', 'fuel', 'gas'),))
    model = _compare(changed, BASE, 'Generator.Model')
    assert model.loc['G2', 'fuel'] == 'gas (was coal)'
    generation_delta = _compare(changed, BASE, 'Generator.Results.generation')
    assert (generation_delta.drop(columns='G2') == 0).all().all()
    assert (generation_delta['G2'] != 0).any()
    assert 'Node.Results.generation_coal' not in DummyDataset(changed).accepted_flags


def test_node_without_country_appears_in_model_and_series():
    dataset = DummyDataset(BASE.replace(node_without_country=True))
    assert pd.isna(dataset.fetch('Node.Model').loc['N0', 'country'])
    assert dataset.fetch('Node.Results.price')['N0'].notna().all()
    lines = dataset.fetch('Line.Model')
    assert 'N0' not in set(lines.node_from) | set(lines.node_to)


def test_differing_horizons_concat_with_nan_and_compare_with_nan_rows():
    short = BASE.replace(name='short', days=2)
    concat = DatasetConcatCollection([DummyDataset(BASE), DummyDataset(short)]).fetch('Node.Results.price')
    assert concat['short'].loc['2024-04-01'].isna().all().all()
    assert concat['short'].loc[:'2024-03-31'].notna().all().all()
    deltas = _compare(short, BASE, 'Node.Results.price')
    assert deltas.loc['2024-04-01'].isna().all().all()
    assert (deltas.loc[:'2024-03-31'] == 0).all().all()


def test_fetched_series_switch_the_step_size_at_the_resolution_change():
    index = DummyDataset(BASE).fetch('Node.Results.demand_total').index
    steps = pd.Series(index[1:] - index[:-1], index=index[:-1])
    assert set(steps.loc[:'2024-03-30 22:00']) == {pd.Timedelta('60min')}
    assert set(steps.loc['2024-03-31':]) == {pd.Timedelta('15min')}
    assert len(index) == 24 + 92 + 96


def test_nan_gaps_and_missing_rows_on_a_fall_back_scenario():
    clean = DummyScenario(name='clean', start='2024-10-26', days=2, tz='Europe/Berlin')
    gappy = clean.replace(
        name='gappy',
        nan_gaps=(('2024-10-27 02:00+01:00', '2024-10-27 03:00+01:00'), ('2024-10-27 10:00', '2024-10-27 12:00')),
        missing_rows=(('2024-10-26 05:00', '2024-10-26 07:00'),),
    )
    prices = DummyDataset(gappy).fetch('Node.Results.price')
    nan_rows = prices.index[prices.isna().all(axis=1)]
    assert [t.strftime('%H:%M%z') for t in nan_rows] == ['02:00+0100', '10:00+0100', '11:00+0100']
    assert prices.isna().any(axis=1).sum() == 3
    assert len(prices.loc['2024-10-27']) == 25
    assert len(prices.loc['2024-10-26']) == 22
    deltas = _compare(gappy, clean, 'Node.Results.price')
    assert len(deltas) == 24 + 25
    nan_delta_rows = deltas.index[deltas.isna().all(axis=1)]
    assert len(nan_delta_rows) == 3 + 2
    assert (deltas.dropna() == 0).all().all()


def test_platform_dataset_with_a_parent_keeps_its_own_flag_index():
    dataset = DummyDataset(BASE)
    index = dataset.flag_index
    dataset.parent_dataset = DatasetLinkCollection([DummyDataset(BASE, name='other')])
    assert dataset.flag_index is index
