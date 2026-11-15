"""Layers, layer specs and their flags in the flag index."""

import dataclasses

import pytest

from mesqual.layer_flags import AreaModel, Sum
from mesqual.enums import ItemTypeEnum, TopologyTypeEnum
from mesqual.flag import (
    AreaFlagSpec,
    AreaLayer,
    BorderFlagSpec,
    BorderLayer,
    Dimension,
    FlagIndex,
    LayerSet,
    ModelFlagSpec,
    TimeSeriesFlagSpec,
    area_flag_specs,
)
from mesqual.flag.exceptions import AmbiguousOverrideError
from mesqual.units import Units
from tests.layer_flags.tables import (
    AREA_DEMAND,
    AREA_EXCHANGE,
    AREA_MODEL,
    AREA_PRICE,
    BORDER_FLOW,
    BORDER_MODEL_SPEC,
    DEMAND_SPEC,
    LINE_MODEL_SPEC,
    NODE_MODEL_SPEC,
    PRICE_SPEC,
    REGION,
    SPREAD,
    ZONE,
    ZONE_BORDER,
    TableInterpreter,
    TablePlatform,
    platform,
)


def node_model(memberships: dict) -> ModelFlagSpec:
    return dataclasses.replace(NODE_MODEL_SPEC, memberships=memberships)


def with_models(*specs) -> type:
    """A platform whose model interpreter declares `specs` instead of the node and line model."""
    models = type('Models', (TableInterpreter,), {'flag_specs': specs, 'tables': {}})
    cls = type('Custom', (TablePlatform,), {})
    cls.register_interpreter(models, override_interpreter=TablePlatform.get_registered_interpreters()[0], version=1)
    return cls


class TestLayerObjects:
    def test_layer_without_index_raises_type_error(self):
        with pytest.raises(TypeError):
            AreaLayer(name='Zone', base_model='Node.Model', layer_model='Zone.Model', description='Zones.')
        with pytest.raises(TypeError):
            BorderLayer(name='ZoneBorder', area_layer='Zone', edge_model='Line.Model', description='Borders.')

    def test_layer_set_rejects_duplicate_names_and_unknown_area_layers(self):
        with pytest.raises(AmbiguousOverrideError):
            LayerSet([ZONE, dataclasses.replace(ZONE, index='other')])
        with pytest.raises(ValueError, match='Zone'):
            LayerSet([], [ZONE_BORDER])


class TestRegistry:
    def test_layers_resolve_along_the_mro_and_never_reach_the_parent(self):
        parent = platform()
        study = type('Study', (parent,), {})
        study.register_area_layer(REGION)
        study.register_area_layer(REGION)
        assert parent.get_area_layers() == [ZONE]
        assert study.get_area_layers() == [ZONE, REGION]
        assert study.get_border_layers() == [ZONE_BORDER]
        assert TablePlatform.get_area_layers() == []

    def test_register_rejects_other_types(self):
        with pytest.raises(TypeError):
            platform().register_area_layer(ZONE_BORDER)

    def test_any_dataset_subclass_is_an_interpreter(self):
        cls = platform(area_specs=(AREA_DEMAND,))
        assert not issubclass(cls.get_registered_interpreters()[-1], cls.get_child_dataset_type())
        with pytest.raises(TypeError, match='Dataset subclass'):
            cls.register_interpreter(object)

    def test_study_subclass_adds_a_layer_and_gets_its_flags(self):
        parent = platform(
            area_specs=(AREA_DEMAND,),
            base=with_models(
                node_model({'zone': 'Zone.Model', 'region': 'Region.Model'}),
                LINE_MODEL_SPEC,
            ),
        )
        study = type('Study', (parent,), {})
        study.register_area_layer(REGION)
        assert 'Region.Results.demand_peak' not in parent().accepted_flags
        assert {'Region.Results.demand_peak', 'Zone.Results.demand_peak'} <= study().accepted_flags

    def test_layer_values_from_the_class_and_from_an_instance(self):
        cls = platform(area_specs=(AREA_PRICE,), border_specs=(BORDER_MODEL_SPEC, SPREAD))
        assert AREA_PRICE.layer_dimension.values_for(cls) == {'Zone': 'Zones.'}
        assert SPREAD.layer_dimension.values_for(cls) == {'ZoneBorder': 'Zone borders.'}
        assert AREA_PRICE.layer_dimension.values_for(cls()) == {'Zone': 'Zones.'}


class TestLayerSpecs:
    def test_one_flag_per_layer_and_variant_value(self):
        ds = platform(area_specs=(AREA_PRICE,))()
        assert {f for f in ds.accepted_flags if f.startswith('Zone.')} == {
            'Zone.Results.price_peak',
            'Zone.Results.price_offpeak',
        }

    def test_fields_come_from_the_base(self):
        assert AREA_PRICE.item_type is ItemTypeEnum.TimeSeries
        assert AREA_PRICE.unit is Units.EUR_per_MWh
        assert AREA_DEMAND.fill_value == 0
        assert AREA_MODEL.item_type is ItemTypeEnum.Model and AREA_MODEL.unit is None
        assert AREA_PRICE.quantity_type == PRICE_SPEC.quantity_type

    def test_object_level_is_the_layer_index(self):
        ds = platform(area_specs=(AREA_PRICE,), border_specs=(BORDER_MODEL_SPEC, SPREAD))()
        assert ds.flag_index.get_schema('Zone.Results.price_peak').columns[0].name == 'zone'
        assert ds.flag_index.get_schema('ZoneBorder.Results.spread_offpeak').columns[0].name == 'zone_border'
        assert ds.flag_index.get_topology_type('ZoneBorder.Model') is TopologyTypeEnum.Border

    def test_layer_placeholder_equal_to_a_base_dimension_raises(self):
        with pytest.raises(ValueError, match='period'):
            AreaFlagSpec(
                name='{period}.Results.price',
                base=PRICE_SPEC,
                aggregation=Sum(),
                description='x',
                layer_placeholder='period',
            )

    def test_base_topology_must_fit_the_class(self):
        on_lines = AreaFlagSpec(
            name='{area_layer}.Model', base=LINE_MODEL_SPEC, aggregation=AreaModel(), description='x'
        )
        with pytest.raises(ValueError, match='Edge'):
            platform(area_specs=(on_lines,))()
        on_nodes = BorderFlagSpec(
            name='{border_layer}.Model',
            base=NODE_MODEL_SPEC,
            aggregation=AreaModel(),
            description='x',
        )
        with pytest.raises(ValueError, match='Node'):
            platform(border_specs=(on_nodes,))()

    def test_base_spec_without_layer_spec_has_no_layer_flags(self):
        ds = platform(area_specs=(AREA_PRICE,))()
        assert not any('capacity' in f and not f.startswith('Line.') for f in ds.accepted_flags)

    def test_layer_model_flag_exists_only_when_declared(self):
        assert 'Zone.Model' not in platform(area_specs=(AREA_PRICE,))().accepted_flags
        ds = platform(area_specs=(AREA_PRICE, AREA_MODEL))()
        assert 'Zone.Model' in ds.accepted_flags
        assert ds.flag_index.get_linked_model_flag('Zone.Results.price_peak') == 'Zone.Model'

    def test_flag_grammar_without_dots(self):
        spec = AreaFlagSpec(name='{area_layer}_demand_{period}', base=DEMAND_SPEC, aggregation=Sum(), description='x')
        ds = platform(area_specs=(spec,))()
        assert ds.fetch('Zone_demand_peak').columns.tolist() == ['A', 'B']

    def test_two_specs_with_the_same_flag_raise(self):
        twin = dataclasses.replace(AREA_DEMAND, description='Other.')
        with pytest.raises(AmbiguousOverrideError):
            platform(area_specs=(AREA_DEMAND, twin))()

    def test_layer_flag_colliding_with_a_base_flag_raises(self):
        spec = AreaFlagSpec(name='{area_layer}.Model', base=NODE_MODEL_SPEC, aggregation=AreaModel(), description='x')
        layer = dataclasses.replace(ZONE, name='Node')
        with pytest.raises(AmbiguousOverrideError):
            platform(area_specs=(spec,), area_layers=(layer,), border_layers=())()

    def test_area_flag_specs_equal_hand_written_specs(self):
        built = area_flag_specs(
            bases=(DEMAND_SPEC,),
            aggregation=Sum(),
            name=lambda base: base.name.replace('Node.', '{area_layer}.'),
            description='Area demand.',
        )
        assert len(built) == 1 and built[0].same_as(AREA_DEMAND)
        assert not built[0].same_as(dataclasses.replace(AREA_DEMAND, aggregation=Sum(version=Sum().version + 1)))

    def test_consumed_dimensions_leave_the_name(self):
        spec = AreaFlagSpec(
            name='{area_layer}.Results.price',
            base=PRICE_SPEC,
            aggregation=Sum(),
            description='x',
            consumes=('period',),
        )
        assert [d.name for d in spec.variants] == ['area_layer']
        with pytest.raises(ValueError, match='sector'):
            dataclasses.replace(spec, consumes=('sector',))


class TestMemberships:
    def test_no_membership_to_the_layer_model_raises(self):
        with pytest.raises(ValueError, match='exactly one membership'):
            platform(area_layers=(REGION,), border_layers=())()

    def test_two_memberships_to_the_layer_model_raise(self):
        base = with_models(node_model({'zone': 'Zone.Model', 'zone_2': 'Zone.Model'}), LINE_MODEL_SPEC)
        with pytest.raises(ValueError, match='exactly one membership'):
            platform(base=base, border_layers=())()

    def test_renamed_membership_column_needs_no_change_of_the_layer(self):
        base = with_models(node_model({'zone_symbol': 'Zone.Model'}), LINE_MODEL_SPEC)
        ds = platform(base=base)()
        assert ds.flag_index.get_mapping_column(ZONE) == 'zone_symbol'

    def test_edge_model_without_edge_ends_raises(self):
        line = dataclasses.replace(LINE_MODEL_SPEC, edge_ends=None)
        with pytest.raises(ValueError, match='edge_ends'):
            platform(base=with_models(NODE_MODEL_SPEC, line))()

    def test_edge_ends_to_another_model_raise(self):
        line = dataclasses.replace(LINE_MODEL_SPEC, memberships={'node_from': 'Bus.Model', 'node_to': 'Bus.Model'})
        bus = dataclasses.replace(NODE_MODEL_SPEC, name='Bus.Model', index='Bus', memberships={})
        with pytest.raises(ValueError, match='Node.Model'):
            platform(base=with_models(NODE_MODEL_SPEC, line, bus))()


class TestBorderBase:
    def test_layer_flag_name_drops_consumed_and_base_layer_dimensions(self):
        assert [d.name for d in AREA_EXCHANGE.variants] == ['area_layer']

    def test_area_layer_with_two_border_layers_raises(self):
        second = dataclasses.replace(ZONE_BORDER, name='ZoneBorder2', index='zone_border_2')
        cls = platform(
            area_specs=(AREA_EXCHANGE,),
            border_specs=(BORDER_MODEL_SPEC, BORDER_FLOW),
            border_layers=(ZONE_BORDER, second),
        )
        with pytest.raises(ValueError, match='ZoneBorder'):
            cls()


class TestTemplateModelFlag:
    def test_time_series_spec_on_a_layer_model_keeps_topology_and_level(self):
        spec = TimeSeriesFlagSpec(
            name='{area_layer}.Results.balance',
            description='Balance per area.',
            model_flag='{area_layer}.Model',
            unit=Units.MW,
            variants=(Dimension('area_layer', 'Area layer.', lambda ds: {'Zone': 'Zones.'}),),
        )
        cls = platform(area_specs=(AREA_MODEL,))
        cls.register_interpreter(type('Balance', (TableInterpreter,), {'flag_specs': (spec,), 'tables': {}}))
        index = cls().flag_index
        assert index.get_linked_model_flag('Zone.Results.balance') == 'Zone.Model'
        assert index.get_topology_type('Zone.Results.balance') is TopologyTypeEnum.Area
        assert index.get_schema('Zone.Results.balance').columns[0].name == 'zone'

    def test_template_placeholder_must_be_a_variant(self):
        with pytest.raises(ValueError, match='area_layer'):
            TimeSeriesFlagSpec(
                name='Zone.Results.balance',
                description='x',
                model_flag='{area_layer}.Model',
                unit=Units.MW,
            )

    def test_template_typo_on_a_family_raises_at_instance_creation(self):
        spec = TimeSeriesFlagSpec(
            name='{area_layer}.Results.balance',
            description='Balance per area.',
            model_flag='{area_layer}.Modell',
            unit=Units.MW,
            variants=(Dimension('area_layer', 'Area layer.', lambda ds: {'Zone': 'Zones.'}),),
        )
        cls = platform(area_specs=(AREA_MODEL,))
        cls.register_interpreter(type('Balance', (TableInterpreter,), {'flag_specs': (spec,), 'tables': {}}))
        with pytest.raises(ValueError, match='Zone.Modell'):
            cls()

    def test_layer_model_flag_placeholders_must_be_variants(self):
        with pytest.raises(ValueError, match='period_kind'):
            dataclasses.replace(AREA_PRICE, model_flag='{area_layer}.Model_{period_kind}')

    def test_layer_model_flag_is_checked_at_instance_creation(self):
        good = dataclasses.replace(AREA_PRICE, model_flag='{area_layer}.Model')
        ds = platform(area_specs=(AREA_MODEL, good))()
        assert ds.flag_index.get_linked_model_flag('Zone.Results.price_peak') == 'Zone.Model'
        with pytest.raises(ValueError, match='Zone.Modell'):
            platform(area_specs=(AREA_MODEL, dataclasses.replace(AREA_PRICE, model_flag='{area_layer}.Modell')))()


class TestBaseFlag:
    def test_consumed_family_dimension_raises(self):
        base = TimeSeriesFlagSpec(
            name='Node.Results.generation_{fuel}',
            description='Generation.',
            model_flag='Node.Model',
            unit=Units.MW,
            variants=(Dimension('fuel', 'Fuel.', lambda ds: {'gas': 'gas'}),),
        )
        spec = AreaFlagSpec(
            name='{area_layer}.Results.generation',
            base=base,
            aggregation=Sum(),
            description='x',
            consumes=('fuel',),
        )
        ds = platform(area_specs=(spec,))()
        with pytest.raises(ValueError, match='fuel'):
            ds.flag_index.get_base_flag('Zone.Results.generation')


class TestIndexLayers:
    def test_check_references_checks_the_layers(self):
        index = FlagIndex([NODE_MODEL_SPEC, LINE_MODEL_SPEC, AREA_DEMAND, DEMAND_SPEC], LayerSet([ZONE, REGION]))
        assert 'Region.Results.demand_peak' in index.static_flags
        with pytest.raises(ValueError, match='Region'):
            index.check_references()

    def test_layer_variant_holds_the_layer_of_the_base(self):
        index = platform(area_specs=(AREA_PRICE,), border_specs=(BORDER_MODEL_SPEC, SPREAD))().flag_index
        variant = index.get_layer_variant('ZoneBorder.Results.spread_peak')
        assert variant.as_dict() == {'border_layer': 'ZoneBorder', 'period': 'peak', 'area_layer': 'Zone'}
        assert variant.base_flag() == 'Zone.Results.price_peak'
        assert variant.base_flag(period='offpeak') == 'Zone.Results.price_offpeak'

    def test_get_base_flag_of_a_base_flag_raises(self):
        index = platform(area_specs=(AREA_DEMAND,))().flag_index
        assert index.get_base_flag('Zone.Results.demand_offpeak') == 'Node.Results.demand_offpeak'
        with pytest.raises(ValueError, match='layer spec'):
            index.get_base_flag('Node.Results.demand_offpeak')

    def test_union_of_layer_sets_with_different_layers_of_one_name_raises(self):
        with pytest.raises(AmbiguousOverrideError):
            LayerSet.union([LayerSet([ZONE]), LayerSet([dataclasses.replace(ZONE, index='other')])])

    def test_layer_model_flag_of_a_border_layer_follows_added_specs(self):
        index = platform(area_specs=(AREA_DEMAND,))().flag_index
        assert index.get_layer_model_flag(ZONE_BORDER) is None
        index.add_spec(BORDER_MODEL_SPEC)
        assert index.get_layer_model_flag(ZONE_BORDER) == 'ZoneBorder.Model'
