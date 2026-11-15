import pytest

from mesqual.enums import ItemTypeEnum, QuantityTypeEnum, TopologyTypeEnum, VisualizationTypeEnum
from mesqual.flag import (
    AmbiguousOverrideError,
    Dimension,
    FlagIndex,
    FrameSchema,
    Level,
    TimeSeriesFlagSpec,
    UnknownFlagError,
)
from mesqual.units import Units
from tests.flag.specs import LINE_FLOW_SPEC, LINE_MODEL_SPEC, NODE_MODEL_SPEC, NODE_PRICE_SPEC, SYSTEM_LOAD_SPEC


@pytest.fixture
def index() -> FlagIndex:
    return FlagIndex([NODE_MODEL_SPEC, LINE_MODEL_SPEC, NODE_PRICE_SPEC, LINE_FLOW_SPEC, SYSTEM_LOAD_SPEC])


class TestGetters:
    def test_item_type(self, index):
        assert index.get_item_type('Node.Model') is ItemTypeEnum.Model
        assert index.get_item_type('Line.Results.flow_sent_up') is ItemTypeEnum.TimeSeries

    def test_unit_and_quantity_type(self, index):
        assert index.get_unit('Node.Results.price') == Units.EUR_per_MWh
        assert index.get_unit('Node.Model') is None
        assert index.get_quantity_type_enum('Node.Results.price') is QuantityTypeEnum.INTENSIVE
        assert index.get_quantity_type_enum('Node.Model') is QuantityTypeEnum.UNKNOWN

    def test_fill_value(self):
        volume = TimeSeriesFlagSpec(
            name='Node.Results.volume',
            description='x',
            unit=Units.MWh,
            model_flag='Node.Model',
            fill_value=0,
        )
        index = FlagIndex([NODE_MODEL_SPEC, NODE_PRICE_SPEC, volume])
        assert index.get_fill_value('Node.Results.volume') == 0
        assert index.get_fill_value('Node.Results.price') is None
        assert index.get_fill_value('Node.Model') is None

    def test_quantity_type_of_ambiguous_unit_comes_from_spec(self, ambiguous_unit):
        spec = TimeSeriesFlagSpec(
            name='Node.Results.cost',
            description='x',
            unit=ambiguous_unit,
            model_flag='Node.Model',
            quantity_type=QuantityTypeEnum.EXTENSIVE,
        )
        assert (
            FlagIndex([NODE_MODEL_SPEC, spec]).get_quantity_type_enum('Node.Results.cost') is QuantityTypeEnum.EXTENSIVE
        )

    def test_quantity_type_of_capacity_price_units_comes_from_the_unit(self):
        specs = [
            TimeSeriesFlagSpec(name=f'Node.Results.{name}', description='x', unit=unit, model_flag='Node.Model')
            for name, unit in (('price', Units.EUR_per_MW), ('cost', Units.EUR_per_MW_per_period))
        ]
        index = FlagIndex([NODE_MODEL_SPEC, *specs])
        assert index.get_quantity_type_enum('Node.Results.price') is QuantityTypeEnum.INTENSIVE
        assert index.get_quantity_type_enum('Node.Results.cost') is QuantityTypeEnum.EXTENSIVE

    def test_topology_and_visualization_type_follow_the_model_spec(self, index):
        assert index.get_topology_type('Line.Results.flow_sent_up') is TopologyTypeEnum.Edge
        assert index.get_visualization_type('Line.Results.flow_sent_up') is VisualizationTypeEnum.Line
        assert index.get_topology_type('System.Results.load') is TopologyTypeEnum.Other

    def test_linked_model_flag(self, index):
        assert index.get_linked_model_flag('Node.Results.price') == 'Node.Model'
        assert index.get_linked_model_flag('Node.Model') == 'Node.Model'
        assert index.get_linked_model_flag('System.Results.load') is None

    def test_default_schemas(self, index):
        assert index.get_schema('Node.Model') == FrameSchema(
            index=(Level('Node', 'object'),),
            columns=(Level(None, 'dimension'),),
        )
        assert index.get_schema('Node.Results.price') == FrameSchema(
            index=(Level('time', 'time'),), columns=(Level('Node', 'object'),)
        )
        assert index.get_schema('System.Results.load') is SYSTEM_LOAD_SPEC.schema

    def test_memberships_and_referencing_columns(self, index):
        assert index.get_memberships('Line.Model') == {'node_from': 'Node.Model', 'node_to': 'Node.Model'}
        assert index.get_referencing_columns('Node.Model') == [('Line.Model', 'node_from'), ('Line.Model', 'node_to')]
        with pytest.raises(ValueError, match='not a model flag'):
            index.get_memberships('Node.Results.price')

    def test_static_flags_and_parameter_names(self, index):
        assert 'Line.Results.flow_received_net' in index.static_flags
        assert index.parameter_names == {'solution'}


class TestUnknownFlags:
    def test_unknown_flag_is_key_error_and_value_error(self, index):
        with pytest.raises(KeyError):
            index.get_unit('Node.Results.unknown')
        with pytest.raises(ValueError):
            index.get_item_type('Node.Results.unknown')
        with pytest.raises(UnknownFlagError, match='Node.Results.unknown'):
            index.get_spec('Node.Results.unknown')
        assert not index.has_spec('Node.Results.unknown')

    def test_no_fallback_for_unit_or_quantity_type(self):
        with pytest.raises(UnknownFlagError):
            FlagIndex().get_unit('Node.Results.price')
        with pytest.raises(UnknownFlagError):
            FlagIndex().get_quantity_type_enum('Node.Results.price')


class TestBuild:
    def test_two_specs_for_one_flag_raise(self):
        with pytest.raises(AmbiguousOverrideError):
            FlagIndex(
                [
                    NODE_PRICE_SPEC,
                    TimeSeriesFlagSpec(
                        name='Node.Results.price',
                        description='x',
                        unit=Units.EUR_per_MWh,
                        model_flag='Node.Model',
                    ),
                ]
            )

    def test_check_references(self):
        FlagIndex([NODE_MODEL_SPEC, NODE_PRICE_SPEC]).check_references()
        with pytest.raises(ValueError, match='Node.Model'):
            FlagIndex([NODE_PRICE_SPEC]).check_references()

    def test_two_families_that_cover_one_flag_raise(self):
        def family(name):
            return TimeSeriesFlagSpec(
                name=name,
                description='x',
                unit=Units.MW,
                model_flag='Node.Model',
                variants=(Dimension('v', 'v', lambda ds: {}),),
            )

        index = FlagIndex([family('a_{v}'), family('a_b_{v}')])
        with pytest.raises(AmbiguousOverrideError):
            index.get_spec('a_b_c')

    def test_union_keeps_same_specs_and_rejects_different_ones(self):
        a = FlagIndex([NODE_MODEL_SPEC, NODE_PRICE_SPEC])
        b = FlagIndex([NODE_MODEL_SPEC, LINE_MODEL_SPEC])
        assert set(FlagIndex.union([a, b]).static_flags) == {'Node.Model', 'Node.Results.price', 'Line.Model'}
        other = TimeSeriesFlagSpec(
            name='Node.Results.price', description='x', unit=Units.EUR_per_Wh, model_flag='Node.Model'
        )
        with pytest.raises(ValueError, match='Two different specs'):
            FlagIndex.union([a, FlagIndex([NODE_MODEL_SPEC, other])])


class TestRegisterNewFlag:
    def test_every_field_comes_from_the_new_spec(self):
        index = FlagIndex([NODE_MODEL_SPEC])
        index.register_new_flag(
            'Generator.Model',
            ItemTypeEnum.Model,
            'Generators.',
            index='Generator',
            topology_type=TopologyTypeEnum.NodeConnectedElement,
            memberships={'node': 'Node.Model'},
        )
        index.register_new_flag(
            'Generator.Results.p',
            ItemTypeEnum.TimeSeries,
            'Output.',
            unit=Units.MW,
            model_flag='Generator.Model',
        )
        assert index.get_item_type('Generator.Results.p') is ItemTypeEnum.TimeSeries
        assert index.get_linked_model_flag('Generator.Results.p') == 'Generator.Model'
        assert index.get_topology_type('Generator.Results.p') is TopologyTypeEnum.NodeConnectedElement
        assert index.get_memberships('Generator.Model') == {'node': 'Node.Model'}
