import dataclasses
import inspect

import pytest

from mesqual.enums import ItemTypeEnum, QuantityTypeEnum, TopologyTypeEnum
from mesqual.flag import (
    Dimension,
    FlagSpec,
    FrameSchema,
    Level,
    ModelFlagSpec,
    OtherFlagSpec,
    TimeSeriesFlagSpec,
    VariantValues,
)
from mesqual.units import Units
from tests.flag.specs import DIRECTION, LINE_FLOW_SPEC, NODE_MODEL_SPEC, NODE_PRICE_SPEC, SIDE


def price_spec(**fields) -> TimeSeriesFlagSpec:
    fields = {
        'name': 'Node.Results.x',
        'description': 'x',
        'unit': Units.EUR_per_MWh,
        'model_flag': 'Node.Model',
        **fields,
    }
    return TimeSeriesFlagSpec(**fields)


class TestSpecChecks:
    def test_time_series_spec_without_unit_raises(self):
        with pytest.raises(TypeError, match='unit'):
            TimeSeriesFlagSpec(name='x', description='x', model_flag='Node.Model')

    @pytest.mark.parametrize('unit', [Units.NaU, Units.MissingUnit, 'MW'])
    def test_time_series_spec_with_invalid_unit_raises(self, unit):
        with pytest.raises(ValueError, match='unit'):
            price_spec(unit=unit)

    def test_ambiguous_unit_needs_quantity_type(self, ambiguous_unit):
        with pytest.raises(ValueError, match='quantity_type'):
            price_spec(unit=ambiguous_unit)
        spec = price_spec(unit=ambiguous_unit, quantity_type=QuantityTypeEnum.INTENSIVE)
        assert spec.quantity_type is QuantityTypeEnum.INTENSIVE

    @pytest.mark.parametrize('unit', [Units.EUR_per_MW, Units.EUR_per_MW_per_period])
    def test_capacity_price_units_need_no_quantity_type(self, unit):
        assert price_spec(unit=unit).quantity_type is None

    def test_quantity_type_on_unambiguous_unit_raises(self):
        with pytest.raises(ValueError, match='quantity_type'):
            price_spec(quantity_type=QuantityTypeEnum.INTENSIVE)

    def test_model_spec_without_topology_type_raises(self):
        with pytest.raises(TypeError, match='topology_type'):
            ModelFlagSpec(name='Node.Model', description='x', index='Node')

    def test_placeholders_must_match_variants(self):
        with pytest.raises(ValueError):
            price_spec(name='Node.Results.price_{period}')
        with pytest.raises(ValueError):
            price_spec(variants=(SIDE,))

    def test_dimension_names_are_unique(self):
        with pytest.raises(ValueError, match='unique'):
            price_spec(name='x_{side}', variants=(SIDE,), params=(Dimension('side', 'x', default=1),))

    def test_variant_needs_values_and_takes_no_default(self):
        with pytest.raises(ValueError, match='needs values'):
            price_spec(name='x_{v}', variants=(Dimension('v', 'v'),))
        with pytest.raises(ValueError, match='no default'):
            price_spec(name='x_{v}', variants=(Dimension('v', 'v', {'a': 'a'}, default='a'),))

    def test_model_flag_none_needs_schema(self):
        with pytest.raises(ValueError, match='schema'):
            price_spec(model_flag=None)

    def test_schema_needs_one_time_level(self):
        with pytest.raises(ValueError, match='time'):
            price_spec(schema=FrameSchema(index=(Level('Node', 'object'),)))

    def test_memberships_only_on_model_specs(self):
        with pytest.raises(TypeError):
            price_spec(memberships={'node': 'Node.Model'})

    def test_edge_ends_need_edge_topology(self):
        with pytest.raises(ValueError, match='Edge'):
            ModelFlagSpec(
                name='Gen.Model',
                description='x',
                index='Gen',
                topology_type=TopologyTypeEnum.NodeConnectedElement,
                memberships={'a': 'Node.Model', 'b': 'Node.Model'},
                edge_ends=('a', 'b'),
            )

    def test_edge_ends_to_different_models_raise(self):
        with pytest.raises(ValueError, match='same model flag'):
            ModelFlagSpec(
                name='Line.Model',
                description='x',
                index='Line',
                topology_type=TopologyTypeEnum.Edge,
                memberships={'a': 'Node.Model', 'b': 'Zone.Model'},
                edge_ends=('a', 'b'),
            )

    def test_other_spec_needs_schema(self):
        with pytest.raises(TypeError, match='schema'):
            OtherFlagSpec(name='x', description='x')

    def test_item_type_follows_from_class(self):
        assert NODE_MODEL_SPEC.item_type is ItemTypeEnum.Model
        assert NODE_PRICE_SPEC.item_type is ItemTypeEnum.TimeSeries
        other = OtherFlagSpec(name='x', description='x', schema=FrameSchema(index=(Level('k', 'dimension'),)))
        assert other.item_type is ItemTypeEnum.Other


class TestModelSpecSchema:
    def test_schema_is_one_object_level_and_one_unnamed_column_level(self):
        assert NODE_MODEL_SPEC.schema == FrameSchema(
            index=(Level('Node', 'object'),),
            columns=(Level(None, 'dimension'),),
        )

    def test_schema_is_read_only(self):
        with pytest.raises(AttributeError):
            NODE_MODEL_SPEC.schema = FrameSchema(index=(Level('x', 'object'),))

    def test_schema_is_not_a_spec_field(self):
        assert 'schema' not in {f.name for f in dataclasses.fields(ModelFlagSpec)}

    def test_all_spec_classes_expose_schema(self):
        other = OtherFlagSpec(name='x', description='x', schema=FrameSchema(index=(Level('k', 'dimension'),)))
        for spec in (NODE_MODEL_SPEC, price_spec(schema=FrameSchema(index=(Level('time', 'time'),))), other):
            assert isinstance(spec.schema, FrameSchema)

    @pytest.mark.parametrize('spec_class', [ModelFlagSpec, TimeSeriesFlagSpec, OtherFlagSpec])
    def test_every_spec_class_overrides_the_base_schema(self, spec_class):
        assert inspect.getattr_static(spec_class, 'schema') is not inspect.getattr_static(FlagSpec, 'schema')

    def test_base_schema_raises_without_override(self):
        @dataclasses.dataclass(frozen=True, eq=False)
        class NoSchemaSpec(FlagSpec):
            item_type = ItemTypeEnum.Other

        with pytest.raises(NotImplementedError, match='NoSchemaSpec'):
            _ = NoSchemaSpec(name='x', description='x').schema


def test_class_for_item_type():
    assert FlagSpec.class_for(ItemTypeEnum.Model) is ModelFlagSpec
    assert FlagSpec.class_for(ItemTypeEnum.TimeSeries) is TimeSeriesFlagSpec
    assert FlagSpec.class_for(ItemTypeEnum.Other) is OtherFlagSpec


class TestSameAs:
    def test_equal_content_is_same_but_not_equal(self):
        copy = price_spec()
        assert price_spec().same_as(copy)
        assert price_spec() != copy

    def test_different_description_is_same(self):
        assert price_spec().same_as(price_spec(description='other text'))

    def test_different_version_unit_or_class_is_not_same(self):
        assert not price_spec().same_as(price_spec(version=2))
        assert not price_spec().same_as(price_spec(unit=Units.EUR_per_Wh))
        assert not NODE_MODEL_SPEC.same_as(price_spec(name='Node.Model'))

    def test_different_dimension_default_is_not_same(self):
        a = price_spec(params=(Dimension('p', 'p', {'x': 'x', 'y': 'y'}, default='x'),))
        b = price_spec(params=(Dimension('p', 'p', {'x': 'x', 'y': 'y'}, default='y'),))
        assert not a.same_as(b)


class TestVariants:
    def test_expansion_is_the_product_of_the_values(self):
        assert LINE_FLOW_SPEC.expand() == [
            'Line.Results.flow_sent_up',
            'Line.Results.flow_sent_net',
            'Line.Results.flow_received_up',
            'Line.Results.flow_received_net',
        ]

    def test_variant_of_round_trips(self):
        for flag in LINE_FLOW_SPEC.expand():
            variant = LINE_FLOW_SPEC.variant_of(flag)
            assert LINE_FLOW_SPEC.name.format(**variant.as_dict()) == flag
        assert LINE_FLOW_SPEC.variant_of('Line.Results.flow_received_net') == VariantValues(
            {'side': 'received', 'direction': 'net'}
        )

    def test_variant_of_unknown_flag_raises(self):
        with pytest.raises(KeyError):
            LINE_FLOW_SPEC.variant_of('Line.Results.flow_x_net')

    def test_covers(self):
        assert LINE_FLOW_SPEC.covers('Line.Results.flow_sent_up')
        assert not LINE_FLOW_SPEC.covers('Line.Results.flow')

    def test_flag_description_combines_variant_descriptions(self):
        assert LINE_FLOW_SPEC.flag_description('Line.Results.flow_received_net') == (
            'Power flow on the line. Side: where the flow leaves. Direction: up minus down.'
        )

    def test_schema_as_function_of_variant_values(self):
        def schema(variant):
            return FrameSchema(index=(Level('time', 'time'),), columns=(Level(f'Line_{variant.side}', 'object'),))

        spec = dataclasses.replace(LINE_FLOW_SPEC, schema=schema)
        assert spec.declared_schema('Line.Results.flow_sent_up').columns == (Level('Line_sent', 'object'),)

    def test_two_combinations_with_one_flag_raise(self):
        a = Dimension('a', 'a', {'x': 'x', 'x_y': 'x_y'})
        b = Dimension('b', 'b', {'y_z': 'y_z', 'z': 'z'})
        with pytest.raises(ValueError, match='two variant combinations'):
            price_spec(name='f_{a}_{b}', variants=(a, b))


class TestFamilies:
    FUEL = Dimension('fuel', 'Fuel.', lambda ds: {f: f'from {f}' for f in ds.fuels})
    SPEC = TimeSeriesFlagSpec(
        name='Node.Results.generation_{fuel}_{direction}',
        description='Generation.',
        unit=Units.MWh,
        model_flag='Node.Model',
        variants=(FUEL, DIRECTION),
    )

    class Source:
        fuels = ('gas', 'coal')

    def test_family_is_detected(self):
        assert self.SPEC.is_family
        assert not LINE_FLOW_SPEC.is_family

    def test_covers_by_template_without_values(self):
        assert self.SPEC.covers('Node.Results.generation_wind_net')
        assert not self.SPEC.covers('Node.Results.generation_wind_down')

    def test_variant_of_reads_static_values(self):
        variant = self.SPEC.variant_of('Node.Results.generation_gas_net')
        assert (variant.fuel, variant.direction) == ('gas', 'net')

    def test_values_come_from_the_dataset(self):
        assert self.SPEC.expand(self.Source()) == [
            'Node.Results.generation_gas_up',
            'Node.Results.generation_gas_net',
            'Node.Results.generation_coal_up',
            'Node.Results.generation_coal_net',
        ]
        self.SPEC.check_variant_values('Node.Results.generation_gas_up', self.Source())
        with pytest.raises(ValueError):
            self.SPEC.check_variant_values('Node.Results.generation_wind_up', self.Source())


class TestResolveParams:
    def test_default_and_given_value(self):
        assert NODE_PRICE_SPEC.resolve_params() == {'solution': 'final'}
        assert NODE_PRICE_SPEC.resolve_params({'solution': 'first'}) == {'solution': 'first'}

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="'best' is not a value of 'solution'"):
            NODE_PRICE_SPEC.resolve_params({'solution': 'best'})

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match=r"declares no parameters \['scenario'\]"):
            NODE_PRICE_SPEC.resolve_params({'scenario': 'low'})

    def test_values_follow_the_order_of_params(self):
        scenario = Dimension('scenario', 'x', {'low': 'low'}, default='low')
        side = Dimension('side', 'x', {'sent': 'sent'}, default='sent')
        spec = price_spec(params=(scenario, side))
        assert list(spec.resolve_params({'side': 'sent', 'scenario': 'low'})) == ['scenario', 'side']

    def test_dataset_reaches_the_values_function(self):
        seen = []
        fuel = Dimension('fuel', 'x', lambda dataset: seen.append(dataset) or {'coal': 'coal'}, default='coal')
        dataset = object()
        assert price_spec(params=(fuel,)).resolve_params(dataset=dataset) == {'fuel': 'coal'}
        assert seen == [dataset]

    def test_spec_without_params_gives_empty_dict(self):
        assert LINE_FLOW_SPEC.resolve_params() == {}
