"""Shipped aggregations, layer flag interpreters, overrides and version identity against hand-computed values."""

import dataclasses
import logging
from dataclasses import dataclass

from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from mesqual.layer_flags import (
    AreaAggregation,
    AreaFlagInterpreter,
    BorderAggregation,
    BorderEdgeSum,
    BorderFlowSum,
    BorderModel,
    PriceSpread,
    Sum,
    WeightedMean,
    edges_on_no_border,
    objects_without_area,
)
from mesqual.datasets import PlatformDataset
from mesqual.energy_data_handling.area_accounting import AreaBorderModelGenerator, AreaMembership
from mesqual.enums import TopologyTypeEnum
from mesqual.units import Units
from mesqual.flag import (
    AreaFlagSpec,
    BorderFlagSpec,
    Dimension,
    ModelFlagSpec,
    TimeSeriesFlagSpec,
    VariantValues,
)
from tests.helpers import InMemoryCache, only_entry
from tests.layer_flags.tables import (
    AREA_DEMAND,
    AREA_EXCHANGE,
    AREA_MODEL,
    AREA_PRICE,
    BORDER_CAPACITY,
    BORDER_FLOW,
    BORDER_MODEL_SPEC,
    DEMAND_SPEC,
    FLOW_SPEC,
    LINES,
    LINE_MODEL_SPEC,
    NODES,
    PRICE_SPEC,
    SPREAD,
    TIME,
    ZONE,
    TableInterpreter,
    TableModels,
    TablePlatform,
    TableResults,
    frame,
    platform,
)

ALL_SPECS = dict(
    area_specs=(AREA_MODEL, AREA_PRICE, AREA_DEMAND),
    border_specs=(BORDER_MODEL_SPEC, BORDER_CAPACITY, BORDER_FLOW, SPREAD),
)


@pytest.fixture(scope='module')
def ds():
    return platform(**ALL_SPECS)()


def series(values) -> pd.Series:
    return pd.Series(values, index=TIME, dtype=float)


def interpreter(ds, base):
    return next(ds.get_interpreter_instance(t) for t in ds.get_registered_interpreters() if issubclass(t, base))


class TestAreaAggregations:
    def test_sum_leaves_out_nodes_without_area(self, ds):
        result = ds.fetch('Zone.Results.demand_peak')
        assert result.columns.tolist() == ['A', 'B']
        pd.testing.assert_series_equal(result['A'], series([4.0, 2.0]), check_names=False)
        assert objects_without_area(ds, ds.flag_index.get_layer('Zone.Results.demand_peak')) == 1

    def test_weighted_mean_fills_the_weight_template(self, ds):
        peak = ds.fetch('Zone.Results.price_peak')
        pd.testing.assert_series_equal(peak['A'], series([25.0, 30.0]), check_names=False)
        offpeak = ds.fetch('Zone.Results.price_offpeak')
        pd.testing.assert_series_equal(offpeak['A'], series([21.0, 31.0]), check_names=False)

    @staticmethod
    def weighted_mean(values: pd.DataFrame, weights: pd.DataFrame, mapping: pd.Series) -> pd.DataFrame:
        dataset = SimpleNamespace(
            fetch=lambda flag, config=None, **params: weights,
            flag_index=SimpleNamespace(get_spec=lambda flag: SimpleNamespace(params=())),
        )
        layer = SimpleNamespace(index='zone')
        return WeightedMean(weight='w').aggregate(values, mapping, dataset, layer, VariantValues({}), None)

    def test_weighted_mean_with_zero_weights_gives_nan(self):
        values = pd.DataFrame({'n1': [10.0], 'n2': [30.0]})
        weights = pd.DataFrame({'n1': [0.0], 'n2': [0.0]})
        result = self.weighted_mean(values, weights, pd.Series({'n1': 'A', 'n2': 'A'}))
        assert result['A'].isna().all()

    def test_zero_weights_in_one_area_leave_the_other_area(self):
        values = pd.DataFrame({'n1': [10.0, 10.0], 'n2': [30.0, 30.0], 'n3': [50.0, 50.0], 'n4': [70.0, 70.0]})
        weights = pd.DataFrame({'n1': [0.0, 1.0], 'n2': [0.0, 3.0], 'n3': [1.0, 0.0], 'n4': [3.0, 0.0]})
        result = self.weighted_mean(values, weights, pd.Series({'n1': 'A', 'n2': 'A', 'n3': 'B', 'n4': 'B'}))
        assert result['A'].isna().tolist() == [True, False] and result['A'].iloc[1] == 25.0
        assert result['B'].isna().tolist() == [False, True] and result['B'].iloc[0] == 65.0

    @staticmethod
    def area_sum(values: pd.DataFrame) -> pd.DataFrame:
        mapping = pd.Series({'n1': 'A', 'n2': 'A', 'n3': 'B'})
        return Sum().aggregate(values, mapping, None, ZONE, None, None)

    def test_sum_of_an_area_whose_objects_are_all_nan_is_nan(self):
        values = frame({'n1': [1.0, float('nan')], 'n2': [1.0, float('nan')], 'n3': [5.0, 6.0]})
        result = self.area_sum(values)
        assert result['A'].isna().tolist() == [False, True]
        assert result['B'].tolist() == [5.0, 6.0]

    def test_sum_leaves_out_nan_where_one_object_has_a_value(self):
        values = frame({'n1': [1.0, 2.0], 'n2': [1.0, float('nan')], 'n3': [5.0, 6.0]})
        assert self.area_sum(values)['A'].tolist() == [2.0, 2.0]

    def test_sum_of_a_base_without_columns_keeps_the_index(self):
        result = self.area_sum(frame({}))
        assert result.index.equals(TIME)
        assert result.columns.empty

    def test_area_model(self, ds):
        model = ds.fetch('Zone.Model')
        assert model.index.name == 'zone'
        assert model['node_count'].to_dict() == {'A': 2, 'B': 1}


class TestBorderAggregations:
    def test_edge_to_a_node_without_zone_is_on_no_border(self, ds):
        assert edges_on_no_border(ds, ds.flag_index.get_layer('ZoneBorder.Model')) == 2

    def test_border_set_and_edge_counts_agree_with_an_edge_to_a_node_without_zone(self, ds):
        counts = ds.fetch('ZoneBorder.Model')[['n_edges_with_border', 'n_edges_against_border']]
        assert counts.to_dict('index') == {
            'A - B': {'n_edges_with_border': 1, 'n_edges_against_border': 1},
            'B - A': {'n_edges_with_border': 1, 'n_edges_against_border': 1},
        }

    def test_border_total_adds_lines_against_border_unchanged(self, ds):
        result = ds.fetch('ZoneBorder.Results.flow_total')
        border = ds.fetch('ZoneBorder.Model').query("area_from == 'A'").index[0]
        pd.testing.assert_series_equal(result[border], series([195.0] * 2), check_names=False)

    def test_border_model_has_the_canonical_border_of_each_pair(self, ds):
        model = ds.fetch('ZoneBorder.Model')
        assert model['canonical_border'].to_dict() == {'A - B': 'A - B', 'B - A': 'A - B'}
        assert model['is_canonical'].to_dict() == {'A - B': True, 'B - A': False}

    def test_unknown_direction_raises(self, ds):
        with pytest.raises(ValueError, match='unknown'):
            ds.fetch('ZoneBorder.Results.flow_unknown')

    @pytest.mark.parametrize('no_area', [None, float('nan')])
    def test_a_node_without_area_gives_no_border(self, no_area):
        nodes = NODES.assign(zone=['A', 'A', 'B', no_area])
        models = type('Models', (TableModels,), {'tables': {'Node.Model': nodes, 'Line.Model': LINES}})
        base = type('NoAreaPlatform', (TablePlatform,), {})
        base.register_interpreter(models, override_interpreter=TableModels, version=1)
        ds = platform(**ALL_SPECS, base=base)()
        assert sorted(ds.fetch('ZoneBorder.Model').index) == ['A - B', 'B - A']
        assert sorted(ds.fetch('ZoneBorder.Results.flow_up').columns) == ['A - B', 'B - A']
        assert sorted(ds.fetch('ZoneBorder.Results.spread_peak').columns) == ['A - B', 'B - A']

    def test_price_spread_of_a_missing_area_is_nan(self):
        model = pd.DataFrame({'area_from': ['A'], 'area_to': ['C']}, index=['A - C'])
        values = pd.DataFrame({'A': [1.0]})
        result = PriceSpread().aggregate(values, model, None, type('L', (), {'index': 'zone_border'}), None, None)
        assert result['A - C'].isna().all()


class TestBorderEdgeSum:
    COST_SPEC = TimeSeriesFlagSpec(
        name='Line.Results.cost',
        description='Line cost.',
        model_flag='Line.Model',
        unit=Units.EUR,
    )
    COST = frame({'l1': [1.0, float('nan')], 'l2': [10.0, float('nan')], 'l3': [100.0, 100.0], 'l4': [1e3, 1e3]})
    """l1 runs A to B, l2 B to A, l3 inside A, l4 from B to a node without zone."""
    BORDER_COST = BorderFlagSpec(
        name='{border_layer}.Results.cost',
        base=COST_SPEC,
        aggregation=BorderEdgeSum(),
        description='x',
    )

    @pytest.fixture
    def ds(self):
        results = type(
            'Costs',
            (TableInterpreter,),
            {
                'flag_specs': (self.COST_SPEC,),
                'tables': {'Line.Results.cost': self.COST},
            },
        )
        base = type('CostPlatform', (TablePlatform,), {})
        base.register_interpreter(results)
        return platform(border_specs=(BORDER_MODEL_SPEC, self.BORDER_COST), base=base)()

    def test_both_borders_of_a_pair_sum_the_edges_in_both_directions(self, ds):
        result = ds.fetch('ZoneBorder.Results.cost')
        assert sorted(result.columns) == ['A - B', 'B - A']
        assert result['A - B'].iloc[0] == 11.0 and result['B - A'].iloc[0] == 11.0

    def test_a_time_step_where_all_edges_of_a_border_are_nan_is_nan(self, ds):
        assert ds.fetch('ZoneBorder.Results.cost').iloc[1].isna().all()

    @pytest.mark.parametrize(
        'edge_areas_of',
        [
            lambda generator: generator.line_areas,
            lambda generator: AreaMembership.edge_areas(LINES, NODES['zone'], 'node_from', 'node_to'),
        ],
        ids=['line_areas', 'AreaMembership.edge_areas'],
    )
    def test_sum_by_border_with_plain_frames_and_default_identifiers(self, edge_areas_of):
        generator = AreaBorderModelGenerator(NODES, LINES, 'zone', 'node_from', 'node_to')
        result = BorderEdgeSum.sum_by_border(
            self.COST,
            edge_areas_of(generator),
            generator.generate_area_border_model(),
        )
        assert result.iloc[0].to_dict() == {'A - B': 11.0, 'B - A': 11.0}
        assert result.iloc[1].isna().all()

    def test_input_flags_are_the_inputs_of_the_edge_areas(self, ds):
        layer = ds.flag_index.get_layer('ZoneBorder.Results.cost')
        variant = ds.flag_index.get_layer_variant('ZoneBorder.Results.cost')
        assert BorderEdgeSum().input_flags(ds, layer, variant) == ('Node.Model', 'Line.Model')

    def test_border_flow_total_equals_the_edge_sum_of_the_total(self):
        edge_flow = BorderFlagSpec(
            name='{border_layer}.Results.edge_flow_{direction}',
            base=FLOW_SPEC,
            aggregation=BorderEdgeSum(),
            description='x',
        )
        ds = platform(border_specs=(BORDER_MODEL_SPEC, BORDER_FLOW, edge_flow))()
        total = ds.fetch('ZoneBorder.Results.flow_total')
        assert total.loc[:, ['A - B', 'B - A']].eq(195.0).all().all()
        pd.testing.assert_frame_equal(total, ds.fetch('ZoneBorder.Results.edge_flow_total'))


class TestFetchPath:
    def test_records_nested_fetches_and_version_identity(self, ds):
        ds.fetch('Zone.Results.price_offpeak')
        record = ds.get_fetch_record('Zone.Results.price_offpeak')
        flags = [d.flag for d in record.iter_dependencies()]
        assert {'Node.Model', 'Node.Results.price_offpeak', 'Node.Results.demand_offpeak'} <= set(flags)
        assert record.version[1] == ('WeightedMean', {'weight': 'Node.Results.demand_{period}', 'version': 2})

    def test_aggregation_version_bump_changes_the_identity(self):
        bumped = AreaFlagSpec(
            name='{area_layer}.Results.demand_{period}',
            base=DEMAND_SPEC,
            aggregation=Sum(version=Sum().version + 1),
            description='Area demand.',
        )
        old = platform(area_specs=(AREA_DEMAND,))().version_identity('Zone.Results.demand_offpeak')
        new = platform(area_specs=(bumped,))().version_identity('Zone.Results.demand_offpeak')
        assert old != new and new[1] == ('Sum', {'version': Sum().version + 1})

    def test_override_of_the_base_reaches_area_and_border_flags(self):
        class DoubledPrices(TableResults):
            def _fetch(self, flag, effective_config, **params):
                return super()._fetch(flag, effective_config, **params) * 2

        study = type('Study', (platform(**ALL_SPECS),), {})
        study.register_interpreter(DoubledPrices, override_interpreter=TableResults, version=1)
        ds = study()
        pd.testing.assert_series_equal(
            ds.fetch('Zone.Results.price_peak')['A'],
            series([50.0, 60.0]),
            check_names=False,
        )
        border = ds.fetch('ZoneBorder.Model').query("area_from == 'A'").index[0]
        pd.testing.assert_series_equal(
            ds.fetch('ZoneBorder.Results.spread_peak')[border],
            series([50.0, 60.0]),
            check_names=False,
        )

    def test_layer_specs_on_a_replaced_spec_take_its_new_parameter(self):
        cap = Dimension('cap', 'Price cap.', default=1000.0)
        own_spec = dataclasses.replace(PRICE_SPEC, params=(cap,), description='Capped node price.')

        class CappedPrices(TableResults):
            flag_specs = (own_spec, *(spec for spec in TableResults.flag_specs if spec.name != own_spec.name))

            def _fetch(self, flag, effective_config, cap=1000.0, **params):
                return super()._fetch(flag, effective_config, **params).clip(upper=cap)

        study = type('Study', (platform(**ALL_SPECS),), {})
        study.register_interpreter(CappedPrices, override_interpreter=TableResults, version=1)
        ds = study()
        area_spec = ds.flag_index.get_spec('Zone.Results.price_peak')
        assert area_spec.base is own_spec
        assert ds.flag_index.get_spec('ZoneBorder.Results.spread_peak').base is area_spec
        assert 'cap' in study.describe_class_flags().loc['{border_layer}.Results.spread_{period}', 'params']
        assert (ds.fetch('Zone.Results.price_peak', cap=5.0) == 5.0).all().all()
        assert (ds.fetch('ZoneBorder.Results.spread_peak', cap=5.0) == 0.0).all().all()
        pd.testing.assert_frame_equal(
            ds.fetch('Zone.Results.price_peak'),
            platform(**ALL_SPECS)().fetch('Zone.Results.price_peak'),
        )

    def test_layer_specs_on_a_spec_that_the_override_leaves_out_are_left_out(self):
        class NoPrices(TableResults):
            flag_specs = tuple(spec for spec in TableResults.flag_specs if spec.name != PRICE_SPEC.name)

        study = type('Study', (platform(**ALL_SPECS),), {})
        study.register_interpreter(NoPrices, override_interpreter=TableResults, version=1)
        ds = study()
        left_out = {AREA_PRICE.name, SPREAD.name}
        assert not left_out & set(study.describe_class_flags().index)
        assert not left_out & set(ds.describe_flags().index)
        assert not {'Zone.Results.price_peak', 'ZoneBorder.Results.spread_peak'} & ds.accepted_flags
        assert 'Zone.Results.demand_peak' in ds.accepted_flags

    def test_layer_spec_on_a_base_that_no_interpreter_declares_is_accepted_and_unavailable(self):
        base = TimeSeriesFlagSpec(name='Node.Results.other', description='x', model_flag='Node.Model', unit=Units.MW)
        spec = AreaFlagSpec(name='{area_layer}.Results.other', base=base, aggregation=Sum(), description='x')
        ds = platform(area_specs=(AREA_MODEL, spec))()
        assert 'Zone.Results.other' in ds.accepted_flags
        availability = ds.availability('Zone.Results.other')
        assert availability.available is False
        assert 'Node.Results.other' in availability.reason

    def test_override_of_a_layer_spec_template(self):
        parent = platform(**ALL_SPECS)
        base = next(t for t in parent.get_registered_interpreters() if t.__name__ == 'Areas')

        class CappedAreaPrices(base):
            def _fetch(self, flag, effective_config, **params):
                return super()._fetch(flag, effective_config, **params).clip(upper=26.0)

        study = type('Study', (parent,), {})
        study.register_interpreter(CappedAreaPrices, override_interpreter=base, version=1)
        ds = study()
        assert ds.fetch('Zone.Results.price_peak')['A'].max() == 26.0
        assert ds.version_identity('Zone.Results.price_peak')[-1][0].endswith('CappedAreaPrices')

    def test_override_of_a_layer_spec_template_answers_on_the_layers_of_the_class(self):
        parent = platform(**ALL_SPECS)

        base = parent.get_answering_interpreter('Zone.Results.price_peak')

        class CappedAreaPrices(base):
            pass

        study = type('Study', (parent,), {})
        study.register_interpreter(CappedAreaPrices, override_interpreter=base, version=1)
        with_second_layer = type('WithSecondLayer', (study,), {})
        with_second_layer.register_area_layer(dataclasses.replace(ZONE, name='BiddingZone'))
        assert 'BiddingZone.Results.price_peak' not in study().get_flags_by_interpreter()[CappedAreaPrices]
        assert 'BiddingZone.Results.price_peak' in with_second_layer().get_flags_by_interpreter()[CappedAreaPrices]

    def test_layer_spec_template_override_through_the_lookup(self):
        parent = platform(**ALL_SPECS)
        base = parent.get_answering_interpreter('Zone.Results.price_peak')

        class CappedAreaPrices(base):
            def _fetch(self, flag, effective_config, **params):
                return super()._fetch(flag, effective_config, **params).clip(upper=26.0)

        study = type('Study', (parent,), {})
        study.register_interpreter(CappedAreaPrices, override_interpreter=base, version=1)
        assert study.get_answering_interpreter('Zone.Results.price_peak') is CappedAreaPrices
        assert study().fetch('Zone.Results.price_peak')['A'].max() == 26.0

    def test_custom_aggregations_record_extra_fetches(self):
        @dataclass(frozen=True, kw_only=True)
        class DemandShare(AreaAggregation):
            def aggregate(self, values, mapping, dataset, layer, variant, effective_config, **params):
                total = dataset.fetch('Node.Results.demand_peak', effective_config).sum(axis=1)
                area_sum = Sum().aggregate(values, mapping, dataset, layer, variant, effective_config)
                return area_sum.div(total, axis=0)

        @dataclass(frozen=True, kw_only=True)
        class PriceOfStart(BorderAggregation):
            def aggregate(self, values, border_model, dataset, layer, variant, effective_config, **params):
                prices = dataset.fetch('Zone.Results.price_peak', effective_config)
                return pd.DataFrame({b: prices[r.area_from] for b, r in border_model.iterrows()})

        share = AreaFlagSpec(
            name='{area_layer}.Results.share_{period}',
            base=DEMAND_SPEC,
            aggregation=DemandShare(),
            description='x',
        )
        start = BorderFlagSpec(
            name='{border_layer}.Results.start_price',
            base=AREA_DEMAND,
            aggregation=PriceOfStart(),
            description='x',
            consumes=('period',),
        )
        ds = platform(area_specs=(AREA_PRICE, AREA_DEMAND, share), border_specs=(BORDER_MODEL_SPEC, start))()
        ds.fetch('Zone.Results.share_offpeak')
        flags = [d.flag for d in ds.get_fetch_record('Zone.Results.share_offpeak').iter_dependencies()]
        assert 'Node.Results.demand_peak' in flags
        ds.fetch('ZoneBorder.Results.start_price')
        flags = [d.flag for d in ds.get_fetch_record('ZoneBorder.Results.start_price').iter_dependencies()]
        assert 'Zone.Results.price_peak' in flags


class TestParams:
    LEVEL = Dimension('level', 'Scale of the result.', {'one': '1', 'ten': '10'}, default='one')

    def test_spec_params_reach_aggregate_and_not_the_base_fetch(self):
        seen = {}

        @dataclass(frozen=True, kw_only=True)
        class Scaled(AreaAggregation):
            def aggregate(self, values, mapping, dataset, layer, variant, effective_config, level, **params):
                seen['params'] = params
                factor = {'one': 1, 'ten': 10}[level]
                return Sum().aggregate(values, mapping, dataset, layer, variant, effective_config) * factor

        spec = AreaFlagSpec(
            name='{area_layer}.Results.scaled_{period}',
            base=DEMAND_SPEC,
            aggregation=Scaled(),
            description='x',
            params=(self.LEVEL,),
        )
        ds = platform(area_specs=(spec,))()
        result = ds.fetch('Zone.Results.scaled_peak', level='ten')
        assert result['A'].tolist() == [40.0, 20.0]
        record = ds.get_fetch_record('Zone.Results.scaled_peak', level='ten')
        base = [d for d in record.iter_dependencies() if d.flag == 'Node.Results.demand_peak']
        assert base and all('level' not in d.params for d in base)


class TestSpecCheck:
    def test_weight_with_a_placeholder_that_is_no_variant_raises_at_declaration(self):
        with pytest.raises(ValueError, match=r"placeholders \['product'\]"):
            AreaFlagSpec(
                name='{area_layer}.Results.price_{period}',
                base=PRICE_SPEC,
                description='x',
                aggregation=WeightedMean(weight='Node.Results.demand_{product}'),
            )

    def test_weight_with_the_variants_of_the_spec_is_accepted(self):
        AreaFlagSpec(
            name='{area_layer}.Results.price_{period}',
            base=PRICE_SPEC,
            description='x',
            aggregation=WeightedMean(weight='Node.Results.demand_{period}'),
        )

    def test_border_flow_sum_on_a_base_without_direction_raises_at_declaration(self):
        with pytest.raises(ValueError, match="needs a 'direction' variant"):
            BorderFlagSpec(
                name='{border_layer}.Results.sum', base=LINE_MODEL_SPEC, aggregation=BorderFlowSum(), description='x'
            )


class TestAvailability:
    def test_unknown_without_parent_check(self):
        ds = platform(**ALL_SPECS)()
        assert interpreter(ds, AreaFlagInterpreter)._is_available('Zone.Results.price_peak').available is None

    def test_false_for_a_border_base_without_border_model(self):
        ds = platform(area_specs=(AREA_EXCHANGE,), border_specs=(BORDER_FLOW,))()
        assert interpreter(ds, AreaFlagInterpreter)._is_available('Zone.Results.exchange').available is False


class TestBorderGeometry:
    def test_border_model_gets_geometry_from_area_polygons(self):
        from mesqual.testing import DummyDataset, DummyScenario

        model = DummyDataset(DummyScenario()).fetch('CountryBorder.Model')
        assert set(model.index) == {'C1 - C2', 'C2 - C1'}
        assert model['geo_line_string'].notna().all()


class TestBorderGeometryOfAreasThatDoNotTouch:
    GEOMETRY_COLUMNS = ['projection_point', 'azimuth_angle', 'geo_line_string']
    ZONE_MODEL_SPEC = ModelFlagSpec(
        name='Zone.Model', index='zone', topology_type=TopologyTypeEnum.Area, description='x'
    )

    POLYGONS = [box(5, 50, 6, 51), box(8, 50, 9, 51), box(6.5, 49, 7.5, 52)]
    ZONES = pd.Index(['A', 'B', 'C'], name='zone')
    """A and B do not touch. C lies between them, so the border path goes around C."""

    def border_model(self, zones: pd.DataFrame, aggregation: BorderModel = BorderModel()) -> pd.DataFrame:
        models = type(
            'Zones',
            (TableInterpreter,),
            {
                'flag_specs': (self.ZONE_MODEL_SPEC,),
                'tables': {'Zone.Model': zones},
            },
        )
        base = type('ZonePlatform', (TablePlatform,), {})
        base.register_interpreter(models)
        spec = BorderFlagSpec(
            name='{border_layer}.Model', base=LINE_MODEL_SPEC, aggregation=aggregation, description='x'
        )
        return platform(border_specs=(spec,), base=base)().fetch('ZoneBorder.Model')

    def test_border_between_areas_that_do_not_touch_gets_geometry_without_a_crs(self, caplog):
        model = self.border_model(pd.DataFrame({'geometry': self.POLYGONS}, index=self.ZONES))
        assert model[self.GEOMETRY_COLUMNS].notna().all().all()
        assert not model['areas_touch'].any()
        assert not [r for r in caplog.records if r.levelname == 'WARNING']

    def test_azimuth_of_a_projected_crs_equals_the_azimuth_in_lon_lat(self):
        lon_lat = gpd.GeoDataFrame(geometry=self.POLYGONS, index=self.ZONES, crs='EPSG:4326')
        expected = self.border_model(pd.DataFrame(lon_lat))['azimuth_angle']
        projected = self.border_model(pd.DataFrame(lon_lat.to_crs('EPSG:3035')))
        assert projected.loc['A - B', 'geo_line_string'].bounds[0] > 1e5, 'coordinates in metres of EPSG:3035'
        assert (projected['azimuth_angle'] - expected).abs().max() < 2.0

    def test_line_of_each_border_runs_from_its_first_area(self):
        model = self.border_model(pd.DataFrame({'geometry': self.POLYGONS}, index=self.ZONES))
        polygons = dict(zip(self.ZONES, self.POLYGONS))
        for border, row in model.iterrows():
            start = Point(row['geo_line_string'].coords[0])
            assert start.distance(polygons[row['area_from']]) < start.distance(polygons[row['area_to']]), border

    def test_border_model_without_polygons_has_nan_geometry_columns(self):
        model = self.border_model(pd.DataFrame({'name': ['a', 'b']}, index=pd.Index(['A', 'B'], name='zone')))
        assert model[list(self.GEOMETRY_COLUMNS) + ['areas_touch']].isna().all().all()

    def test_border_model_with_geometry_false_has_nan_geometry_columns(self):
        zones = pd.DataFrame({'geometry': self.POLYGONS}, index=self.ZONES)
        model = self.border_model(zones, BorderModel(geometry=False))
        assert model[list(self.GEOMETRY_COLUMNS) + ['areas_touch']].isna().all().all()

    def test_failed_geometry_logs_a_warning_and_prints_nothing(self, caplog, capsys):
        zones = pd.DataFrame({'geometry': [box(5, 50, 6, 51)]}, index=pd.Index(['A'], name='zone'))
        with caplog.at_level(logging.WARNING):
            model = self.border_model(zones)
        assert model[self.GEOMETRY_COLUMNS].isna().all().all()
        assert any('A - B' in r.getMessage() for r in caplog.records if r.levelname == 'WARNING')
        assert capsys.readouterr().out == ''


class TestCacheSeparation:
    """Version identities in the stored records and in the recorded dependencies."""

    ZONE_PRICE = 'Zone.Results.price_peak'

    @staticmethod
    def fetch_from_both(first, second, flag) -> tuple[InMemoryCache, PlatformDataset, PlatformDataset]:
        cache = InMemoryCache()
        first_dataset, second_dataset = (
            first(name='scenario', cache=cache),
            second(name='scenario', cache=cache),
        )
        first_dataset.fetch(flag)
        second_dataset.fetch(flag)
        return cache, first_dataset, second_dataset

    @staticmethod
    def dependency_versions(record, flag) -> list[tuple]:
        return [dependency.version for dependency in record.iter_dependencies() if dependency.flag == flag]

    @staticmethod
    def study(version: int) -> type:
        class DoubledPrices(TableResults):
            def _fetch(self, flag, effective_config, **params):
                return super()._fetch(flag, effective_config, **params) * 2

        study = type('Layered', (platform(**ALL_SPECS),), {})
        study.register_interpreter(DoubledPrices, override_interpreter=TableResults, version=version)
        return study

    def test_override_version_bump_replaces_the_entry_and_the_layer_flag_entry(self):
        node_price = 'Node.Results.price_peak'
        cache, first, second = self.fetch_from_both(self.study(1), self.study(2), self.ZONE_PRICE)
        assert first.version_identity(node_price) != second.version_identity(node_price)
        assert only_entry(cache, second, node_price)[0].version == second.version_identity(node_price)
        zone, _ = only_entry(cache, second, self.ZONE_PRICE)
        assert second.version_identity(node_price) in self.dependency_versions(zone, node_price)
        assert first.version_identity(node_price) not in self.dependency_versions(zone, node_price)

    def test_aggregation_version_bump_replaces_the_entry_of_the_layer_flag(self):
        bumped = AreaFlagSpec(
            name='{area_layer}.Results.demand_{period}',
            base=DEMAND_SPEC,
            aggregation=Sum(version=Sum().version + 1),
            description='Area demand.',
        )
        first, second = platform(area_specs=(AREA_DEMAND,)), platform(area_specs=(bumped,))
        cache, first, second = self.fetch_from_both(first, second, 'Zone.Results.demand_peak')
        record, _ = only_entry(cache, second, 'Zone.Results.demand_peak')
        version = record.version
        assert version == second.version_identity('Zone.Results.demand_peak')
        assert version != first.version_identity('Zone.Results.demand_peak')

    def test_weight_flag_version_bump_makes_the_layer_flag_entry_stale(self):
        bumped_demand = dataclasses.replace(DEMAND_SPEC, version=2)
        results = type('TableResults', (TableResults,), {'flag_specs': (PRICE_SPEC, bumped_demand)})
        base = type(
            'TablePlatform',
            (PlatformDataset,),
            {
                'build_source_inventory': TablePlatform.build_source_inventory,
                'get_child_dataset_type': TablePlatform.get_child_dataset_type,
            },
        )
        base.register_interpreter(TableModels)
        base.register_interpreter(results)
        first, second = platform(area_specs=(AREA_PRICE,)), platform(area_specs=(AREA_PRICE,), base=base)
        cache, first, second = self.fetch_from_both(first, second, self.ZONE_PRICE)
        demand = 'Node.Results.demand_peak'
        assert first.version_identity(demand) != second.version_identity(demand)
        assert set(self.dependency_versions(only_entry(cache, second, self.ZONE_PRICE)[0], demand)) == {
            second.version_identity(demand)
        }
