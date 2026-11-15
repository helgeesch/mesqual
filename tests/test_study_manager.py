from pathlib import Path

import pandas as pd
import pytest

from mesqual.datasets import (
    DatasetComparison,
    DatasetConcatCollection,
    DatasetConcatCollectionOfComparisons,
    DatasetConfig,
)
from mesqual.enums import ComparisonTypeEnum
from mesqual.study_manager import StudyManager
from mesqual.testing import DummyDataset, DummyScenario


def test_study_folder_setter_stores_a_path(tmp_path):
    study = StudyManager(DatasetConcatCollection([]), DatasetConcatCollectionOfComparisons([]))
    study.study_folder = str(tmp_path)
    assert isinstance(study.study_folder, Path)
    assert study.study_folder == tmp_path


@pytest.mark.parametrize('scen_axis, comp_axis', [(0, 1), (1, 0), (0, 0)])
def test_collections_that_stack_rows_raise(scen_axis, comp_axis):
    scen = DatasetConcatCollection([], name='scen', concat_axis=scen_axis)
    comp = DatasetConcatCollectionOfComparisons([], name='comp', concat_axis=comp_axis)
    with pytest.raises(ValueError, match='StudyManager needs concat_axis=1, got 0'):
        StudyManager(scen, comp)


@pytest.mark.parametrize('scen_top, comp_top', [(False, True), (True, False)])
def test_concat_level_at_the_bottom_raises(scen_top, comp_top):
    scen = DatasetConcatCollection([], name='scen', concat_top=scen_top)
    comp = DatasetConcatCollectionOfComparisons([], name='comp', concat_top=comp_top)
    with pytest.raises(ValueError, match='StudyManager needs concat_top=True, got False'):
        StudyManager(scen, comp)


def test_different_concat_level_names_raise():
    scen = DatasetConcatCollection([], name='scen', concat_level_name='scenario')
    comp = DatasetConcatCollectionOfComparisons([], name='comp')
    with pytest.raises(ValueError, match="needs the concat_level_name of scen, 'scenario', got 'dataset'"):
        StudyManager(scen, comp)


def test_one_custom_concat_level_name_on_both_is_accepted():
    scen = DatasetConcatCollection([], name='scen', concat_level_name='scenario')
    comp = DatasetConcatCollectionOfComparisons([], name='comp', concat_level_name='scenario')
    assert StudyManager(scen, comp).scen_comp.concat_level_name == 'type'


def test_concat_level_name_of_the_type_level_raises():
    name = 'type'
    scen = DatasetConcatCollection([], name='scen', concat_level_name=name)
    comp = DatasetConcatCollectionOfComparisons([], name='comp', concat_level_name=name)
    with pytest.raises(ValueError, match=f"needs a concat_level_name other than '{name}'"):
        StudyManager(scen, comp)


def test_concat_top_1_is_stored_as_true_and_accepted():
    scen = DatasetConcatCollection([], name='scen', concat_top=1)
    comp = DatasetConcatCollectionOfComparisons([], name='comp', concat_top=1)
    assert scen.concat_top is True
    assert scen.version_identity('x')[0] == DatasetConcatCollection([], name='scen').version_identity('x')[0]
    StudyManager(scen, comp)


class TestStudyConfig:
    """The config of a study is the instance config of `scen`, `comp` and `scen_comp`."""

    DEMAND = 'Demand.Results.demand_total'

    @staticmethod
    def study(config: DatasetConfig | None = None) -> StudyManager:
        scenarios = [
            DummyDataset(DummyScenario(name=name, days=1, resolution='15min', demand_factor=factor))
            for name, factor in [('base', 1.0), ('high', 1.2)]
        ]
        return StudyManager.factory_from_scenarios(scenarios, [('high', 'base')], config=config)

    @pytest.mark.parametrize('collection', ['scen', 'comp', 'scen_comp'])
    def test_granularity_applies_to_each_collection(self, collection):
        native = getattr(self.study(), collection).fetch('Node.Results.price')
        hourly = getattr(self.study(DatasetConfig(granularity='1h')), collection).fetch('Node.Results.price')
        assert len(native) == 96
        pd.testing.assert_frame_equal(hourly, native.resample('1h').mean(), check_freq=False)

    @pytest.mark.parametrize('collection', ['scen', 'comp', 'scen_comp'])
    def test_param_defaults_apply_to_each_collection(self, collection):
        native = getattr(self.study(), collection).fetch(self.DEMAND)
        first = getattr(self.study(DatasetConfig(param_defaults={'solution': 'first'})), collection).fetch(self.DEMAND)
        pd.testing.assert_frame_equal(first, native * 0.98)

    def test_a_scenario_added_later_gets_the_study_config_and_keeps_its_own(self):
        study = self.study(DatasetConfig(granularity='1h'))
        late = DummyDataset(DummyScenario(name='late', days=1, resolution='15min'))
        study.add_scenario(late)
        prices = study.scen.fetch('Node.Results.price')
        assert len(prices['late']) == 24
        assert late.instance_config.granularity is None
        assert len(late.fetch('Node.Results.price')) == 96

    def test_a_comparison_added_later_gets_the_study_config_and_keeps_its_own(self):
        study = self.study(DatasetConfig(granularity='1h'))
        late = DatasetComparison(study.scen.get_dataset('base'), study.scen.get_dataset('high'))
        study.add_comparison(late)
        assert len(study.comp.fetch('Node.Results.price')['base vs high']) == 24
        assert late.instance_config.granularity is None
        assert len(late.fetch('Node.Results.price')) == 96

    def test_each_collection_has_its_own_copy(self):
        study = self.study(DatasetConfig(granularity='1h'))
        study.comp.set_instance_config_kwargs(granularity='D')
        assert study.scen.instance_config.granularity == '1h'
        assert study.scen_comp.instance_config.granularity == '1h'

    def test_the_study_config_merges_on_top_of_the_collection_config(self):
        scen = DatasetConcatCollection([], config=DatasetConfig(granularity='D', use_cache=False))
        study = StudyManager(scen, DatasetConcatCollectionOfComparisons([]), config=DatasetConfig(granularity='1h'))
        assert study.scen.instance_config.granularity == '1h'
        assert study.scen.instance_config.use_cache is False
        assert study.comp.instance_config.granularity == '1h'

    def test_merge_instance_config_keeps_other_fields_and_copies(self):
        dataset = DatasetConcatCollection([], config=DatasetConfig(granularity='D', use_cache=False))
        upper = DatasetConfig(granularity='1h')
        dataset.merge_instance_config(upper)
        upper.granularity = '2h'
        assert dataset.instance_config.granularity == '1h'
        assert dataset.instance_config.use_cache is False
        empty = DatasetConcatCollection([])
        empty.merge_instance_config(upper)
        assert empty.instance_config.granularity == '2h'
        assert empty._config is not upper

    def test_without_config_the_collections_keep_their_own(self):
        scen = DatasetConcatCollection([], config=DatasetConfig(granularity='1h'))
        study = StudyManager(scen, DatasetConcatCollectionOfComparisons([]))
        assert study.scen.instance_config.granularity == '1h'
        assert study.comp.instance_config.granularity is None

    def test_a_comparison_parameter_raises_on_the_scenarios(self):
        study = self.study(DatasetConfig(param_defaults={'comparison_type': ComparisonTypeEnum.VARIATION}))
        variation = study.comp.fetch('Node.Results.price')['high vs base']
        pd.testing.assert_frame_equal(variation, study.scen.get_dataset('high').fetch('Node.Results.price'))
        with pytest.raises(ValueError, match='comparison_type'):
            study.scen.fetch('Node.Results.price')
