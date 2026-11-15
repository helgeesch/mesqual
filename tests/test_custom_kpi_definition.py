import pytest

from mesqual.kpis.definitions.custom import CustomKPIDefinition
from mesqual.units import Units
from tests.helpers import FrameDataset


class PriceDefinition(CustomKPIDefinition):
    def get_unit(self):
        return Units.EUR_per_MWh

    def required_flags(self) -> set:
        return {FrameDataset.FLAG}


class BatchDefinition(PriceDefinition):
    def compute_batch(self, dataset, objects):
        raise NotImplementedError('missing feature inside user code')


class WorkingBatchDefinition(PriceDefinition):
    def compute_batch(self, dataset, objects):
        names = ['a', 'b'] if objects == 'auto_from_batch_generation' else objects
        return {name: 2.0 for name in names}


class PerObjectDefinition(PriceDefinition):
    def compute_for_object(self, dataset, object_name):
        return 1.0


class EmptyDefinition(PriceDefinition):
    pass


def make_definition(definition_class):
    return definition_class('items.price', model_flag=FrameDataset.FLAG, objects=['a', 'b'])


def test_error_inside_compute_batch_reaches_the_caller():
    with pytest.raises(NotImplementedError, match='missing feature inside user code'):
        make_definition(BatchDefinition).generate_kpis(FrameDataset(name='ds'))


def test_per_object_path_computes_one_kpi_per_object():
    kpis = make_definition(PerObjectDefinition).generate_kpis(FrameDataset(name='ds'))
    assert [kpi.attributes.object_name for kpi in kpis] == ['a', 'b']


def test_definition_without_compute_method_raises():
    with pytest.raises(NotImplementedError, match='Must override'):
        make_definition(EmptyDefinition).generate_kpis(FrameDataset(name='ds'))


def test_batch_path_computes_one_kpi_per_object():
    kpis = make_definition(WorkingBatchDefinition).generate_kpis(FrameDataset(name='ds'))
    assert [(kpi.attributes.object_name, kpi.value) for kpi in kpis] == [('a', 2.0), ('b', 2.0)]


def test_auto_from_batch_generation_without_compute_batch_raises():
    definition = PerObjectDefinition('items.price', model_flag=FrameDataset.FLAG, objects='auto_from_batch_generation')
    with pytest.raises(ValueError, match='auto_from_batch_generation'):
        definition.generate_kpis(FrameDataset(name='ds'))
