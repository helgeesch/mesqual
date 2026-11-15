import pytest

from mesqual.datasets.dataset_config import DatasetConfig
from tests.helpers import FrameDataset, InMemoryCache, only_entry


def test_hash_works_with_dataclass_config():
    dataset = FrameDataset(name='ds', config=DatasetConfig(use_cache=False))
    assert hash(dataset) == hash(dataset)
    assert dataset in {dataset}


def test_hash_is_consistent_with_identity_equality():
    first = FrameDataset(name='ds')
    second = FrameDataset(name='ds')
    assert first != second
    assert len({first, second}) == 2


def test_fetch_from_cache_returns_a_copy():
    cache = InMemoryCache()
    dataset = FrameDataset(name='ds', cache=cache)
    dataset.fetch(FrameDataset.FLAG)

    cached = dataset.fetch(FrameDataset.FLAG)
    cached.loc['a', 'value'] = 99.0

    assert dataset.fetch(FrameDataset.FLAG).loc['a', 'value'] == 1.0


def test_fetch_without_cache_hit_returns_a_copy():
    cache = InMemoryCache()
    dataset = FrameDataset(name='ds', cache=cache)

    first = dataset.fetch(FrameDataset.FLAG)
    first.loc['a', 'value'] = 99.0

    _, value = only_entry(cache, dataset, FrameDataset.FLAG)
    assert value.loc['a', 'value'] == 1.0


@pytest.mark.parametrize('attributes', [{'year': [2030]}, {'scenario': None}, {1: 'a'}])
def test_constructor_rejects_invalid_attributes(attributes):
    with pytest.raises(TypeError):
        FrameDataset(name='ds', attributes=attributes)


def test_constructor_accepts_primitive_attributes():
    attributes = {'year': 2030, 'price': 50.0, 'base': True, 'label': 'x'}
    dataset = FrameDataset(name='ds', attributes=attributes)
    assert dataset.attributes == attributes
