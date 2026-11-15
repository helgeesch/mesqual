from mesqual.datasets import DatasetConcatCollection
from tests.datasets.fixtures import StaticDataset, hourly_frame
from tests.helpers import InMemoryCache


def test_constructor_arguments_take_effect():
    """All constructor arguments of `DatasetConcatCollection` take effect."""
    cache = InMemoryCache()
    collection = DatasetConcatCollection(
        [StaticDataset({'x': hourly_frame(periods=2)}, name='a')],
        name='c',
        attributes={'year': 2024},
        cache=cache,
        config={'use_cache': False},
    )
    assert collection.flag_index.get_spec('x').name == 'x'
    assert collection.attributes == {'year': 2024}
    assert collection.cache is cache
    assert collection.instance_config.use_cache is False
