"""Cache key, recorded dependencies and the dependency check on the fetch path."""

import collections
import copy
import dataclasses
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mesqual.cache import CacheKey, PickleCache
from mesqual.datasets import (
    Dataset,
    DatasetComparison,
    DatasetConcatCollection,
    DatasetConfig,
    FetchContext,
    FileRead,
    TimeIndexFilter,
    config_field,
)
from mesqual.datasets.dataset_comparison import DatasetConcatCollectionOfComparisons
from mesqual.datasets.dependency_check import DependencyCheck
from mesqual.enums import ComparisonTypeEnum
from mesqual.flag import Dimension
from tests.datasets.fixtures import PlatformConfig, hourly_frame, series_spec
from tests.helpers import InMemoryCache, only_entry

UTC = pd.Timestamp('2024-01-01 00:00', tz='UTC')


@dataclasses.dataclass(frozen=True)
class Holder:
    value: object


SOLUTION = Dimension('solution', 'Solutions.', {'final': 'final', 'first': 'first'}, default='final')


def spec_set(raw_version: int = 1) -> tuple:
    return (
        series_spec('raw', params=(SOLUTION,), version=raw_version),
        series_spec('derived'),
        series_spec('plain'),
        series_spec('whole'),
    )


class Source(Dataset):
    """`derived` fetches `raw`; `raw` reads `convert_periods`; `plain` reads no config; `whole` reads all fields.

    `calls` counts the `_fetch` calls per flag.
    """

    def __init__(self, raw_version: int = 1, **kwargs):
        self.flag_specs = spec_set(raw_version)
        super().__init__(**kwargs)
        self.calls: dict[str, int] = {}

    @classmethod
    def get_config_type(cls):
        return PlatformConfig

    def _fetch(self, flag, effective_config, **params):
        self.calls[flag] = self.calls.get(flag, 0) + 1
        if flag == 'derived':
            return self.fetch('raw') + 1
        if flag == 'raw':
            factor = 2 if effective_config.convert_periods else 1
            return hourly_frame(periods=2) * factor * (10 if params['solution'] == 'first' else 1)
        if flag == 'whole':
            return hourly_frame(periods=2) * len(dataclasses.asdict(effective_config))
        return hourly_frame(periods=2)


class Catching(Source):
    """`raw` raises; `derived` catches the error and returns a frame."""

    def _fetch(self, flag, effective_config, **params):
        if flag == 'raw':
            raise RuntimeError('source missing')
        try:
            self.fetch('raw')
        except RuntimeError:
            pass
        return hourly_frame(periods=2)


def source(cache, **kwargs) -> Source:
    return Source(name='scenario', cache=cache, **kwargs)


@pytest.fixture
def memory_cache():
    return InMemoryCache()


class TestKey:
    def test_key_has_class_name_flag_and_canonical_params(self, memory_cache):
        ds = source(memory_cache)
        ds.fetch('raw', solution='first')
        (key,) = memory_cache.list_keys(ds, 'raw')
        assert key == CacheKey(
            'tests.cache.test_cache.Source',
            'scenario',
            'raw',
            (('solution', "'first'"),),
        )

    def test_dict_order_does_not_change_the_canonical_params(self):
        record = Source(name='x', cache=None)._new_record('raw', {}, {}, {'p': {'a': 1, 'b': 2}})
        reordered = dataclasses.replace(record, params={'p': {'b': 2, 'a': 1}})
        assert CacheKey.of(record) == CacheKey.of(reordered)

    def test_config_is_not_in_the_key(self, memory_cache):
        ds = source(memory_cache)
        ds.fetch('plain')
        ds.fetch('plain', config={'convert_periods': False})
        assert len(memory_cache.list_keys(ds, 'plain')) == 1
        assert ds.calls['plain'] == 1

    def test_parameter_change_gives_a_new_key_and_the_same_parameters_hit(self, memory_cache):
        ds = source(memory_cache)
        ds.fetch('raw')
        ds.fetch('raw', solution='first')
        ds.fetch('raw', solution='first')
        assert len(memory_cache.list_keys(ds, 'raw')) == 2
        assert ds.calls['raw'] == 2


class TestParamDefaults:
    def test_unused_entry_keeps_the_key_and_the_entry(self, memory_cache):
        ds = source(memory_cache)
        ds.fetch('plain')
        ds.fetch('plain', config={'param_defaults': {'solution': 'first'}})
        assert ds.calls['plain'] == 1
        assert len(memory_cache.list_keys(ds, 'plain')) == 1

    def test_entry_equal_to_the_spec_default_reuses_the_entry(self, memory_cache):
        ds = source(memory_cache)
        ds.fetch('raw')
        ds.fetch('raw', config={'param_defaults': {'solution': 'final'}})
        assert ds.calls['raw'] == 1
        assert len(memory_cache.list_keys(ds, 'raw')) == 1

    def test_entry_with_another_value_gives_another_key(self, memory_cache):
        ds = source(memory_cache)
        plain = ds.fetch('raw')
        first = ds.fetch('raw', config={'param_defaults': {'solution': 'first'}})
        assert len(memory_cache.list_keys(ds, 'raw')) == 2
        assert (first == plain * 10).all().all()

    def test_entry_that_reaches_only_a_dependency_makes_the_entry_stale(self, memory_cache):
        source(memory_cache).fetch('derived')
        changed = source(memory_cache, config=PlatformConfig(param_defaults={'solution': 'first'}))
        result = changed.fetch('derived')
        assert changed.calls == {'derived': 1, 'raw': 1}
        assert len(memory_cache.list_keys(changed, 'derived')) == 1
        assert (result == hourly_frame(periods=2) * 20 + 1).all().all()


class TestVersion:
    @pytest.mark.parametrize('cache_type', ['memory', 'pickle'])
    def test_version_bump_of_a_dependency_overwrites_the_entry_and_the_next_fetch_hits(self, cache_type, tmp_path):
        cache = InMemoryCache() if cache_type == 'memory' else PickleCache(tmp_path)
        source(cache).fetch('derived')
        bumped = source(cache, raw_version=2)
        bumped.fetch('derived')
        assert bumped.calls == {'derived': 1, 'raw': 1}
        record, _ = only_entry(cache, bumped, 'derived')
        (dependency,) = record.dependencies
        assert dependency.version == bumped.version_identity('raw')
        again = source(cache, raw_version=2)
        again.fetch('derived')
        assert again.calls == {}

    def test_same_version_in_a_new_dataset_object_hits(self, memory_cache):
        source(memory_cache).fetch('derived')
        again = source(memory_cache)
        again.fetch('derived')
        assert again.calls == {}

    def test_base_and_subclass_with_one_name_do_not_share_entries(self, tmp_path):
        """The key holds the dataset class."""

        class Doubled(Source):
            def _fetch(self, flag, effective_config, **params):
                return super()._fetch(flag, effective_config, **params) * 2

        cache = PickleCache(tmp_path)
        base = source(cache).fetch('plain')
        doubled = Doubled(name='scenario', cache=cache).fetch('plain')
        assert (doubled == base * 2).all().all()
        assert len(cache.list_keys()) == 2


class TestConfigReads:
    def test_field_read_by_a_nested_flag_makes_only_its_dependents_stale(self, memory_cache):
        source(memory_cache).fetch('derived')
        source(memory_cache).fetch('plain')
        changed = source(memory_cache, config=PlatformConfig(convert_periods=False))
        changed.fetch('plain')
        result = changed.fetch('derived')
        assert changed.calls == {'derived': 1, 'raw': 1}
        assert (result == hourly_frame(periods=2) + 1).all().all()

    def test_field_that_the_flag_does_not_read_keeps_the_entry(self, memory_cache):
        ds = source(memory_cache)
        ds.fetch('derived')
        ds.fetch('derived', config={'auto_sort_index': True, 'keep_duplicate_indices': 'first'})
        assert ds.calls == {'derived': 1, 'raw': 1}

    def test_reads_of_the_post_processing_are_recorded(self, memory_cache):
        ds = source(memory_cache)
        ds.fetch('plain')
        ds.fetch('plain', config={'auto_sort_index': False})
        assert ds.calls['plain'] == 2

    def test_whole_config_read_records_all_fields_but_not_the_settings_after_the_cache(self, memory_cache):
        ds = source(memory_cache)
        ds.fetch('whole')
        ds.fetch('whole', config={'time_index_filter': TimeIndexFilter.dates(include=['2024-01-01'])})
        ds.fetch('whole', granularity='2h')
        assert ds.calls['whole'] == 1
        ds.fetch('whole', config={'convert_periods': False})
        assert ds.calls['whole'] == 2

    def test_config_change_overwrites_the_entry(self, memory_cache):
        ds = source(memory_cache)
        ds.fetch('raw')
        ds.fetch('raw', config={'convert_periods': False})
        record, _ = only_entry(memory_cache, ds, 'raw')
        assert record.config_reads['convert_periods'] is False
        ds.fetch('raw', config={'convert_periods': False})
        assert ds.calls['raw'] == 2

    def test_inherited_config_read_by_a_child_makes_the_collection_entry_stale(self, memory_cache):
        children = [Source(name='a'), Source(name='b')]
        collection = DatasetConcatCollection(children, name='both', cache=memory_cache)
        collection.fetch('raw')
        collection.fetch('raw', config={'convert_periods': False})
        assert [child.calls['raw'] for child in children] == [2, 2]


class TestGivenParams:
    def test_collection_that_passes_a_param_default_to_a_child_checks_it(self, memory_cache):
        inner = DatasetComparison(
            Source(name='a'),
            Source(name='b'),
            name='inner',
            config=DatasetConfig(param_defaults={'comparison_type': ComparisonTypeEnum.VARIATION}),
        )
        outer = DatasetConcatCollectionOfComparisons([inner], name='outer', cache=memory_cache)
        given = outer.fetch('plain', config={'param_defaults': {'comparison_type': ComparisonTypeEnum.DELTA}})
        own = outer.fetch('plain')
        assert (given == 0).all().all()
        assert (own == hourly_frame(periods=2).to_numpy()).all().all()
        assert outer.get_fetch_record('plain').reads_given_params


class TestConcatLayoutIdentity:
    """`concat_axis`, `concat_top` and `concat_level_name` change the result, so they are in the version identity."""

    @staticmethod
    def collection(cache, **layout) -> DatasetConcatCollection:
        return DatasetConcatCollection([Source(name='a'), Source(name='b')], name='c', cache=cache, **layout)

    @pytest.mark.parametrize('layout', [{'concat_axis': 0}, {'concat_top': False}, {'concat_level_name': 'scenario'}])
    def test_same_name_with_another_layout_does_not_reuse_the_entry(self, memory_cache, layout):
        default = self.collection(memory_cache).fetch('plain')
        other = self.collection(memory_cache, **layout)
        result = other.fetch('plain')
        assert [child.calls['plain'] for child in other.datasets] == [1, 1]
        assert names_of(result) != names_of(default)
        again = self.collection(memory_cache, **layout)
        pd.testing.assert_frame_equal(again.fetch('plain'), result)
        assert [child.calls for child in again.datasets] == [{}, {}]


def names_of(data: pd.DataFrame) -> tuple:
    return list(data.index.names), list(data.columns.names)


def file_dataset(paths: dict, cache, nested: dict | None = None):
    """Each flag reads its file; a flag in `nested` also fetches the flag it maps to."""

    class Files(Dataset):
        flag_specs = tuple(series_spec(flag) for flag in paths)
        calls = []

        def _fetch(self, flag, effective_config, **params):
            self.calls.append(flag)
            FetchContext.record_file_read(paths[flag])
            value = hourly_frame(periods=2) * float(Path(paths[flag]).read_text())
            if nested and flag in nested:
                value = value + self.fetch(nested[flag])
            return value

    return Files(name='files', cache=cache)


class TestFileReads:
    def test_edited_file_makes_stale_exactly_the_entries_that_read_it(self, tmp_path, memory_cache):
        paths = {'x': tmp_path / 'x.txt', 'y': tmp_path / 'y.txt'}
        for path in paths.values():
            path.write_text('1')
        ds = file_dataset(paths, memory_cache)
        ds.fetch('x'), ds.fetch('y')
        paths['x'].write_text('22')
        assert ds.fetch('x')['a'].iloc[1] == 22.0
        ds.fetch('y')
        assert ds.calls == ['x', 'y', 'x']

    def test_touched_file_with_the_same_content_keeps_the_entry(self, tmp_path, memory_cache):
        path = tmp_path / 'x.txt'
        path.write_text('1')
        ds = file_dataset({'x': path}, memory_cache)
        ds.fetch('x')
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        ds.fetch('x')
        assert ds.calls == ['x']

    def test_record_holds_path_size_time_and_hash(self, tmp_path, memory_cache):
        path = tmp_path / 'x.txt'
        path.write_text('1')
        ds = file_dataset({'x': path}, memory_cache)
        ds.fetch('x')
        (file_read,) = ds.get_fetch_record('x').file_reads
        assert file_read == FileRead.of(path)
        assert (file_read.path, file_read.size) == (str(path.absolute()), 1) and file_read.sha256

    def test_large_file_has_no_hash_and_a_new_time_makes_it_stale(self, tmp_path, monkeypatch):
        monkeypatch.setattr(FileRead, 'HASH_MAX_BYTES', 0)
        path = tmp_path / 'x.txt'
        path.write_text('1')
        file_read = FileRead.of(path)
        assert file_read.sha256 is None and file_read.is_unchanged()
        os.utime(path, ns=(0, file_read.mtime_ns + 10**9))
        assert not file_read.is_unchanged()

    def test_key_of_a_small_file_does_not_depend_on_the_folder(self, tmp_path):
        paths = [tmp_path / folder / 'x.txt' for folder in ('a', 'b', 'c')]
        for path, content in zip(paths, ('1', '1', '2')):
            path.parent.mkdir()
            path.write_text(content)
        keys = [FileRead.of(path).key for path in paths]
        assert keys[0] == keys[1] == FileRead.of(paths[0]).sha256
        assert keys[2] != keys[0]

    def test_key_of_a_large_file_is_name_size_and_time(self, tmp_path, monkeypatch):
        monkeypatch.setattr(FileRead, 'HASH_MAX_BYTES', 0)
        path = tmp_path / 'x.txt'
        path.write_text('1')
        file_read = FileRead.of(path)
        assert file_read.key == ('x.txt', 1, file_read.mtime_ns)

    def test_deleted_file_is_changed(self, tmp_path):
        path = tmp_path / 'x.txt'
        path.write_text('1')
        file_read = FileRead.of(path)
        path.unlink()
        assert not file_read.is_unchanged()

    def test_outside_of_a_fetch_or_while_not_recording_nothing_is_recorded(self, tmp_path):
        path = tmp_path / 'x.txt'
        path.write_text('1')
        FetchContext.record_file_read(path)
        with FetchContext.frame(Source(name='x')._new_record('plain', {}, {}, {})) as frame:
            with FetchContext.internal():
                FetchContext.record_file_read(path)
        assert frame.file_reads == {}

    def test_same_size_and_restored_modification_time_with_new_content_is_changed(self, tmp_path):
        path = tmp_path / 'x.txt'
        path.write_text('1')
        file_read = FileRead.of(path)
        path.write_text('2')
        os.utime(path, ns=(0, file_read.mtime_ns))
        assert not file_read.is_unchanged()

    @pytest.mark.parametrize('on_disk', [False, True])
    def test_dot_dot_after_a_linked_folder_follows_the_link(self, tmp_path, memory_cache, on_disk):
        """`x.txt` beside the link has the old content, so a check of that file instead would hit."""
        cache = PickleCache(tmp_path / 'cache') if on_disk else memory_cache
        (tmp_path / 'real' / 'sub').mkdir(parents=True)
        (tmp_path / 'real' / 'x.txt').write_text('5')
        (tmp_path / 'x.txt').write_text('5')
        (tmp_path / 'link').symlink_to(tmp_path / 'real' / 'sub')
        ds = file_dataset({'x': tmp_path / 'link' / '..' / 'x.txt'}, cache)
        ds.fetch('x')
        (tmp_path / 'real' / 'x.txt').write_text('9')
        pd.testing.assert_frame_equal(ds.fetch('x'), hourly_frame(periods=2) * 9.0)

    def test_symlink_that_points_to_another_file_is_changed(self, tmp_path, memory_cache):
        (tmp_path / 'jan.txt').write_text('1')
        (tmp_path / 'feb.txt').write_text('2')
        link = tmp_path / 'current.txt'
        link.symlink_to(tmp_path / 'jan.txt')
        ds = file_dataset({'x': link}, memory_cache)
        ds.fetch('x')
        link.unlink()
        link.symlink_to(tmp_path / 'feb.txt')
        pd.testing.assert_frame_equal(ds.fetch('x'), hourly_frame(periods=2) * 2.0)

    def test_directory_raises(self, tmp_path, memory_cache):
        ds = file_dataset({'x': tmp_path}, memory_cache)
        with pytest.raises(ValueError, match='is a directory'):
            ds.fetch('x')


class TestFileHashes:
    @staticmethod
    def count_hashes(monkeypatch) -> list:
        hashed = []
        original = FileRead._sha256

        def counting(path):
            hashed.append(path)
            return original(path)

        monkeypatch.setattr(FileRead, '_sha256', staticmethod(counting))
        return hashed

    def test_fetch_without_cache_does_not_hash(self, tmp_path, monkeypatch):
        hashed = self.count_hashes(monkeypatch)
        path = tmp_path / 'x.txt'
        path.write_text('1')
        ds = file_dataset({'x': path}, None)
        ds.fetch('x')
        assert hashed == []
        assert ds.get_fetch_record('x').file_reads[0].sha256 is None

    def test_storing_parent_hashes_the_file_read_of_a_non_storing_child(self, tmp_path, memory_cache):
        path = tmp_path / 'x.txt'
        path.write_text('1')
        collection = DatasetConcatCollection([file_dataset({'x': path}, None)], name='c', cache=memory_cache)
        collection.fetch('x')
        assert collection.get_fetch_record('x').dependencies[0].file_reads[0].sha256 is not None

    def test_check_hashes_each_file_once(self, tmp_path, monkeypatch, memory_cache):
        path = tmp_path / 'x.txt'
        path.write_text('1')

        class Both(Dataset):
            flag_specs = (series_spec('a'), series_spec('b'), series_spec('top'))

            def _fetch(self, flag, effective_config, **params):
                if flag == 'top':
                    return self.fetch('a') + self.fetch('b')
                FetchContext.record_file_read(path)
                return hourly_frame(periods=2)

        ds = Both(name='both', cache=memory_cache)
        ds.fetch('top')
        hashed = self.count_hashes(monkeypatch)
        ds.fetch('top')
        assert len(hashed) == 1


class TestCollectionChildren:
    def test_added_child_makes_the_collection_entry_stale(self, memory_cache):
        collection = DatasetConcatCollection([Source(name='a')], name='c', cache=memory_cache)
        collection.fetch('plain')
        collection.add_dataset(Source(name='b'))
        result = collection.fetch('plain')
        assert list(result.columns.get_level_values(0).unique()) == ['a', 'b']
        record, _ = only_entry(memory_cache, collection, 'plain')
        assert record.version == collection.version_identity('plain')

    def test_child_replaced_by_the_same_class_and_name_keeps_the_entry(self, memory_cache):
        collection = DatasetConcatCollection([Source(name='a')], name='c', cache=memory_cache)
        collection.fetch('plain')
        collection.add_dataset(Source(name='a'))
        collection.fetch('plain')
        assert collection.datasets[0].calls == {}
        assert len(memory_cache.list_keys(collection, 'plain')) == 1

    def test_availability_change_of_a_child_makes_the_collection_entry_stale(self, memory_cache):
        class Switchable(Source):
            available = False

            def _is_available(self, flag):
                return self.available

        a, b = Source(name='a'), Switchable(name='b')
        collection = DatasetConcatCollection([a, b], name='c', cache=memory_cache)
        collection.fetch('plain')
        b.available = True
        result = collection.fetch('plain')
        assert list(result.columns.get_level_values(0).unique()) == ['a', 'b']
        record, _ = only_entry(memory_cache, collection, 'plain')
        assert record.version == collection.version_identity('plain')


class TestMissingValues:
    def test_comparison_fill_value_none_and_nan_do_not_share_an_entry(self, memory_cache):
        class Points(Dataset):
            flag_specs = (series_spec('raw', fill_value=0.0),)

            def __init__(self, index, **kwargs):
                self.index = index
                super().__init__(**kwargs)

            def _fetch(self, flag, effective_config, **params):
                return pd.DataFrame({'a': 1.0}, index=pd.DatetimeIndex(self.index, name='datetime'))

        def build(fill, cache):
            variation = Points(['2024-01-01 00:00', '2024-01-01 01:00'], name='v')
            reference = Points(['2024-01-01 00:00'], name='r')
            comparison = DatasetComparison(
                variation,
                reference,
                config=DatasetConfig(param_defaults={'fill_value': fill}),
            )
            return DatasetConcatCollection([comparison], name='c', cache=cache)

        build(float('nan'), memory_cache).fetch('raw')
        cached = build(None, memory_cache).fetch('raw')
        fresh = build(None, None).fetch('raw')
        pd.testing.assert_frame_equal(cached, fresh)


class TestStoredEntries:
    def test_entry_with_a_raised_dependency_is_not_stored(self, memory_cache):
        ds = Catching(name='x', cache=memory_cache)
        ds.fetch('derived')
        ds.fetch('derived')
        assert memory_cache.list_keys(ds, 'derived') == []

    def test_reason_for_not_storing_is_logged_once(self, memory_cache, caplog):
        caplog.set_level(logging.INFO, logger='mesqual')
        ds = Catching(name='x', cache=memory_cache)
        ds.fetch('derived')
        ds.fetch('derived')
        messages = [r.getMessage() for r in caplog.records if 'is not cached' in r.getMessage()]
        assert messages == ['x derived is not cached: x raw: the fetch raised.']

    def test_stale_entry_logs_its_reason_at_debug(self, memory_cache, caplog):
        caplog.set_level(logging.DEBUG, logger='mesqual')
        source(memory_cache).fetch('raw')
        source(memory_cache, raw_version=2).fetch('raw')
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG and 'stale' in r.getMessage()]
        assert messages == ['Flag raw: cache entry is stale: the version identity changed']

    def test_cache_whose_entry_raises_counts_as_a_miss(self, caplog):
        class Unreadable(InMemoryCache):
            def entry(self, key):
                raise OSError('disk gone')

        ds = source(Unreadable())
        ds.fetch('plain')
        result = ds.fetch('plain')
        assert ds.calls['plain'] == 2
        assert len(result) == 2
        assert 'cannot read the entry (OSError: disk gone)' in caplog.text

    def test_cache_hit_records_the_stored_dependencies(self, memory_cache):
        source(memory_cache).fetch('derived')
        again = source(memory_cache)
        again.fetch('derived')
        record = again.get_fetch_record('derived')
        assert [d.flag for d in record.dependencies] == ['raw']
        assert record.dependencies[0].config_reads == {
            'convert_periods': True,
            'keep_duplicate_indices': 'first',
            'auto_sort_index': True,
        }


class TestPickleCache:
    @staticmethod
    def moved_files_dataset(tmp_path):
        """Fetch `y`, which reads `y.txt` and fetches `x`, then copy the data and the cache folder together."""
        data = tmp_path / 'before' / 'data'
        data.mkdir(parents=True)
        for name in ('x', 'y'):
            (data / f'{name}.txt').write_text('1')
        nested = {'y': 'x'}
        file_dataset(
            {'x': data / 'x.txt', 'y': data / 'y.txt'}, PickleCache(tmp_path / 'before' / 'cache'), nested
        ).fetch('y')
        shutil.copytree(tmp_path / 'before', tmp_path / 'after')
        moved = tmp_path / 'after' / 'data'
        cache = PickleCache(tmp_path / 'after' / 'cache')
        return file_dataset({'x': moved / 'x.txt', 'y': moved / 'y.txt'}, cache, nested), moved

    def test_data_and_cache_moved_together_keep_the_entries(self, tmp_path):
        ds, moved = self.moved_files_dataset(tmp_path)
        shutil.rmtree(tmp_path / 'before')
        ds.fetch('y')
        assert ds.calls == []
        record = ds.get_fetch_record('y')
        assert {read.path for each in [record, *record.iter_dependencies()] for read in each.file_reads} == {
            str(moved / 'x.txt'),
            str(moved / 'y.txt'),
        }

    def test_file_changed_after_the_move_makes_the_entries_stale(self, tmp_path):
        ds, moved = self.moved_files_dataset(tmp_path)
        (moved / 'x.txt').write_text('22')
        ds.fetch('y')
        assert ds.calls == ['y', 'x']

    def test_flag_with_a_slash_works(self, tmp_path):
        """The file name is the digest of the key."""
        cache = PickleCache(tmp_path)
        record = dataclasses.replace(source(cache)._new_record('plain', {}, {}, {}), flag='a/b')
        key = CacheKey.of(record)
        cache.set(key, hourly_frame(periods=2), record)
        assert cache.list_keys(flag='a/b') == [key]

    def test_dataset_a_does_not_match_keys_of_dataset_a_b(self, tmp_path):
        """`list_keys` and `delete` match the dataset name exactly."""
        cache = PickleCache(tmp_path)
        a, a_b = Source(name='a', cache=cache), Source(name='a_b', cache=cache)
        a.fetch('plain'), a_b.fetch('plain')
        assert [key.dataset_name for key in cache.list_keys(a)] == ['a']
        cache.delete(a, 'plain')
        assert [key.dataset_name for key in cache.list_keys()] == ['a_b']

    def test_foreign_pickle_files_in_the_folder_are_ignored(self, tmp_path):
        pd.DataFrame({'a': [1]}).to_pickle(tmp_path / 'scenario_plain_config_123.pickle')
        cache = PickleCache(tmp_path)
        ds = source(cache)
        ds.fetch('plain')
        assert [key.flag for key in cache.list_keys()] == ['plain']
        cache.delete(ds)
        assert [path.name for path in tmp_path.iterdir()] == ['scenario_plain_config_123.pickle']

    def test_entries_survive_a_new_cache_object(self, tmp_path):
        source(PickleCache(tmp_path)).fetch('derived')
        again = source(PickleCache(tmp_path))
        again.fetch('derived')
        assert again.calls == {}

    def test_corrupt_file_counts_as_no_entry_and_the_next_set_replaces_it(self, tmp_path, caplog):
        cache = PickleCache(tmp_path)
        source(cache).fetch('plain')
        (path,) = tmp_path.glob(f'*{PickleCache.SUFFIX}')
        path.write_bytes(b'not a pickle')
        assert cache.list_keys() == []
        ds = source(cache)
        assert len(ds.fetch('plain')) == 2
        assert ds.calls == {'plain': 1}
        assert [key.flag for key in cache.list_keys()] == ['plain']
        assert 'cannot be read' in caplog.text

    def test_set_replaces_the_entry_in_the_one_file_of_the_key(self, tmp_path):
        cache = PickleCache(tmp_path)
        source(cache).fetch('plain')
        (key,) = cache.list_keys()
        stored, _ = cache.entry(key)
        cache.set(key, hourly_frame(periods=3), stored)
        _, value = cache.entry(key)
        assert len(value) == 3
        assert len(list(tmp_path.glob(f'*{PickleCache.SUFFIX}'))) == 1


class TestUncheckableValues:
    @staticmethod
    def dataset_reading(value, cache):
        @dataclasses.dataclass
        class ValueConfig(PlatformConfig):
            value: object = config_field(None, 'A value.')

        class Reader(Dataset):
            flag_specs = (series_spec('raw'),)
            calls = 0

            @classmethod
            def get_config_type(cls):
                return ValueConfig

            def _fetch(self, flag, effective_config, **params):
                type(self).calls += 1
                _ = effective_config.value
                return hourly_frame(periods=2)

        return Reader(name='x', cache=cache, config=ValueConfig(value=value))

    def test_value_that_cannot_be_pickled_is_returned_and_not_stored(self, tmp_path, caplog):
        cache = PickleCache(tmp_path)
        ds = self.dataset_reading(lambda v: v, cache)
        assert len(ds.fetch('raw')) == 2
        assert cache.list_keys() == []
        assert list(tmp_path.iterdir()) == []

    def test_value_that_cannot_be_pickled_is_logged_once(self, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger='mesqual')
        ds = self.dataset_reading(lambda v: v, PickleCache(tmp_path))
        for _ in range(3):
            ds.fetch('raw')
        messages = [r.getMessage() for r in caplog.records if 'is not cached' in r.getMessage()]
        assert len(messages) == 1 and 'the cache cannot store it' in messages[0]

    def test_value_that_cannot_be_deep_copied_is_not_stored(self, memory_cache):
        class NoCopy:
            def __deepcopy__(self, memo):
                raise TypeError('no copy')

        ds = self.dataset_reading(NoCopy(), memory_cache)
        ds.fetch('raw')
        ds.fetch('raw')
        assert type(ds).calls == 2
        assert memory_cache.list_keys() == []

    @pytest.mark.parametrize('value', [float('nan'), np.float32('nan'), (1.0, float('nan')), pd.NaT])
    def test_missing_values_count_as_the_same_value_and_hit(self, value, memory_cache):
        ds = self.dataset_reading(value, memory_cache)
        ds.fetch('raw')
        ds.fetch('raw')
        assert type(ds).calls == 1

    def test_value_that_is_not_equal_to_itself_is_not_stored(self, memory_cache):
        class NeverEqual:
            def __eq__(self, other):
                return False

            def __deepcopy__(self, memo):
                return NeverEqual()

        ds = self.dataset_reading(NeverEqual(), memory_cache)
        ds.fetch('raw')
        assert memory_cache.list_keys() == []
        record = ds.get_fetch_record('raw')
        check = DependencyCheck(ds, record, ds._cache_config(ds._prepare_config(None)))
        assert check.stale_reason(record) == 'config field value cannot be compared'

    def test_dataclass_with_another_time_zone_is_changed(self, memory_cache):
        @dataclasses.dataclass(frozen=True)
        class Window:
            start: pd.Timestamp

        utc = pd.Timestamp('2024-01-01 00:00', tz='UTC')
        self.dataset_reading(Window(utc), memory_cache).fetch('raw')
        ds = self.dataset_reading(Window(utc.tz_convert('Europe/Berlin')), memory_cache)
        ds.fetch('raw')
        assert type(ds).calls == 1
        assert len(memory_cache.list_keys()) == 1


class TestDataFrameReads:
    """A config read that holds a DataFrame: the cache hit needs no equality match of records."""

    @pytest.mark.parametrize('kind', ['memory', 'pickle'])
    def test_cache_hit_with_a_dataframe_in_a_config_read(self, tmp_path, kind):
        @dataclasses.dataclass
        class TableConfig(PlatformConfig):
            table: object = config_field(None, 'A table.')

        class Tables(Dataset):
            flag_specs = (series_spec('raw'),)
            calls = 0

            @classmethod
            def get_config_type(cls):
                return TableConfig

            def _fetch(self, flag, effective_config, **params):
                type(self).calls += 1
                _ = effective_config.table
                return hourly_frame(periods=2)

        cache = InMemoryCache() if kind == 'memory' else PickleCache(tmp_path)
        config = TableConfig(table=pd.DataFrame({'a': [1]}))
        Tables(name='x', cache=cache, config=config).fetch('raw')
        Tables(name='x', cache=cache, config=config).fetch('raw')
        assert Tables.calls == 1


class TestDependencyCheck:
    @staticmethod
    def version_stale_reason(current_version, stored_version) -> str | None:
        ds = source(None)
        record = ds._new_record('plain', {}, {}, {})
        check = DependencyCheck(
            ds, dataclasses.replace(record, version=current_version), ds._cache_config(ds._prepare_config(None))
        )
        return check.stale_reason(dataclasses.replace(record, version=stored_version))

    def test_aggregation_field_of_another_type_makes_the_version_identity_stale(self):
        reason = self.version_stale_reason((('Mean', {'weight': 1}),), (('Mean', {'weight': 1.0}),))
        assert reason == 'the version identity changed'

    def test_version_field_with_the_same_repr_and_another_value_is_stale(self):
        long, changed = pd.Series(range(1000)), pd.Series(range(1000))
        changed[500] = -1
        assert repr(long) == repr(changed)
        reason = self.version_stale_reason((('Mean', {'weights': long}),), (('Mean', {'weights': changed}),))
        assert reason == 'the version identity changed'

    def test_same_version_identity_is_up_to_date(self):
        assert self.version_stale_reason((('Mean', {'weight': 1}),), (('Mean', {'weight': 1}),)) is None

    @pytest.mark.parametrize(
        'current, stored',
        [
            (None, float('nan')),
            (float('nan'), pd.NA),
            (pd.NaT, float('nan')),
            (np.float32('nan'), float('nan')),
            (1, 1.0),
            (1, True),
            (1.0, True),
            (pd.Timestamp('2024-01-01 00:00', tz='UTC'), pd.Timestamp('2024-01-01 01:00', tz='Europe/Berlin')),
        ],
    )
    def test_values_of_another_type_or_time_zone_differ(self, current, stored):
        assert not DependencyCheck._same(current, stored)
        assert not DependencyCheck._same((current,), (stored,))

    @pytest.mark.parametrize(
        'current, stored',
        [
            (pd.Series([UTC]), pd.Series([UTC.tz_convert('Europe/Berlin')])),
            (pd.DataFrame({'a': [1]}, index=[UTC]), pd.DataFrame({'a': [1]}, index=[UTC.tz_convert('Europe/Berlin')])),
            (pd.Index([1, 2]), pd.Index([1.0, 2.0])),
            (pd.Index([1]), pd.Index([1.0])),
            (pd.DataFrame({'a': [1, 2]}, index=[1, 2]), pd.DataFrame({'a': [1, 2]}, index=[1.0, 2.0])),
            (pd.Series([1], name='a'), pd.Series([1], name='b')),
            (np.array([1]), np.array([1.0])),
            ({1}, {1.0}),
            ({True}, {1}),
            (frozenset([UTC]), frozenset([UTC.tz_convert('Europe/Berlin')])),
            ({1: 'a'}, {True: 'a'}),
            (Holder(1), Holder(1.0)),
            (Holder(UTC), Holder(UTC.tz_convert('Europe/Berlin'))),
            ([np.array([1])], [np.array([1.0])]),
            ({'a': 1}, collections.OrderedDict(a=1)),
        ],
    )
    def test_nested_values_of_another_type_differ(self, current, stored):
        assert not DependencyCheck._same(current, stored)

    @pytest.mark.parametrize(
        'value',
        [
            None,
            float('nan'),
            np.float32('nan'),
            pd.NA,
            pd.NaT,
            pd.Timestamp('2024-01-01', tz='UTC'),
            (float('nan'),),
            Holder(float('nan')),
            pd.Series([np.nan]),
            np.array([1.0, np.nan]),
            np.array([]),
            pd.DataFrame({'a': [1.0, np.nan]}, index=pd.date_range('2024', periods=2, tz='UTC')),
        ],
    )
    def test_missing_and_tz_values_of_the_same_type_are_the_same(self, value):
        assert DependencyCheck._same(value, copy.deepcopy(value))

    def test_values_compare_per_item_and_with_equals(self):
        frame = hourly_frame(periods=2)
        assert DependencyCheck._same({'a': frame, 'b': 1}, {'a': frame.copy(), 'b': 1})
        assert not DependencyCheck._same({'a': frame}, {'a': frame * 2})
        assert not DependencyCheck._same({'a': 1}, {'a': 1, 'b': 2})
        assert not DependencyCheck._same(frame.to_numpy(), frame.to_numpy() * 2)


class TestKeyAcrossSessions:
    """A `PickleCache` entry written in one Python process hits in the next one, also with another hash seed."""

    SESSION = '''
import sys
from mesqual.cache import PickleCache
from mesqual.testing import DummyDataset, DummyScenario


class StoreRecordingCache(PickleCache):
    """Records the flag of each `set`. A fetch catches errors of `set`, so a raise here would not end the session."""

    stored = []

    def set(self, key, value, record):
        self.stored.append(key.flag)
        super().set(key, value, record)


ds = DummyDataset(DummyScenario(), cache=StoreRecordingCache(sys.argv[1]))
print(ds.fetch('Country.Results.price', solution='first').sum().sum())
if sys.argv[2] == 'write' and not StoreRecordingCache.stored:
    sys.exit('nothing stored')
if sys.argv[2] == 'read' and StoreRecordingCache.stored:
    sys.exit(f'cache miss: {StoreRecordingCache.stored}')
'''

    def run_session(self, folder: Path, mode: str, hash_seed: str) -> str:
        env = {**os.environ, 'PYTHONHASHSEED': hash_seed}
        result = subprocess.run(
            [sys.executable, '-c', self.SESSION, str(folder), mode],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    def test_entry_written_in_one_process_hits_in_another_process_with_another_hash_seed(self, tmp_path):
        written = self.run_session(tmp_path, 'write', '1')
        files = sorted(tmp_path.glob(f'*{PickleCache.SUFFIX}'))
        assert self.run_session(tmp_path, 'read', '2') == written
        assert sorted(tmp_path.glob(f'*{PickleCache.SUFFIX}')) == files
