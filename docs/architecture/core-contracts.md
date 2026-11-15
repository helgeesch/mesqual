# Core contracts

Classes, signatures and examples are in the API reference: [Datasets](../api_reference/datasets.md), [StudyManager](../api_reference/study_manager.md).

## 1. Datasets

- A dataset subclasses `Dataset`, declares its flags with flag specs in the class attribute `flag_specs` ([flags-and-layers.md](flags-and-layers.md)) and implements `_fetch(flag, effective_config, **params)`. A platform dataset (`PlatformDataset`) has no specs of its own: it routes each flag to the registered interpreter that answers it ([other-contracts.md](other-contracts.md#interpreter-overrides)).
- An unknown keyword argument of a platform dataset raises `TypeError`. The rules are in the docstring of `PlatformDataset.__init__`.
- `accepted_flags`, `flag_is_accepted`, the flag index and the interpreter chains come from the specs and the registries only. Building them never opens source data. `accepted_flags` may narrow the spec flags using declarations of siblings or attributes, never source data.
- `flag_index` is a read-only property: from `flag_specs`; for a dataset with a parent (an interpreter), the index of its parent; for a collection, the union of its children; for a platform dataset, the specs of its answering interpreters and the layers of its class.
- `_fetch` reads config fields one by one from `effective_config`, for example `effective_config.convert_periods`. It does not read the whole config (`dataclasses.asdict`, `vars`, `copy`, `pickle`), `self.instance_config`, or config values in `__init__`. The fetch context records each field that `_fetch` reads, and the cache checks only these fields. A read of the whole config records all fields; a value read elsewhere is not recorded, so its change does not make a cache entry stale.
- `effective_config` is read-only in `_fetch`. To pass a changed config to a nested fetch, use `effective_config.merge({...})`.
- An interpreter passes `effective_config` to each fetch of an input: `self.parent_dataset.fetch(flag, effective_config, **params)`. So the config of the call, for example a fetch-time config, reaches the inputs. A layer aggregation follows the same rule: `aggregate` gets the `effective_config` of the layer flag fetch and passes it to each fetch ([flags-and-layers.md](flags-and-layers.md#5-layers)). The record holds the config of such a call as `INHERIT`. Passing `effective_config` on is no read of the whole config: each record holds only the fields that are read in its own fetch.
- `_fetch` records each source file that its result depends on with `FetchContext.record_file_read(path)`, so an edited file makes its cache entries stale.
- `_fetch` runs nested fetches in its own thread, or in a thread started with `contextvars.copy_context().run(...)`. Else the fetch context does not record them, and the cache does not check them.
- `name` identifies a dataset in a collection and in the cache key. Names in a collection are unique: the constructor raises `ValueError` on duplicates, and `add_dataset` with an existing name logs a warning and replaces. A platform dataset names each interpreter `f'{platform.name}.{Interpreter.__qualname__}'`; two interpreters with one `__qualname__` raise at instance creation.
- `attributes` hold primitive values only (`bool`, `int`, `float`, `str`). A platform dataset adds default attributes with `_default_attributes()`, from the source inventory or model flags, never from time series. Explicit attributes win. `description` is free text.
- State belongs to an instance. Class state holds only registrations, settings and the records of messages that log only once, never fetch results or caches.

## 2. Fetch

`fetch(flag, config=None, **params)` is the only path to `_fetch`; `fetch_merged` and `fetch_multiple_flags_and_concat` call it. The steps:

1. Raise `UnknownFlagError` for a flag without spec, and `ValueError` if the dataset does not accept the flag. A dataset that is not a collection raises `FlagUnavailableError` if the flag is not available. A collection fetches only from the children that accept the flag and on which it is not unavailable, and raises `FlagUnavailableError` only if there is none.
2. Resolve the effective config (section 4).
3. Resolve the flag and dataset parameters ([flags-and-layers.md](flags-and-layers.md#2-dimensions-variants-and-parameters)).
4. Open a frame in the fetch context. Nested fetches, config reads and file reads go into its `FetchRecord`. `get_fetch_record(flag, ...)` returns the record of the last such call; `flag_info(flag).dependencies` gives the dependencies of the last fetch without arguments.
5. If the dataset has a cache and `use_cache` is set, return the cached value when its entry is up to date ([other-contracts.md](other-contracts.md#cache)).
6. Else call `_fetch`, post-process (duplicate index entries, `auto_sort_index`) and, under the condition of step 5, store the result with its record.
7. In the outermost call only, apply the time index filter, then the granularity conversion.
8. Return a copy, also on the cache path.

## 3. Results

- A model flag returns a `DataFrame` with one row per object. Its index holds the object ids and is unique: duplicate ids raise `ValueError`.
- A time series flag returns a `DataFrame` with one time level in the index and one column per object; column labels match the index of the linked model flag. The time level is a `DatetimeIndex` or another time level that the schema declares: years, periods, or tuples such as `(run, period)`. Functions that need calendar time (granularity conversion, the date form of the time index filter) raise `TypeError` without a `DatetimeIndex`.
- Duplicate time index entries: `keep_duplicate_indices='first'` (default) or `'last'` keeps one row and logs a warning once per dataset class and flag; `'all'` keeps all rows.
- `fetch_multiple_flags_and_concat` checks that all results have the same type and the same levels and index types before the concat, else `ValueError`.

## 4. Config

- A config is a dataclass that subclasses `DatasetConfig`; a dataset names its class in `get_config_type()`. Each field has a description: `config_field(default, description='', merge_per_key=False, **kwargs)`. With `merge_per_key=True`, a dict field merges per key across layers. `describe_config()` shows each field with its effective value and the layer that set it.
- Resolution, later wins: field defaults, the class config of each class in the MRO (base first), the instance config, the fetch-time config. A layer overrides only its explicit fields: constructor arguments, fields set later, or the keys of a dict (`explicit_fields`). An explicit `None` overrides; `DatasetConfig.CLEAR` resets a field to its default.
- A collection passes its effective config to its children as their fetch-time config, so only its explicit fields override a child's own config. A key that the config type does not define passes through collections; a dataset that is not a collection logs a warning for it, once per config class and key.
- `set_instance_config` replaces the instance config; `merge_instance_config` merges on top of it. `DatasetConfigManager` holds the class configs (`clear_class_configs()` resets them).
- Base fields: `use_cache`, `auto_sort_index` (sorts the row index of any kind), `keep_duplicate_indices`, `time_index_filter`, `param_defaults`, `granularity`.

**Parameter defaults.** `param_defaults` maps a parameter name to a value for every flag and dataset class that declares that parameter, for example `{'solution': 'first', 'comparison_type': ComparisonTypeEnum.BOTH}`. Layers merge it per key. The outermost `fetch` checks the names: a name in the instance or fetch-time config must be a parameter of the dataset tree; a name in a class config may be a dataset parameter of any class. Other names raise `ValueError`. Nested fetches ignore names that they do not declare.

**Settings applied after the cache.** `time_index_filter` and `granularity` shape the result of the outermost fetch of a time series flag only, after the cache; model and `Other` flags pass unchanged. `_fetch` gets them as `None`, nested fetches return native values (so layer flags aggregate native values), and both settings share one cache entry with the native fetch. Set them on the outermost dataset or at fetch time.

- `TimeIndexFilter(level, include, exclude)` keeps the rows whose value on one level of the time index is in `include` and not in `exclude`; `TimeIndexFilter.dates(...)` filters calendar dates, `Range(start, end)` an inclusive range. Several filters combine with AND. A result without the level passes unchanged.
- `granularity` (for example `'15min'`, `'1h'`) converts time series with their quantity type through `TimeSeriesGranularityConverter`; `fetch(flag, granularity=...)` is a short form. It is not a parameter. A flag with an `UNKNOWN` or `AMBIGUOUS` quantity type and no spec value raises `ValueError`. The conversion rules are in the docstring of the [converter](../api_reference/energy_data_handling/granularity_modules.md).

## 5. Collections

| Class | Contract |
|---|---|
| `DatasetLinkCollection` | Each flag routes to exactly one child; two children with one flag raise `AmbiguousOverrideError` at construction. |
| `DatasetMergeCollection` | All children with the flag answer; results combine, and `precedence='first'` (default), `'last'` or a list of child names decides on overlap. |
| `DatasetConcatCollection` | Results concat with a new level `concat_level_name` (default `dataset`) on `concat_axis` (`1` columns, default, or `0` rows), at the top unless `concat_top=False`. The layout is fixed per collection. Join is outer. A child result without levels of the longest result gets them with the value `''`. |
| `DatasetSumCollection` | Numeric time series add with the `fill_value` of the spec; model flags merge. Other data raises. |
| `DatasetComparison` | Dataset parameters `comparison_type` (`DELTA`, `VARIATION`, `BOTH`), `replace_unchanged_values_by_nan` (not for time series flags: raises) and `fill_value` (default: the spec value). Non-numeric data gets text markers for changed, new and deleted entries. `BOTH` adds the last column level `compared`. |
| `DatasetConcatCollectionOfComparisons` | As concat; it passes the comparison parameters of the call or its `param_defaults` to its comparisons. |
| `StudyManager` | `scen`, `comp` and `scen_comp` concat on the columns with the levels `dataset` and `type`. `factory_from_scenarios` builds comparisons from `(variation, reference)` pairs; its optional `config` merges on top of the instance config of all three. |

- A collection has the union of the specs of its children. Two different specs for one flag name raise `ValueError` at construction and in `add_dataset`. `add_dataset` runs the checks of the constructor and leaves the collection unchanged if one fails.
- Each collection class implements `transform_schema`: a concat adds its level with role `dataset`, a comparison with default `BOTH` adds `compared`. `flag_info` on a collection gives the transformed schema of its first child that fetches the flag.
