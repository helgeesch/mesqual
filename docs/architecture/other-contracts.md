# Other contracts

## Interpreter overrides

A platform package or a study replaces a whole interpreter on a dataset subclass and reuses the original code; the layer flags on the specs of the override pick up the replacement. The registry rules, arguments, errors and an example are in the docstrings of `PlatformDataset` and `PlatformDataset.register_interpreter` ([Datasets](../api_reference/datasets.md)).

- **One answer per original.** On one class, an override beats the original. A spec that the override leaves out leaves the dataset, with the layer specs on it.
- **One scenario.** An override that changes only the logic can sit on the class of one scenario. An override that changes a spec applies to all scenarios: a collection of both raises `ValueError`. Use the override class for every scenario, and set the behavior per scenario with a parameter default or a dataset attribute.
- **After the chain.** `PlatformDataset._fetch` is the hook to process each result after its interpreter chain: a subclass overrides it and changes the result of `super()._fetch(...)`. Nested fetches through `parent_dataset.fetch` get the processed result too, and a flag derived from processed inputs passes the hook again. So the processing selects its flags, or gives the same result when it runs twice. It is not part of the version identity: to change results with a version, register an override.

## Describe and availability

The describe methods and their differences are in the [Datasets](../api_reference/datasets.md) reference. They read the specs, the source inventory, and model flags where a flag family, a layer or a default attribute needs them. Facts that need time series are flags, not descriptions.

**Availability.** A flag is available on a dataset when the dataset can answer it. An interpreter implements `_is_available(flag)` with cheap reads of the source inventory: `True`, `False`, `None` for unknown (the default), or an `Availability(available, reason)`.

- On a platform dataset, the answering interpreter decides; `None` stays unknown until the first successful fetch. On any other dataset that is not a collection, `None` counts as available.
- A collection is available if one child is, else unknown if one child is unknown. A comparison needs both children.
- `availability(flag)` gives the `Availability` with its reason; `is_flag_available(flag)` its `available` field. `Availability.of_inputs` combines the inputs of a derived flag.
- `describe_flags(available_only=True)` keeps available and unknown flags; `flag_info` gives the reason.

## Cache

`Dataset.fetch` caches results in the `Cache` of the dataset (for example `PickleCache`) when `use_cache` is set. The cache stores entries; `fetch` checks them. A custom cache implements `entry`, `set`, `delete` and `list_keys` ([Cache](../api_reference/cache.md)).

- **Key.** The key ([`CacheKey`](../api_reference/cache.md)) holds the dataset class and name, not the source: two datasets of one class and name with different sources share their entries.
- **Version identity.** One `(class name, version)` pair per link of the interpreter chain: the original interpreter with the version of its spec, then each override with its `register_interpreter(version=)`; for a layer flag also the aggregation with its fields and `version`; for a collection its children (class and name) and, for a concat, its layout. Bump a version by hand when the logic changes.
- **Check.** An entry is up to date if its call did not raise, the version identity and the resolved parameters are the same, each recorded config field has the same value (same type and value: `1`, `1.0` and `True` differ), each recorded file has the same size and content hash (the modification time for a file over 16 MiB), and each dependency is up to date, with its call rebuilt from its record. A stale entry is a miss, and the new result replaces it. `fetch` stores a result only if its record can be up to date.
- **Limits.** A logic change without a version bump keeps the old entries. A file read without `record_file_read` is not checked. Unused keys stay until `delete`. Two processes that store one key at the same time: the last `set` wins.

## Validation

A check of `mesqual.validation` does not return `ERROR`: it raises, and `DatasetValidator` reports the exception as `ERROR` ([Validation](../api_reference/validation.md)).

## KPIs

A KPI is one value per object and dataset, with the attributes of both ([KPIs](../api_reference/kpis/index.md)).

- `KPIDefinition` and `CustomKPIDefinition` are the base classes for own KPIs. A platform package or a study subclasses them with its own name, unit and computation per dataset.
- The unit of a `FlagAggKPIDefinition` is the unit of its aggregation if it has one, else the unit of the flag.
- Two `Aggregation`s with the same name are equal.
- `Dataset.add_kpis` skips a KPI whose name is already in the collection and logs a warning; `add_kpis_from_definitions` adds through it.
