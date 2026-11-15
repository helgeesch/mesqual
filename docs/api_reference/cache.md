# Cache References

A cache stores fetch results. Pass it to a dataset: `MyDataset(name='base', cache=PickleCache('cache'))`.

- The key (`CacheKey`) holds the dataset class, the dataset name, the flag and the resolved parameters. The version identity and the config are not in the key.
- A key holds one entry: the value and the record of its fetch, with the version identity, the config fields and files that the fetch read, and its nested fetches. A fetch uses the entry only if the version identity is the same, each recorded read gives the same value now and each nested fetch is up to date. Else the fetch computes the value and replaces the entry.
- In `_fetch`, call `FetchContext.record_file_read(path)` for each source file, so an edited file makes its entries stale.

::: mesqual.cache
