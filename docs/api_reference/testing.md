# Synthetic test data

The examples in the API reference run on this platform, with the shared names of [StudyManager References](study_manager.md).

::: mesqual.testing

## Interpreters and inventory

The interpreters of `DummyDataset` and the inventory of its source. A study subclasses `DummyInterpreter` for its own flags, or an interpreter to override it.

::: mesqual.testing.dataset
    options:
        members: [DummyInterpreter, DummyModelInterpreter, DummyTimeSeriesInterpreter, DummyDerivedInterpreter, DummyAreaFlags, DummyBorderFlags]
        show_root_heading: false
        heading_level: 3

::: mesqual.testing.source.DummyInventory
    options:
        heading_level: 3
