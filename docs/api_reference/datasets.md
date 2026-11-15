# Dataset References

## Describe a dataset

The describe methods show what a dataset offers. They load no time series data.

Describe methods work on an instance. A method with `class_` in its name works on the class.

- `describe_class_flags()`: the flag specs that a dataset, interpreter or platform dataset class declares.
- `describe_flags()`: the flags that a dataset can deliver, with resolved values, parameter defaults and availability.
- `flag_info(flag)`: all details of one flag, as a `FlagInfo`.
- `describe_class_area_layers()`: the area layers registered on a dataset class.
- `describe_class_border_layers()`: the border layers registered on a dataset class.
- `describe_datasets(max_depth=None)`: the tree of a dataset and its children, down to `max_depth` levels.
- `describe_config()`: the config fields of a dataset, with their effective values.
- `StudyManager.describe_study()`: the scenarios and comparisons of a study, and the flags of each.

Use `describe_class_flags` to see what a platform offers before you build a dataset. Use `describe_flags` to see what you can fetch from a dataset you have.

Both give one row per spec, with the same index. The columns of `describe_class_flags` are the first columns of `describe_flags`, in the same order. `describe_flags` only adds columns.

A layer spec on area or border layers (`AreaFlagSpec`, `BorderFlagSpec`) has one row. `layers` names its layers. `object` names the model flags of these layers, in the same order, for example `Country.Model`.

| | `describe_class_flags()` | `describe_flags()` |
|---|---|---|
| Question | Which flags does this kind of dataset declare? | Which flags can this dataset deliver, and with which settings? |
| Called on | a dataset, interpreter or platform dataset class. A collection class raises `TypeError`. | an instance |
| Reads | spec declarations only | model flags and the source inventory; never time series |
| Flag families | template, for example `Node.Results.generation_{fuel}`, with the declared values or `resolved per dataset` | resolved from the dataset, for example one entry per fuel |
| Layers | the layers registered on the class | the registered layers of the dataset |
| Parameter defaults | spec default | effective default and its source: `spec`, `dataset`, `config`, `required` or `per dataset` |
| Availability | none | `available`, `unknown`, `partial` or `unavailable`, and the number of flags |
| Collections | not supported: a collection has no flags of its own | union of the children, plus which children have each flag |

All describe methods except `flag_info` and `describe_config` take `as_records=True` and then return plain dicts, ready for JSON.

::: mesqual.datasets
