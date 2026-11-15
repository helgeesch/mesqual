---
name: mesqual-study
description: Analyze a mesqual study, the multi-scenario results of energy market simulations. Use when code imports mesqual or works with a StudyManager, scenarios, comparisons, datasets or flags. Gives the workflow of an analysis, with links to docs.mesqual.io. Discovery with describe_study, describe_datasets, describe_flags and flag_info (as_records=True for JSON). Fetch and its result conventions (index and column levels, the dataset level, comparisons as variation minus reference, units). Study-specific flags, KPIs, folium maps and HTML dashboards.
---

# Analyze a mesqual study

mesqual gives one `fetch` interface for all scenarios of a study, plus comparisons, KPIs, maps and dashboards. This skill gives the order of the steps. The flag names and level names below come from the synthetic platform `mesqual.testing`. In a real study, take them from the describe methods.

Docs: https://docs.mesqual.io/

## 1. Check the platform package

mesqual knows no modeling tool. A platform package reads the output of one modeling tool and gives the scenario datasets of a study.

- Search the dependencies and the code of the project for a `PlatformDataset` subclass outside `mesqual`. `mesqual.testing` is the synthetic test platform; it does not count.
- If there is none, ask the user which platform package to install, or whether to build one by the guide. Do not guess a package name.

Docs: https://docs.mesqual.io/guides/build-a-platform-package/

## 2. Find the study

- Search the project for the code that builds the `StudyManager`, usually `StudyManager.factory_from_scenarios`.
- A study has three collections: `study.scen` (scenarios), `study.comp` (comparisons) and `study.scen_comp` (both). Their names: `study.scenario_names` and `study.comparison_names`.
- No study yet, or you want to try a call: `mesqual.testing.make_dummy_study` builds a synthetic study offline.

Docs: https://docs.mesqual.io/api_reference/study_manager/, https://docs.mesqual.io/guides/use-in-a-study/#set-up-a-study, https://docs.mesqual.io/api_reference/testing/

## 3. Discover before you fetch

The describe methods load no time series data. With `as_records=True`, they return plain dicts, ready for JSON. `FlagInfo.as_record()` does the same for one flag.

```python
overview = study.describe_study(as_records=True)
scenarios = study.scen.describe_datasets(max_depth=1, as_records=True)
dataset = study.scen.get_dataset(study.scenario_names[0])
price_flags = dataset.describe_flags(contains='price', as_records=True)
template = next(row for row in price_flags if row['variants'])  # name '{area_layer}.Results.price'
flag = template['name'].format(**{key: values[0] for key, values in template['variants'].items()})
price_info = dataset.flag_info(flag).as_record()  # flag is 'Country.Results.price'
config = dataset.describe_config()  # the config fields of the dataset, with their effective values
```

- Never guess a flag name. Take it from `dataset.describe_flags()` or from `overview['scenario_flags']`.
<!-- --8<-- [start:templates] -->
- A name with `{...}` is the template of a spec with variants: `describe_flags` gives one row per spec. Replace each placeholder with one of its values from the column `variants`: `{area_layer}.Results.price` with `area_layer` `Country` gives `Country.Results.price`. `fetch` does not accept the template. `contains=` also matches the full names, for example `contains='generation_gas'`.
<!-- --8<-- [end:templates] -->
- Check the column `availability`: `available`, `partial` (some flags of the spec are unavailable), `unknown` or `unavailable`. `describe_flags` hides unavailable specs; `dataset.describe_flags(available_only=False)` shows them, and `dataset.availability(flag).reason` says why.
- Check the parameters of a flag in `dataset.flag_info()`. A required parameter has no default. Pass a parameter as a keyword: `dataset.fetch(flag, <param>=<value>)`.

Docs: https://docs.mesqual.io/api_reference/datasets/#describe-a-dataset, https://docs.mesqual.io/api_reference/datasets/#mesqual.datasets.FlagInfo, https://docs.mesqual.io/architecture/other-contracts/#describe-and-availability

## 4. Fetch and read the result

```python
prices = study.scen.fetch('Node.Results.price')
price_deltas = study.comp.fetch('Node.Results.price')
prices_and_deltas = study.scen_comp.fetch('Node.Results.price')
hourly_prices = dataset.fetch('Node.Results.price', granularity='1h')
first_solution_prices = dataset.fetch('Node.Results.price', solution='first')
levels = study.scen_comp.flag_info('Node.Results.price').schema  # index time; columns type, dataset, Node
```

<!-- --8<-- [start:conventions] -->
- `fetch` returns a pandas DataFrame or Series. The flag spec declares its index and column levels: the `schema` of `dataset.flag_info(flag)`. By default, a time series has a time index and one column per object of its model flag, for example per `Node`. For a model flag, the column level is unnamed (`None: dimension`).
- `study.scen` and `study.comp` add a column level on top, with the name of each scenario or comparison. `study.scen_comp` adds the level `type` above it. `study.scen.flag_info(flag).schema` shows all levels of a collection result, the collection levels included. With `StudyManager.factory_from_scenarios`, the scenario or comparison level is named `dataset`, the values of `type` are `scenario` and `comparison`, and a comparison is named `<variation> vs <reference>`.
- A comparison gives variation minus reference for numbers by default (parameter `comparison_type`). For text values, an unchanged entry keeps its value, a changed entry becomes `<variation> (was <reference>)`, an entry that only the reference has becomes `DELETED: <reference>`, and an entry that only the variation has becomes `<variation> (new)`. `comparison_type=ComparisonTypeEnum.BOTH` puts variation and reference side by side, in a last column level `compared` that `study.comp.flag_info(flag).schema` does not show.
- A scenario that lacks an object has no column for it in a time series. With `DELTA` (default) and `BOTH`, a comparison has the entries (objects and rows) of both sides; `VARIATION` has the entries of the variation. For numbers in a `DELTA`, a missing side counts as the `fill_value`: the parameter of the comparison, by default the value of the flag spec. Without a fill value, the result is NaN there.
- The collections join the results of their datasets outer. So a scenario has NaN where it has no value, for example outside its horizon. A comparison fills such entries with its `fill_value` first, as above.
- A collection leaves out a dataset on which the flag is unavailable: that dataset has no columns. If the flag is unavailable on all datasets, `fetch` raises `FlagUnavailableError`.
- The values are in the `unit` of `dataset.flag_info(flag)`: the unit of the flag spec, or for a layer flag the unit of its base spec (`unit_source`). The result carries no unit, and `fetch` converts no units.
<!-- --8<-- [end:conventions] -->

Details: https://docs.mesqual.io/architecture/core-contracts/#3-results, https://docs.mesqual.io/api_reference/datasets/#mesqual.datasets.Dataset.fetch, https://docs.mesqual.io/api_reference/datasets/#mesqual.datasets.DatasetConcatCollection, https://docs.mesqual.io/api_reference/flag/#mesqual.flag.Level

## 5. Compare scenarios

- The parameter `comparison_type` of `study.comp.fetch()` selects `ComparisonTypeEnum.DELTA` (default), `ComparisonTypeEnum.VARIATION` or `ComparisonTypeEnum.BOTH` (`from mesqual.enums import ComparisonTypeEnum`).
- For a quick metric, pandas on the result is enough:

```python
mean_prices = prices.mean().unstack('dataset')
mean_price_deltas = price_deltas.mean().unstack('dataset')
```

Docs: https://docs.mesqual.io/api_reference/datasets/#mesqual.datasets.DatasetComparison

## 6. Add a study-specific flag

A study adds its own flags on a subclass of its platform dataset, here called `StudyDataset`. Use the subclass of the project if it has one; else create one, and build the study from it. The registrations apply to this class and its subclasses only. The steps hold for every platform; the linked guide shows them on the synthetic platform.

- Write an interpreter: a subclass of the interpreter base of the platform, `StudyDataset.get_child_dataset_type()`, with `Dataset.flag_specs` (for example a `TimeSeriesFlagSpec`) and `_fetch`. The base gets the constructor arguments from the platform. For availability, follow the rules in the docstring of the base; `Dataset._is_available` is the hook only where the base leaves it open.
- In `_fetch`, fetch the inputs through `Dataset.parent_dataset` and pass `effective_config` to each `Dataset.fetch`, so the config of the call reaches the inputs. An interpreter that reads a file of its own records it with `FetchContext.record_file_read`, so the cache sees changes.
- Register it with `@StudyDataset.register_interpreter`. To replace a platform interpreter, pass `override_interpreter=` (the interpreter that `StudyDataset.get_answering_interpreter(flag)` gives) and `version=`.
- An area layer groups the objects of a model flag into areas, for example nodes into regions. It needs a membership in `ModelFlagSpec.memberships` of that model flag to the layer model, and a column of that name; an override of the model flag adds them if the platform has none. The layer model is an existing model flag (for example zones on `Country.Model`) or comes from an interpreter. `StudyDataset.register_area_layer` with an `AreaLayer` then gives the area flags of the layer specs of the platform.
- A custom layer aggregation (a subclass of `AreaAggregation` or `BorderAggregation`) gets `effective_config` after `variant` in `AreaAggregation.aggregate` or `BorderAggregation.aggregate`, and passes it to each fetch, as an interpreter does.
- Check the new flags with `dataset.describe_flags()`.
- In a notebook, define a subclass (of the project subclass, if there is one) in the same cell as its registrations, and build the study again after each run. An interpreter or override that is registered twice on one class raises (`ValueError` or `AmbiguousOverrideError`).

Docs: https://docs.mesqual.io/guides/extend-a-platform-in-a-study/, https://docs.mesqual.io/architecture/flags-and-layers/#5-layers

## 7. KPIs

A KPI is one value per object and dataset, with the attributes of both. Use KPIs for maps, and to filter or convert many values.

```python
from mesqual.kpis import Aggregations, ComparisonKPIBuilder, FlagAggKPIBuilder, ValueComparisons

definitions = FlagAggKPIBuilder().for_flag('Country.Results.price').with_aggregation(Aggregations.Mean).build()
study.scen.add_kpis_from_definitions_to_all_child_datasets(definitions)
kpi_table = study.scen.get_merged_kpi_collection().to_dataframe()

comparison_definitions = ComparisonKPIBuilder(definitions).with_comparisons([ValueComparisons.Increase]).build()
study.comp.add_kpis_from_definitions_to_all_child_datasets(comparison_definitions)
```

Docs: https://docs.mesqual.io/guides/use-in-a-study/#compute-kpis, https://docs.mesqual.io/api_reference/kpis/, https://docs.mesqual.io/api_reference/kpis/builders/, https://docs.mesqual.io/api_reference/kpis/collection/

## 8. Maps

`folviz.KPICollectionMapVisualizer` draws a KPI collection on a folium map, one feature group per KPI group. `folviz.PropertyMapper.from_kpi_value` maps the values to colors, for example with `valmap.SegmentedContinuousColorscale`.

The generator must fit the geometry of the objects. The model flag of a flag is `dataset.flag_info(flag).model_flag`; fetch it and look at its columns:

- polygons in `geometry`: `folviz.AreaGenerator`;
- points in `location`, `lat` and `lon`, or `geometry`: `folviz.CircleMarkerGenerator`;
- line strings in `geometry` or `line_string`: `folviz.LineGenerator`.

For another column, pass it to the feature resolver, for example `folviz.LineFeatureResolver(geometry=folviz.PropertyMapper.from_item_attr('geo_line_string'))`. A generator without geometry draws nothing, so check that the map has bounds before you use them.

```python
import folium

from mesqual.visualizations import folium_viz_system as folviz
from mesqual.visualizations import value_mapping_system as valmap

model_columns = dataset.fetch(dataset.flag_info('Country.Results.price').model_flag).columns  # has 'geometry'
colors = valmap.SegmentedContinuousColorscale.single_segment_autoscale_factory_from_array(kpi_table['value'], 'YlOrRd')
visualizer = folviz.KPICollectionMapVisualizer(
    generators=[folviz.AreaGenerator(folviz.AreaFeatureResolver(fill_color=folviz.PropertyMapper.from_kpi_value(colors)))],
    study_manager=study,
)
price_map = folium.Map()
visualizer.generate_and_add_feature_groups_to_map(study.scen.get_merged_kpi_collection(), price_map)
bounds = price_map.get_bounds()
assert None not in bounds[0], 'no features: choose the generator that fits the geometry'
price_map.fit_bounds(bounds)
price_map.save('price_map.html')
```

Docs: https://docs.mesqual.io/guides/use-in-a-study/#build-a-map, https://docs.mesqual.io/api_reference/visualization/folium_viz_system/kpi_collection/, https://docs.mesqual.io/api_reference/visualization/value_mapping/

## 9. Dashboards

`HTMLDashboard` puts figures, maps and tables into one HTML file. `TimeSeriesDashboardGenerator` makes heatmaps of time series, faceted by the column levels of a result. It takes at most two column levels: use `study.scen` or `study.comp`, or select one `type` of `study.scen_comp`, for example `prices_and_deltas['scenario']`.

```python
from mesqual.visualizations import HTMLDashboard, HTMLTable, TimeSeriesDashboardGenerator

# The facets are column levels of the result: see study.scen.flag_info(flag).schema.
heatmaps = TimeSeriesDashboardGenerator(x_axis='date', facet_col='dataset', facet_row='Country')
dashboard = HTMLDashboard(name='Prices')
dashboard.add_plotly_figure(heatmaps.get_figure(study.scen.fetch('Country.Results.price')), name='prices')
dashboard.add_table(HTMLTable(mean_prices.round(2).reset_index(), title='Mean price'))
dashboard.save('prices.html')
```

Check the numbers as text (a frame, `KPICollection.to_dataframe()`, records) before you build figures.

Docs: https://docs.mesqual.io/guides/use-in-a-study/#build-a-dashboard, https://docs.mesqual.io/api_reference/visualization/html_dashboard/, https://docs.mesqual.io/api_reference/visualization/time_series_dashboard/

## 10. When a call fails

- `UnknownFlagError`: no spec declares the flag. Check the name with `dataset.describe_flags()`. For a template name, replace each `{...}` placeholder with a value from the column `variants`.
- `FlagUnavailableError`: the flag is not available in this dataset. `dataset.availability(flag).reason` says why.
- A parameter value that is not allowed raises `ValueError` with the allowed values. See the parameters in `dataset.flag_info()`. A variant value is part of the flag name: a value that the spec does not declare gives `UnknownFlagError`, one that the dataset does not have gives `FlagUnavailableError`.
- `mesqual.utils.logging.ConsoleLogging.enable(logging.DEBUG)` prints all records of mesqual, for example the reason of a stale cache entry.

Docs: https://docs.mesqual.io/api_reference/utils/logging/
