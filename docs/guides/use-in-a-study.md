# Use mesqual in a study

This guide goes from the installed packages to a map and a dashboard of a study. A study holds scenarios, each a dataset of a platform package, and comparisons between them. Here, the scenarios come from the synthetic platform `mesqual.testing`, so every step runs with mesqual alone. In your study, the dataset class of your platform package takes the place of `DummyDataset`.

## Set up a study

`StudyManager.factory_from_scenarios` takes the scenario datasets and the comparisons as `(variation, reference)` pairs. `DummyScenario` describes a synthetic scenario; `DummyDataset` is its platform dataset:

```python
from mesqual import StudyManager
from mesqual.testing import DummyDataset, DummyScenario

base = DummyScenario(name='base', start='2024-03-28', days=3)
study = StudyManager.factory_from_scenarios(
    scenarios=[
        DummyDataset(base),
        DummyDataset(base.replace(name='high_demand', demand_factor=1.2)),
        DummyDataset(base.replace(name='without_N4', removed_objects=('N4',))),
    ],
    comparisons=[('high_demand', 'base'), ('without_N4', 'base')],
)
print(study.scenario_names)
print(study.comparison_names)
```

```text
['base', 'high_demand', 'without_N4']
['high_demand vs base', 'without_N4 vs base']
```

A study has three collections: `study.scen` (the scenarios), `study.comp` (the comparisons) and `study.scen_comp` (both). For a ready-made synthetic study, use `mesqual.testing.make_dummy_study()`.

## Find the flags

A flag names one variable, for example `Node.Results.price`. Never guess a flag name: ask the study. The describe methods load no time series data:

```python
overview = study.describe_study()                       # scenarios, comparisons and their flags
scenarios = study.scen.describe_datasets(max_depth=1)  # the scenarios with descriptions and attributes
dataset = study.scen.get_dataset('base')
price_flags = dataset.describe_flags(contains='price')  # one row per flag spec, with availability
price_info = dataset.flag_info('Country.Results.price')  # description, unit, schema and parameters
print(price_flags[['variants', 'availability']])
```

```text
                                                        variants availability
name
Node.Results.price                                                  available
{area_layer}.Results.price                   area_layer: Country    available
{border_layer}.Results.price_spread  border_layer: CountryBorder    available
```

- A name with `{...}` is the template of a spec with variants, for example `{area_layer}.Results.price`. Replace each placeholder with one of its values from the column `variants`: `Country.Results.price`. `fetch` does not accept the template.
- The column `availability` says whether the dataset can answer the flag. `dataset.availability(flag).reason` says why. `describe_flags` hides unavailable specs by default; `available_only=False` shows them.
- With `as_records=True`, each describe method returns plain dicts, ready for JSON.

## Fetch scenario and comparison data

`fetch` returns a pandas frame. Each collection adds a column level on top of the levels of the flag:

```python
prices = study.scen.fetch('Node.Results.price')               # one column per scenario and node
price_deltas = study.comp.fetch('Node.Results.price')         # variation minus reference
prices_and_deltas = study.scen_comp.fetch('Node.Results.price')
nodes = study.scen.get_dataset('base').fetch('Node.Model')    # the model table of one scenario
print(prices.columns.names)
print(prices_and_deltas.columns.names)
```

```text
['dataset', 'Node']
['type', 'dataset', 'Node']
```

- A flag can have parameters, for example the solution of a run: `study.scen.fetch('Node.Results.price', solution='first')`. `flag_info` lists them.
- A comparison gives variation minus reference by default. The parameter `comparison_type` selects another type, for example `ComparisonTypeEnum.BOTH` for both sides (`from mesqual.enums import ComparisonTypeEnum`).
- The values are in the unit of the flag: `price_info.unit`. `fetch` converts no units.

For a quick metric, pandas on the result is enough:

```python
mean_prices = prices.mean().unstack('dataset')
mean_price_deltas = price_deltas.mean().unstack('dataset')
```

## Compute KPIs

A KPI is one value per object and dataset, with the attributes of both. A builder makes the KPI definitions; a collection computes them for each of its datasets:

```python
from mesqual.kpis import Aggregations, FlagAggKPIBuilder

definitions = FlagAggKPIBuilder().for_flag('Country.Results.price').with_aggregation(Aggregations.Mean).build()
study.scen.add_kpis_from_definitions_to_all_child_datasets(definitions)
kpis = study.scen.get_merged_kpi_collection()

kpi_table = kpis.to_dataframe()  # one row per KPI: name, object, dataset, value, unit and the attributes
mean_country_prices = kpi_table.pivot_table(index='object_name', columns='dataset_name', values='value')
print(mean_country_prices.round(1))
```

```text
dataset_name  base  high_demand  without_N4
object_name
C1            42.0         53.8        42.0
C2            46.3         59.3        41.1
```

`kpis.filter(object_name='C1')` keeps the KPIs of one object; `kpis.group_by('dataset_name')` splits them per dataset.

## Build a map

`KPICollectionMapVisualizer` draws a KPI collection on a folium map, with one feature group per KPI group, here one per scenario. The generator must fit the geometry of the objects: the countries have polygons in the column `geometry` of `Country.Model`, so `AreaGenerator` draws them. A `PropertyMapper` maps each KPI value to a color:

```python
import folium

from mesqual.visualizations import folium_viz_system as folviz
from mesqual.visualizations import value_mapping_system as valmap

colors = valmap.SegmentedContinuousColorscale.single_segment_autoscale_factory_from_array(kpi_table['value'], 'YlOrRd')
visualizer = folviz.KPICollectionMapVisualizer(
    generators=[
        folviz.AreaGenerator(
            folviz.AreaFeatureResolver(fill_color=folviz.PropertyMapper.from_kpi_value(colors), tooltip=True)
        )
    ],
)
price_map = folium.Map()
visualizer.generate_and_add_feature_groups_to_map(kpis, price_map, show='first')
price_map.fit_bounds(price_map.get_bounds())
folium.LayerControl().add_to(price_map)
price_map.save('price_map.html')
```

For points, use `folviz.CircleMarkerGenerator`; for lines, `folviz.LineGenerator`.

## Build a dashboard

`HTMLDashboard` puts figures, maps and tables into one HTML file. `TimeSeriesDashboardGenerator` makes heatmaps of time series, one facet per value of a column level:

```python
from mesqual.visualizations import HTMLDashboard, HTMLTable, TimeSeriesDashboardGenerator

heatmaps = TimeSeriesDashboardGenerator(x_axis='date', facet_col='dataset', facet_row='Country')
dashboard = HTMLDashboard(name='Country prices')
dashboard.add_plotly_figure(heatmaps.get_figure(study.scen.fetch('Country.Results.price')), name='prices')
dashboard.add_folium_map(price_map, name='mean_prices')
dashboard.add_table(HTMLTable(mean_country_prices.round(2).reset_index(), title='Mean price (EUR/MWh)'))
dashboard.save('study_dashboard.html')
```

## Next steps

- Add study-specific flags, or change a platform flag: [Extend a platform in a study](extend-a-platform-in-a-study.md).
- Connect mesqual to a modeling tool: [Build a platform package](build-a-platform-package.md).
- Let an agent run the analysis: [Use mesqual with agents](../use-with-agents.md).
- The contracts behind fetch, flags and collections: [Architecture](../architecture/index.md).
