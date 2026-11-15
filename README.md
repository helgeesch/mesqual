[![Python >=3.11](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![CI](https://github.com/helgeesch/mesqual/actions/workflows/ci.yml/badge.svg)](https://github.com/helgeesch/mesqual/actions/workflows/ci.yml)

# <img src="https://raw.githubusercontent.com/helgeesch/mesqual/18fe3fc20bace115a116555b2872d57925698e48/assets/logo-turq.svg" width="30" height="30" alt="logo"> MESQUAL 
<!-- --8<-- [start:home] -->
**M**odular **E**nergy **S**cenario Comparison Library for **Qu**antitative and **Qu**alitative An**al**ysis

mesqual is a Python framework for multi-scenario analysis of energy market simulations.

mesqual is made for humans and agents alike: its API lets anyone understand, navigate and analyze a study. The describe methods (`StudyManager.describe_study()`, `Dataset.describe_flags()`, `Dataset.flag_info()`) show what a study holds and load no time series data. So people and agents write study code against one documented API, not new pandas code for each study.

## Why mesqual

- **Scenarios and comparisons are objects.** A study holds scenarios and the comparisons between them. One `fetch` call gives a variable (a flag) for all scenarios (`study.scen`), all comparisons (`study.comp`) or both (`study.scen_comp`).
- **One API and framework over any simulation platform.** A platform package maps the output of one modeling tool to flags. Each flag has a description, a schema and, where it applies, a unit. Study code asks for flags, not for files.
- **KPIs, maps and dashboards.** KPI collections, interactive Folium maps and HTML dashboards use the same flags.
- **Energy logic, written once.** Units, volume-weighted area prices, area and border accounting, and time series granularity conversion are part of the package. A study does not write them again.
- **Extension in layers.** Platform packages extend mesqual. A study extends its platform with its own flags, interpreter overrides and area layers, and does not change the platform code.

## Installation

```bash
pip install mesqual
```

mesqual needs Python 3.11 or later. Optional extras:

- `color`: colored log output of `ConsoleLogging`, for example `pip install "mesqual[color]"`
- `screenshot`: map screenshots with `FoliumScreenshotter` (needs Chrome)

To work on mesqual itself, clone the [repository](https://github.com/helgeesch/mesqual) and run `uv sync --all-extras`. The [AGENTS.md](https://github.com/helgeesch/mesqual/blob/main/AGENTS.md) of the repository gives the rules for changes and the checks, which include the docs build. To preview the docs, run `uv run mkdocs serve`.

## Quickstart on synthetic data

`mesqual.testing` is a synthetic platform: it generates small scenarios of a network with countries, nodes, lines, generators and demands. It needs no data files. The comments say what each call returns.

```python
from mesqual.testing import make_dummy_study

study = make_dummy_study()  # scenarios base, high_demand, without_N4, short; comparisons against base

study.describe_study()                      # dict: scenarios, comparisons and their flags
study.scen.flag_info('Node.Results.price')  # FlagInfo: description, unit and schema of one flag

prices = study.scen.fetch('Node.Results.price')        # all scenarios: columns dataset, Node
price_deltas = study.comp.fetch('Node.Results.price')  # all comparisons: variation minus reference
```

The full runnable example, [`examples/quickstart.py`](https://github.com/helgeesch/mesqual/blob/main/examples/quickstart.py), builds the synthetic study, describes it and saves a tabbed HTML dashboard.

## Documentation

- Guides, step by step on the synthetic platform:
    - [Use mesqual in a study](https://docs.mesqual.io/guides/use-in-a-study/): set up a study, find the flags, fetch, KPIs, a map and a dashboard.
    - [Extend a platform in a study](https://docs.mesqual.io/guides/extend-a-platform-in-a-study/): study flags, interpreter overrides, area layers and custom variables.
    - [Build a platform package](https://docs.mesqual.io/guides/build-a-platform-package/): connect mesqual to a modeling tool.
- [Use mesqual with agents](https://docs.mesqual.io/use-with-agents/): install the skill with `python -m mesqual.skills install .`, and add the snippet to the `AGENTS.md` of your project.
- [Architecture](https://docs.mesqual.io/architecture/): the parts of mesqual, its principles and its contracts.
- [API reference](https://docs.mesqual.io/api_reference/study_manager/): the classes and functions of each module, from `StudyManager` on.
    - [Logging](https://docs.mesqual.io/api_reference/utils/logging/): console output with `ConsoleLogging`; otherwise the logging setup of your application applies.
<!-- --8<-- [end:home] -->

## mesqual 0.x and its packages

`mesqual-pypsa` and `mesqual-vanilla-studies` are deprecated. Deprecated packages work only with mesqual 0.x. For mesqual 2, start with the quickstart and guides at [docs.mesqual.io](https://docs.mesqual.io/), which run on the synthetic platform `mesqual.testing`.

## Attribution and Licenses
This project is licensed under the GNU Lesser General Public License v3.0 or later. See [LICENSES/LGPL-3.0-or-later.txt](https://github.com/helgeesch/mesqual/blob/main/LICENSES/LGPL-3.0-or-later.txt).

### Third-party assets
- `countries.geojson`: Made with [Natural Earth](https://github.com/nvkelso/natural-earth-vector.git)

## Contact
For questions or feedback, don't hesitate to [open an issue](https://github.com/helgeesch/mesqual/issues) or reach out via LinkedIn.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/helge-e-8201041a7/)
