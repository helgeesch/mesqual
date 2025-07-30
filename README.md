[![Python >=3.10](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
![Notebook Tests](https://github.com/helgeesch/mescal/actions/workflows/test-with-vanilla-studies.yml/badge.svg)

# MESCAL <img src="assets/logo_no_text_no_bg.svg" width="70" height="70" alt="logo">
**M**odular **E**nergy **S**cenario **C**omparison **A**nalysis **L**ibrary

A modular Python framework for energy market data analysis, with a focus on scenario comparison, KPI calculation and interactive visualizations.

## Overview

MESCAL provides a flexible framework for handling energy market data from various sources (simulations, real market data, scenarios). Its modular architecture allows easy integration with different energy market platforms and tools through dedicated interface packages.

**MESCAL value proposition and feature highlights**:
- Unified interface for handling energy systems data across any modeling platform, or custom data sources and formats
- Efficient data handling for multi-scenario and multi-scenario-comparison studies
- Built-in scenario comparison capabilities
- Modular design enabling easy integration of **study-specific** modules, data-sources, data-handling rules and custom variables 
- Extensible KPI calculation framework with unit handling
- Visualization modules for interactive maps, time-series dashboards, and more
- Library of flexible data aggregation and transformation tools
- Package for area layer accounting (topological aggregation), enabling spatial and temporal comparison between simulations of different raw topologies
- Support for time series analysis and topology-based variable computations
- Library of common utilities for handling pandas, plotly, colors, string-conventions
- And so much more...

This is the foundation package for a whole suite of libraries and repositories. 
In most cases, you will want to combine this foundation package with at least one existing mescal-platform-interface (e.g. mescal-pypsa, mescal-plexos, ...), or build your own.

To view a hands-on repository and see how the MESCAL-suite is used in action, please visit the vanilla-studies repository. For platform-interfaces, visit those, respectively. The full list of the current MESCAL-suite is:
- [mescal](https://github.com/helgeesch/mescal)
- [mescal-vanilla-studies](https://github.com/helgeesch/mescal-vanilla-studies)
- [mescal-pypsa](https://github.com/helgeesch/mescal-pypsa)
- [mescal-plexos](https://github.com/helgeesch/mescal-plexos) (to be released)

[//]: # (- [mescal-etp]&#40;https://github.com/helgeesch/mescal-etp&#41; &#40;to be released&#41;)
[//]: # (- [mescal-gui]&#40;https://github.com/helgeesch/mescal-gui&#41; &#40;to be released&#41;)
[//]: # (- [mescal-antares]&#40;https://github.com/helgeesch/mescal-antares&#41; &#40;to be released&#41;)
[//]: # (- [mescal-bid3]&#40;https://github.com/helgeesch/mescal-bid3&#41; &#40;to be released&#41;)

---

## Minimum usage examples

#### Example using PyPSA interface to set up a study with multiple scenarios and scenario comparisons

```python
import pypsa
from mescal import StudyManager
from mescal_pypsa import PyPSADataset

# Load networks
n_base = pypsa.Network('your_base_network.nc')
n_scen1 = pypsa.Network('your_scen1_network.nc')
n_scen2 = pypsa.Network('your_scen2_network.nc')

# Initialize study manager
study = StudyManager.factory_from_scenarios(
    scenarios=[
        PyPSADataset(n_base, name='base'),
        PyPSADataset(n_scen1, name='scen1'),
        PyPSADataset(n_scen2, name='scen2'),
    ],
    comparisons=[("scen1", "base"), ("scen2", "base")],
    export_folder="output"
)

# Access MultiIndex df with data for all scenarios
df_prices = study.scen.fetch("buses_t.marginal_price")

# Access MultiIndex df with data for all comparisons (delta values)
df_price_deltas = study.comp.fetch("buses_t.marginal_price")

# Access buses model df of base case
df_bus_model = study.scen.get_dataset('base').fetch('buses')
```

#### Example using Plexos interface to set up simple dataset and fetch data
```python
from mescal_plexos import PlexosDataset

# Initialize dataset
dataset = PlexosDataset.from_xml_and_solution_zip(
   model='path/to/my_plexos_model.xml', 
   solution='path/to/my_plexos_solution.zip',
   name='my_name',
)

# Fetch data as DataFrame
df_prices = dataset.fetch("ST.Node.Price")
df_nodes = dataset.fetch("Node.Model")
```

For more elaborate and practical examples, please visit the [mescal-vanilla-studies](https://github.com/helgeesch/mescal-vanilla-studies.git) repository.

---

## Requirements
- Python ≥ 3.10
- Install runtime dependencies with: `pip install -e .`
- Some arrow visualization modules require the [captain-arro](https://github.com/helgeesch/captain-arro/tree/main) package. You can either integrate it as a submodule and install it in the editable mode (use same instructions as described in Option 2 below), or install it directly via 
- ```bash 
  pip install git+https://github.com/helgeesch/captain-arro.git
  ```

---

## Architecture
MESCAL follows a modular design where platform-specific implementations are handled through separate packages:

```
mescal/                         # Core package
mescal-pypsa/                   # PyPSA interface (separate package)
mescal-plexos/                  # PLEXOS interface (separate package)
...                             # Other platform interfaces
mescal-your-custom-interface/   # Custom interface for your platform
```

The core package provides:
- Abstract interfaces for data handling
- Base classes for platform-specific implementations
- Scenario comparison tools
- KPI calculation framework
- Visualization modules
- Data transformation modules and utilities
- Pandas / Plotly / Folium utilities

## Integrate mescal and mescal-interface packages in your project

There are two options: You can either install the packages via pip and git, or you can include them as submodules and install them in editable mode for easy development.

### Option 1: Install the package(s) via pip and git
```bash
pip install git+https://github.com/helgeesch/mescal.git
pip install git+https://github.com/helgeesch/mescal-any-interface.git
```

### Option 2: Add packages as submodules and install in editable mode
#### Step 1: Navigate to your study-repository
In your console, navigate to the repository to which you want to add mescal submodules. This could be a study-repo similar to [mescal-vanilla-studies](https://github.com/helgeesch/mescal-vanilla-studies.git) or any git repo in which you handle your studies. 

#### Step 2: Add mescal submodules
Add all required mescal packages as submodules. If you want to build your own interface, just start by including the foundation package and start building your-custom-mescal-interface. If you want to integrate an existing interface, just add that one as a submodule, respectively.
```bash
git submodule add https://github.com/helgeesch/mescal.git submodules/mescal
git submodule add https://github.com/path/to/any/mescal-any-interface.git submodules/mescal-any-interface
git submodule update --init --recursive
```
The folder `submodules/` should now include the respective packages.

#### Step 3: Install in editable mode
```bash
pip install -e ./submodules/mescal
pip install -e ./submodules/mescal-any-interface
```

## Attribution and Licenses
This project is licensed under the LGPL License - see the LICENSE file for details.

### Third-party assets:
- `countries.geojson`: Made with [Natural Earth](https://github.com/nvkelso/natural-earth-vector.git)

## Contact
For questions or feedback, don't hesitate to [open an issue](https://github.com/helgeesch/mescal/issues) or reach out via LinkedIn.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/helge-e-8201041a7/)