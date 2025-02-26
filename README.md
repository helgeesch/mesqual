

# <img src="assets/logo_no_text_no_bg.svg" width="70" height="70" alt="logo"> MESCAL 
**M**odular **E**nergy **S**cenario **C**omparison **A**nalysis and **L**ibrary

A Python framework for energy market data analysis, with a focus on scenario comparison and KPI calculation.

## Overview

mescal provides a flexible framework for handling energy market data from various sources (simulations, real market data, scenarios). Its modular architecture allows easy integration with different energy market platforms and tools through dedicated connector packages.

Key features:
- Unified interface for handling energy market data across different platforms
- Built-in scenario comparison capabilities
- Extensible KPI calculation framework
- Flexible data aggregation and transformation tools
- Support for time series analysis and topology-based computations

## Minimum usage examples

#### Example using Plexos interface to set up simple dataset and fetch data
```python
from mescal_plexos import PlexosDataset

# Initialize dataset
dataset = PlexosDataset.from_paths(model='plexos_model.xml', solution='my_solution.zip', name='my_name')

# Fetch data
df_prices = dataset.fetch("ST.Node.Price")
df_nodes = dataset.fetch("Node.Model")
```



#### Example using PyPSA interface to set up a study with multiple scenarios and scenario comparisons

```python
import pypsa
from mescal import StudyManager
from mescal_pypsa import PyPSADataset

# Create scenarios
n_base = pypsa.Network('your_base_network.nc')
n_base.name = 'base'
n_scen1 = pypsa.Network('your_scen1_network.nc')
n_scen1.name = 'scen1'
n_scen2 = pypsa.Network('your_scen2_network.nc')
n_scen2.name = 'scen2'

# Initialize study manager
study = StudyManager.factory_from_scenarios(
    scenarios=[
        PyPSADataset(n_base),
        PyPSADataset(n_scen1),
        PyPSADataset(n_scen2),
    ],
    comparisons=[("scen1", "base"), ("scen2", "base")],
    export_folder="output"
)

# Access MultiIndex DF with data for all scenarios
df_prices = study.scen.fetch("buses_t.marginal_price")

# Access MultiIndex DF with data for all comparisons (delta values)
df_price_deltas = study.comp.fetch("buses_t.marginal_price")

# Access buses model df of base case
df_bus_model = study.scen.get_dataset('base').fetch('buses')
```

For more elaborate and practical examples, please visit to the [mescal-vanilla-studies](https://github.com/helgeesch/mescal-vanilla-studies.git) repository.


## Requirements
- Python ≥ 3.10
- Dependencies listed in requirements.txt

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
- Data transformation utilities
- Pandas / Plotly / Folium utilities

## Integrate mescal and mescal-interface packages in your project

MESCAL is designed to be used as a Git submodule. To include it in your project:

```bash
git submodule add https://github.com/helgeesch/mescal.git submodules/mescal
git submodule add [https://repository/url/to/any/mescal-interface.git] submodules/mescal-interface
git submodule update --init --recursive
```
The folder `submodules/` should now include the respective packages.

### Install requirements:
```bash
pip install -r requirements.txt
pip install -r submodules/mescal/requirements.txt
pip install -r submodules/mescal-interface/requirements.txt
```

### PyCharm Configuration
If you're using PyCharm, ensure that the submodule directories are properly recognized as part of the source code by setting them as "Sources Root":

1. In PyCharm's Project Explorer, locate the submodule directories and the relevant source code directories:
   - `submodules/mescal`
   - `submodules/mescal/mescal`
   - `submodules/mescal-interface`
   - `submodules/mescal-interface/mescal_interface`
2. Right-click on each of the directories above.
3. Select Mark Directory as -> Sources Root.


### VSCode Configuration
In Visual Studio Code, you can add the submodules to the python.analysis.extraPaths setting:
1. Open your project folder.
2. Create (or modify) .vscode/settings.json:
    ```json
    {
        "python.analysis.extraPaths": [
          "submodules/mescal",
          "submodules/mescal/mescal",
          "submodules/mescal-interface",
          "submodules/mescal-interface/mescal_interface"
        ]
    }
    ```

### Jupyter Notebook Configuration
If you work with Jupyter, extend the sys.path directly in your notebook:
```python
import sys
sys.path.append("submodules/mescal")
sys.path.append("submodules/mescal/mescal")
sys.path.append("submodules/mescal-interface")
sys.path.append("submodules/mescal-interface/mescal_interface")
```

## License

This project is licensed under the LGPL License - see the LICENSE file for details.

## Contributing

Contributions are welcome! The modular architecture makes it easy to:
- Add support for new platforms via interface packages
- Add or enhance new visualization modules
- Add data transformation utilities
- Be creative :)
