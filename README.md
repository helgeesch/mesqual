# mescal

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

## Requirements

- Python ≥ 3.10
- Dependencies are listed in requirements.txt

## Architecture

mescal follows a modular design where platform-specific implementations are handled through separate packages:

```
mescal/             # Core package
mescal_plexos/      # PLEXOS connector (separate package)
mescal_pypsa/       # PyPSA connector (separate package)
...                 # Other platform connectors
```

The core package provides:
- Abstract interfaces for data handling
- Base classes for platform-specific implementations
- Scenario comparison tools
- KPI calculation framework
- Data transformation utilities

## Usage Example

```python
# TODO: mock DataSet; mock study
from mescal.study_manager import StudyManager
from mescal.data_sets import DataSet

# Create scenarios
base_case = DataSet("base_case")
scenario_1 = DataSet("scenario_1")

# Initialize study manager
study = StudyManager.factory_from_scenarios(
    scenarios=[base_case, scenario_1],
    comparisons=[("scenario_1", "base_case")],
    export_folder="results"
)

# Access scenario data
prices = study.scen.fetch("Node.Price")

# Access comparison data
price_delta = study.comp.fetch("Node.Price")
```

## Integration with Platform-Specific Packages

To use mescal with a specific platform, install the corresponding connector package:

```python
# Example using PyPSA connector
from mescal_pypsa import PyPSADataSet

# Load PyPSA network
dataset = PyPSADataSet("my_scenario")
prices = dataset.fetch("Node.Price")
```

## Development

Currently, mescal is designed to be used as a Git submodule. To include it in your project:

```bash
git submodule add [repository_url] mescal
git submodule update --init --recursive
```

## License

This project is licensed under the LGPL License - see the LICENSE file for details.

## Contributing

Contributions are welcome! The modular architecture makes it easy to:
- Add support for new platforms via connector packages
- Implement new KPIs
- Add data transformation utilities
- Enhance visualization capabilities