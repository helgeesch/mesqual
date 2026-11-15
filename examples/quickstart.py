"""Quickstart: a study of synthetic scenarios with comparisons and a dashboard."""

from pathlib import Path

import plotly.express as px

from mesqual.testing import make_dummy_study
from mesqual.visualizations import HTMLDashboard, HTMLTable

OUTPUT_FILE = Path(__file__).parent / '_tmp' / 'quickstart.html'


class Quickstart:
    """Builds the dummy study of synthetic scenarios and saves a tabbed dashboard.

    See `make_dummy_study` for the scenarios and comparisons.

    Args:
        output_file: Path of the dashboard HTML file.
    """

    def __init__(self, output_file: Path):
        self._output_file = output_file
        self.study = make_dummy_study()

    def describe(self):
        """Print the scenarios, their attributes and their flags."""
        print('Scenarios:', self.study.scenario_names)
        print('Comparisons:', self.study.comparison_names)
        print(self.study.scen.get_attributes_concat_df())
        print('Flag specs:')
        for spec in sorted(self.study.scen.flag_index.specs, key=lambda s: s.name):
            print(f'  {spec.name}: {spec.description}')

    def run(self):
        """Fetch scenario and comparison data and save the dashboard."""
        prices = self.study.scen.fetch('Node.Results.price')
        demand_deltas = self.study.comp.fetch('Node.Results.demand_total')
        mean_prices = prices.mean().unstack('dataset').round(2)
        mean_demand_deltas = demand_deltas.mean().unstack('dataset').round(2)
        print(mean_prices)
        print(mean_demand_deltas)

        dashboard = HTMLDashboard(name='MESQUAL quickstart')
        price_fig = px.line(prices.xs('N1', level='Node', axis=1), title='Price at node N1 (EUR/MWh)')
        dashboard.add_plotly_figure(price_fig, name='price_n1', tab='Scenarios')
        dashboard.add_table(HTMLTable(mean_prices.reset_index(), title='Mean price (EUR/MWh)'), tab='Scenarios')
        delta_fig = px.bar(mean_demand_deltas, barmode='group', title='Mean demand delta (MW)')
        dashboard.add_plotly_figure(delta_fig, name='demand_delta', tab='Comparisons')
        attributes = self.study.scen.get_attributes_concat_df().reset_index()
        dashboard.add_table(HTMLTable(attributes, title='Scenario attributes'), tab='Study')
        self._output_file.parent.mkdir(parents=True, exist_ok=True)
        dashboard.save(str(self._output_file))
        print('Dashboard saved to', self._output_file)


if __name__ == '__main__':
    quickstart = Quickstart(OUTPUT_FILE)
    quickstart.describe()
    quickstart.run()
