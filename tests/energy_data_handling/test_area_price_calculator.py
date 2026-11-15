import numpy as np
import pandas as pd

from mesqual.energy_data_handling.area_accounting import AreaPriceCalculator

NODES = pd.DataFrame({'zone': ['A', 'A', 'B']}, index=['a1', 'a2', 'b1'])
TIME = pd.date_range('2024-01-01', periods=2, freq='h')
PRICES = pd.DataFrame({'a1': [10.0, 20.0], 'a2': [30.0, 40.0], 'b1': [50.0, 60.0]}, index=TIME)


def test_simple_average_per_area():
    result = AreaPriceCalculator(NODES, 'zone').calculate(PRICES)
    assert result['A'].tolist() == [20.0, 30.0]
    assert result['B'].tolist() == [50.0, 60.0]


def test_weighted_average_per_area():
    demand = pd.DataFrame({'a1': [3.0, 1.0], 'a2': [1.0, 1.0], 'b1': [5.0, 5.0]}, index=TIME)
    result = AreaPriceCalculator(NODES, 'zone').calculate(PRICES, demand)
    assert result['A'].tolist() == [15.0, 30.0]


def test_weights_that_sum_to_zero_give_nan():
    demand = pd.DataFrame({'a1': [0.0, 1.0], 'a2': [0.0, 1.0], 'b1': [5.0, 5.0]}, index=TIME)
    result = AreaPriceCalculator(NODES, 'zone').calculate(PRICES, demand)
    assert np.isnan(result.loc[TIME[0], 'A'])
    assert result.loc[TIME[1], 'A'] == 30.0


def test_area_without_nodes_in_the_data_has_no_column():
    result = AreaPriceCalculator(NODES, 'zone').calculate(PRICES.drop(columns='b1'))
    assert result.columns.tolist() == ['A']
