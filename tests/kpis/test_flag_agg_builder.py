import pandas as pd

from mesqual.kpis.builders.flag_agg_builder import ModelPropertyFilter


def test_filter_funcs_combine_with_properties_and_query():
    model = pd.DataFrame(
        {'fuel': ['gas', 'gas', 'wind', 'gas'], 'capacity': [100, 200, 300, 400]},
        index=['A', 'B', 'C', 'D'],
    )
    model_filter = ModelPropertyFilter(
        properties={'fuel': 'gas'},
        query_expr='capacity > 100',
        filter_funcs={'capacity': lambda capacity: capacity < 400},
    )
    assert model_filter.apply_filter(model) == ['B']
