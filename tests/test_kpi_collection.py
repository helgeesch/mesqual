import pandas as pd
import pytest

from mesqual.kpis.attributes import KPIAttributes
from mesqual.kpis.collection import KPICollection
from mesqual.kpis.kpi import KPI
from mesqual.units import Units
from tests.helpers import FrameDataset


def make_kpi(value: float, dataset_attributes: dict | None = None, **attributes) -> KPI:
    dataset = FrameDataset(name='ds', attributes=dataset_attributes)
    kpi_attributes = KPIAttributes(
        flag='items.price',
        unit=Units.EUR_per_MWh,
        dataset_name=dataset.name,
        dataset_attributes=dataset.attributes,
        **attributes,
    )
    return KPI(value=value, attributes=kpi_attributes, dataset=dataset)


@pytest.mark.parametrize('falsy_value', [0, False, ''])
def test_group_by_keeps_falsy_kpi_attribute(falsy_value):
    kpi = make_kpi(1.0, dataset_attributes={'object_name': 'from_dataset'}, object_name=falsy_value)

    groups = KPICollection([kpi]).group_by('object_name')

    assert list(groups) == [(falsy_value,)]


def test_group_by_falls_back_to_dataset_attribute_when_missing():
    kpi = make_kpi(1.0, dataset_attributes={'year': 2030})
    assert list(KPICollection([kpi]).group_by('year')) == [(2030,)]


def test_filter_by_kpi_attributes_raises_on_query_syntax_error():
    collection = KPICollection([make_kpi(1.0)])
    with pytest.raises(SyntaxError):
        collection.filter_by_kpi_attributes(query_expr='value >')


def test_filter_by_kpi_attributes_raises_on_column_no_kpi_has():
    collection = KPICollection([make_kpi(1.0), make_kpi(2.0)])
    with pytest.raises(pd.errors.UndefinedVariableError):
        collection.filter_by_kpi_attributes(query_expr='valeu > 0')


def test_filter_by_kpi_attributes_query_excludes_kpis_missing_the_column():
    with_year = make_kpi(1.0, dataset_attributes={'year': 2030})
    without_year = make_kpi(2.0)

    result = KPICollection([with_year, without_year]).filter_by_kpi_attributes(query_expr='year == 2030')

    assert list(result) == [with_year]


def test_filter_by_kpi_attributes_treats_raising_filter_function_as_no_match():
    numeric = make_kpi(1.0, dataset_attributes={'year': 2030})
    text = make_kpi(2.0, dataset_attributes={'year': 'n/a'})

    result = KPICollection([numeric, text]).filter_by_kpi_attributes(filter_funcs={'year': lambda y: y > 2000})

    assert list(result) == [numeric]


def test_filter_by_kpi_attributes_raises_on_typo_query_with_attribute_filter():
    collection = KPICollection([make_kpi(1.0, dataset_attributes={'year': 2030})])
    with pytest.raises(pd.errors.UndefinedVariableError):
        collection.filter_by_kpi_attributes(attributes={'year': 2031}, query_expr='valeu > 0')


def test_filter_by_kpi_attributes_treats_type_error_in_query_as_no_match():
    numeric = make_kpi(1.0, dataset_attributes={'year': 2030})
    text = make_kpi(2.0, dataset_attributes={'year': 'n/a'})

    result = KPICollection([numeric, text]).filter_by_kpi_attributes(query_expr='year > 2000')

    assert list(result) == [numeric]


def make_model_kpi(object_name: str) -> KPI:
    return make_kpi(1.0, model_flag=FrameDataset.FLAG, object_name=object_name)


def test_filter_by_model_properties_with_query_and_properties():
    german, french = make_model_kpi('a'), make_model_kpi('b')
    collection = KPICollection([german, french])

    assert list(collection.filter_by_model_properties(query_expr='country == "FR"')) == [french]
    assert list(collection.filter_by_model_properties(properties={'country': ['DE']})) == [german]
    assert list(collection.filter_by_model_properties(properties={'country': 'DE'}, query_expr='value > 1')) == []


def test_filter_by_model_properties_raises_on_typo_query():
    collection = KPICollection([make_model_kpi('a'), make_model_kpi('b')])
    with pytest.raises(pd.errors.UndefinedVariableError):
        collection.filter_by_model_properties(properties={'country': 'DE'}, query_expr='contry == "DE"')


def test_filter_by_kpi_attributes_treats_failing_str_accessor_as_no_match():
    text = make_kpi(1.0, dataset_attributes={'label': 'abc'})
    number = make_kpi(2.0, dataset_attributes={'label': 5})

    result = KPICollection([text, number]).filter_by_kpi_attributes(query_expr='label.str.contains("a")')

    assert list(result) == [text]
