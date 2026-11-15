import numpy as np
import pandas as pd
import pytest

from mesqual.visualizations.value_mapping_system import is_missing


@pytest.mark.parametrize('value', [None, float('nan'), np.nan, pd.NA, pd.NaT, np.datetime64('NaT', 'ns')])
def test_is_missing_for_missing_scalars(value):
    assert is_missing(value) is True


@pytest.mark.parametrize('value', [0, 0.0, '', 'a', (1, 2), [None], np.array([np.nan])])
def test_is_missing_false_for_values_and_containers(value):
    assert is_missing(value) is False
