from types import SimpleNamespace

import numpy as np
import pytest

from mesqual.flag import REQUIRED, Dimension, FlagUnavailableError, VariantValues
from tests.flag.specs import SOLUTION


class TestDimension:
    def test_values_are_read_only(self):
        with pytest.raises(TypeError):
            SOLUTION.values['new'] = 'x'

    def test_default_and_required(self):
        assert SOLUTION.has_default
        assert not Dimension('scenario', 'Scenario.', {'a': 'a'}).has_default
        assert Dimension('scenario', 'Scenario.').default is REQUIRED

    def test_check_value_rejects_value_outside_values(self):
        SOLUTION.check_value('first')
        with pytest.raises(ValueError, match='not a value of'):
            SOLUTION.check_value('best')

    def test_check_value_keeps_bool_and_int_apart(self):
        axis = Dimension('axis', 'Axis.', {None: 'default', 0: 'rows', 1: 'columns'})
        flag = Dimension('flag', 'Flag.', {False: 'off', True: 'on'})
        numpy_flag = Dimension('numpy_flag', 'Flag.', {np.False_: 'off', np.True_: 'on'})
        for dimension, value in (
            (axis, 0),
            (axis, np.int64(1)),
            (flag, True),
            (flag, np.True_),
            (flag, np.False_),
            (numpy_flag, True),
            (numpy_flag, np.False_),
        ):
            dimension.check_value(value)
        for dimension, value in (
            (axis, False),
            (axis, True),
            (axis, np.False_),
            (axis, np.True_),
            (flag, 0),
            (flag, 1),
            (flag, 1.0),
            (flag, np.int64(1)),
            (numpy_flag, 0),
            (numpy_flag, 1),
        ):
            with pytest.raises(ValueError, match='not a value of'):
                dimension.check_value(value)

    def test_check_value_runs_validate_for_open_values(self):
        def positive(value):
            if value <= 0:
                raise ValueError('must be positive')

        dimension = Dimension('threshold', 'Threshold.', validate=positive, default=1)
        dimension.check_value(5)
        with pytest.raises(ValueError, match='positive'):
            dimension.check_value(-1)

    def test_values_per_dataset_need_a_dataset(self):
        dimension = Dimension('fuel', 'Fuel.', lambda ds: {f: f for f in ds})
        assert dimension.is_resolved_per_dataset
        assert dict(dimension.values_for(['gas'])) == {'gas': 'gas'}
        with pytest.raises(ValueError, match='per dataset'):
            dimension.values_for()

    def test_same_as_compares_name_values_and_default(self):
        assert SOLUTION.same_as(Dimension('solution', 'Other text.', dict(SOLUTION.values), default='final'))
        assert not SOLUTION.same_as(Dimension('solution', 'x', dict(SOLUTION.values), default='first'))
        assert not SOLUTION.same_as(Dimension('solution', 'x', {'final': 'final solution'}, default='final'))

    def test_values_of_wrong_type_raise(self):
        with pytest.raises(TypeError):
            Dimension('x', 'x', values=['a', 'b'])


class TestVariantValues:
    def test_attribute_access_and_dict(self):
        values = VariantValues({'side': 'sent', 'direction': 'net'})
        assert (values.side, values.direction) == ('sent', 'net')
        assert values.as_dict() == {'side': 'sent', 'direction': 'net'}

    def test_wrong_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match='sid'):
            VariantValues({'side': 'sent'}).sid

    def test_read_only(self):
        with pytest.raises(AttributeError):
            VariantValues({'side': 'sent'}).side = 'received'

    def test_equality_and_hash(self):
        assert VariantValues({'a': 1}) == VariantValues({'a': 1})
        assert len({VariantValues({'a': 1}), VariantValues({'a': 1})}) == 1


class TestResolve:
    def test_given_value_else_default(self):
        assert SOLUTION.resolve({}) == 'final'
        assert SOLUTION.resolve({'solution': 'first', 'other': 1}) == 'first'

    def test_required_without_value_raises(self):
        scenario = Dimension('scenario', 'Scenario.', {'a': 'a'})
        with pytest.raises(ValueError, match="^parameter 'scenario' is required.$"):
            scenario.resolve({})
        assert scenario.resolve({'scenario': 'a'}) == 'a'

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="'best' is not a value of 'solution'"):
            SOLUTION.resolve({'solution': 'best'})

    def test_invalid_default_raises(self):
        with pytest.raises(ValueError, match="'best' is not a value of 'solution'"):
            Dimension('solution', 'x', {'final': 'final'}, default='best').resolve({})

    def test_validate_runs_for_open_values(self):
        def positive(value):
            if value <= 0:
                raise ValueError('must be positive')

        threshold = Dimension('threshold', 'x', default=1, validate=positive)
        assert threshold.resolve({'threshold': 5}) == 5
        with pytest.raises(ValueError, match='must be positive'):
            threshold.resolve({'threshold': -1})

    def test_values_per_dataset_are_checked_only_with_a_dataset(self):
        def reject(value):
            raise ValueError('validate ran')

        fuel = Dimension('fuel', 'x', lambda dataset: {'coal': 'coal'}, default='gas', validate=reject)
        assert fuel.resolve({}) == 'gas'
        with pytest.raises(ValueError, match="'gas' is not a value of 'fuel'"):
            fuel.resolve({}, dataset=object())


class TestValuesOfAnUnavailableFlag:
    DATASET = SimpleNamespace(name='base')

    @staticmethod
    def unavailable(dataset):
        raise FlagUnavailableError('Generator.Model', 'base', 'no generator table')

    def test_values_for_is_empty_when_the_values_function_fetches_an_unavailable_flag(self, caplog):
        fuel = Dimension('fuel', 'Fuel.', self.unavailable)
        with caplog.at_level('DEBUG', logger='mesqual.flag.dimension'):
            assert dict(fuel.values_for(self.DATASET)) == {}
        assert 'no generator table' in caplog.text

    def test_check_value_names_the_unavailable_flag(self):
        fuel = Dimension('fuel', 'Fuel.', self.unavailable)
        expected = "values of 'fuel' need the unavailable flag Generator.Model: no generator table"
        with pytest.raises(FlagUnavailableError, match=expected):
            fuel.check_value('coal', self.DATASET)

    def test_other_errors_of_the_values_function_raise(self):
        def broken(dataset):
            raise KeyError('fuel')

        with pytest.raises(KeyError):
            Dimension('fuel', 'Fuel.', broken).values_for(object())


def test_flag_unavailable_error_keeps_its_fields_through_pickle():
    import pickle

    error = pickle.loads(pickle.dumps(FlagUnavailableError('Node.Results.price', 'base', 'no price table')))
    assert (error.flag, error.dataset, error.reason) == ('Node.Results.price', 'base', 'no price table')
    assert str(error) == 'Flag Node.Results.price is not available on dataset base: no price table'
