import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from mesqual.datasets import Availability
from mesqual.flag import UnknownFlagError
from mesqual.testing import DummyDataset, DummyScenario
from mesqual.validation import (
    CheckResult,
    CheckStatus,
    Comparison,
    ConstraintCheck,
    DatasetCheck,
    DatasetValidator,
    ValidationReport,
)

CONSTANT = DummyScenario(name='constant', profile='constant', tz=None, days=1)
FLOW = 'Line.Results.flow_sent_up'
DEMAND = 'Demand.Results.demand_hourly'


@pytest.fixture(scope='module')
def dataset():
    return DummyDataset(CONSTANT)


@pytest.fixture(scope='module')
def without_demand():
    return DummyDataset(CONSTANT.replace(name='no_demand', missing_tables=('demand_load',)))


class FlowCount(DatasetCheck):
    FLAGS = (FLOW,)

    def _run(self, dataset):
        return self.result(CheckStatus.INFO, f'{dataset.fetch(FLOW).size} flow values.')


class DemandCount(DatasetCheck):
    FLAGS = (FLOW, DEMAND)

    def _run(self, dataset):
        return self.result(CheckStatus.PASS, 'demand present.')


class Raises(DatasetCheck):
    def _run(self, dataset):
        raise ZeroDivisionError('division by zero')


class FetchesUnavailable(DatasetCheck):
    def _run(self, dataset):
        dataset.fetch(DEMAND)
        return self.result(CheckStatus.PASS, 'unreachable.')


class Typo(DatasetCheck):
    FLAGS = ('Line.Results.flow_sent_upp',)

    def _run(self, dataset):
        return self.result(CheckStatus.PASS, 'unreachable.')


class UnmetRaises(FlowCount):
    def _unmet(self, dataset):
        raise RuntimeError('broken hook')


@dataclass
class Tolerance(DatasetCheck):
    tol: float = 0.0

    def _run(self, dataset):
        flow = dataset.fetch(FLOW)
        return self.compared(Comparison.of(flow + 0.5, flow, tol=self.tol), 'line MTUs off')


class TestCheckStatus:
    def test_values_and_text(self):
        assert [str(s) for s in CheckStatus] == ['PASS', 'FAIL', 'INFO', 'SKIP', 'ERROR']
        assert CheckStatus('FAIL') is CheckStatus.FAIL


class TestCheckResult:
    def test_details_default_empty_and_not_compared(self):
        result = CheckResult('a', CheckStatus.PASS, 'ok')
        assert result.details.empty
        assert result == CheckResult('a', CheckStatus.PASS, 'ok', pd.DataFrame({'x': [1]}))


class TestComparison:
    def test_violations_beyond_tolerance(self):
        index = pd.Index([1, 2, 3], name='time')
        actual = pd.DataFrame({'L1': [1.0, 2.0, 3.5], 'L2': [1.0, np.nan, 1.0]}, index=index)
        expected = pd.DataFrame({'L1': [1.0, 2.05, 3.0], 'L2': [1.0, 1.0, 1.0]}, index=index)
        comparison = Comparison.of(actual, expected, tol=0.1)
        assert comparison.compared == 5
        assert comparison.violations.to_dict('records') == [
            {'time': 3, 'column': 'L1', 'actual': 3.5, 'expected': 3.0, 'difference': 0.5}
        ]

    def test_names_unnamed_column_levels(self):
        columns = pd.MultiIndex.from_tuples([('L1', 'up')])
        frame = pd.DataFrame([[1.0]], index=pd.Index([1], name='time'), columns=columns)
        assert list(Comparison.long(frame).index.names) == ['time', 'column_0', 'column_1']

    def test_different_levels_raise(self):
        frame = pd.DataFrame({'L1': [1.0]}, index=pd.Index([1], name='time'))
        with pytest.raises(ValueError, match='2 index and column levels, expected 1'):
            Comparison.of(frame, frame['L1'])
        with pytest.raises(ValueError, match='levels'):
            Comparison.of(pd.Series([1.0]), frame.reset_index(drop=True))

    def test_levels_matched_by_name(self):
        index = pd.MultiIndex.from_tuples([(1, 'L1'), (1, 'L2')], names=['time', 'line'])
        actual = pd.Series([1.0, 5.0], index=index)
        expected = pd.Series([1.0, 2.0], index=index.reorder_levels(['line', 'time']))
        comparison = Comparison.of(actual, expected)
        assert comparison.compared == 2
        assert comparison.violations[['time', 'line', 'expected']].to_dict('records') == [
            {'time': 1, 'line': 'L2', 'expected': 2.0}
        ]

    def test_other_level_names_raise(self):
        actual = pd.Series([1.0], index=pd.Index([1], name='time'))
        with pytest.raises(ValueError, match='level names differ'):
            Comparison.of(actual, pd.Series([1.0], index=pd.Index([1], name='hour')))

    def test_unnamed_levels_matched_by_position(self):
        actual = pd.Series([1.0, 2.0], index=pd.Index([1, 2], name='time'))
        comparison = Comparison.of(actual, pd.Series([1.0, 3.0], index=[1, 2]))
        assert comparison.violations[['time', 'expected']].to_dict('records') == [{'time': 2, 'expected': 3.0}]
        assert Comparison.of(pd.Series([1.0]), pd.Series([2.0])).violations.columns[0] == 'index'

    def test_reserved_names_and_duplicates_raise(self):
        with pytest.raises(ValueError, match='actual'):
            Comparison.of(pd.Series([1.0], index=pd.Index([1], name='actual')), pd.Series([1.0]))
        with pytest.raises(ValueError, match='duplicate'):
            Comparison.of(pd.Series([1.0, 2.0], index=[1, 1]), pd.Series([1.0]))

    def test_not_equal_by_value(self):
        comparison = Comparison.of(pd.Series([1.0]), pd.Series([1.0]))
        assert comparison == comparison
        assert comparison != Comparison.of(pd.Series([1.0]), pd.Series([1.0]))

    def test_relative_tolerance(self):
        comparison = Comparison.of(pd.Series([100.0, 10.0]), pd.Series([101.0, 11.0]), rel_tol=0.02)
        assert comparison.violations['actual'].tolist() == [10.0]

    def test_relative_tolerance_is_symmetric(self):
        # |50 - 100| = 50 fits 0.5 * max(50, 100) = 50; a share of |actual| alone (25) would not.
        assert Comparison.of(pd.Series([50.0]), pd.Series([100.0]), rel_tol=0.5).violations.empty
        assert Comparison.of(pd.Series([100.0]), pd.Series([50.0]), rel_tol=0.5).violations.empty
        assert len(Comparison.of(pd.Series([40.0]), pd.Series([100.0]), rel_tol=0.5).violations) == 1

    def test_empty_is_off(self):
        actual = pd.Series([np.nan, 1.0])
        expected = pd.Series([1.0, np.nan])
        assert Comparison.of(actual, expected).compared == 0
        off = Comparison.of(actual, expected, empty_is_off=True)
        assert off.compared == 1
        assert len(off.violations) == 1

    def test_empty_is_off_with_frames(self):
        index = pd.Index([1, 2], name='time')
        actual = pd.DataFrame({'L1': [np.nan, 1.0], 'L2': [1.0, np.nan]}, index=index)
        expected = pd.DataFrame({'L1': [1.0, 1.0], 'L2': [1.0, np.nan]}, index=index)
        off = Comparison.of(actual, expected, empty_is_off=True)
        assert off.compared == 3
        assert off.violations[['time', 'column']].to_dict('records') == [{'time': 1, 'column': 'L1'}]

    def test_combine(self):
        bad = Comparison.of(pd.Series([1.0]), pd.Series([2.0]))
        good = Comparison.of(pd.Series([1.0]), pd.Series([1.0]))
        combined = Comparison.combine({'a': bad, 'b': good}, 'part')
        assert combined.compared == 2
        assert combined.violations.columns[0] == 'part'
        assert combined.violations['part'].tolist() == ['a']
        assert Comparison.combine({'b': good}, 'part').violations.empty
        with pytest.raises(ValueError, match='actual'):
            Comparison.combine({'a': bad}, 'actual')


class TestDatasetCheck:
    def test_run_and_name(self, dataset):
        result = FlowCount().run(dataset)
        assert (result.check, result.status) == ('FlowCount', CheckStatus.INFO)
        assert result.summary == '120 flow values.'

    def test_skip_for_unavailable_flag(self, without_demand):
        result = DemandCount().run(without_demand)
        assert result.status is CheckStatus.SKIP
        assert DEMAND in result.summary and FLOW not in result.summary
        assert result.details['flag'].tolist() == [DEMAND]
        assert 'demand_load' in result.details['reason'].iloc[0]

    def test_unaccepted_flag_raises(self, dataset):
        with pytest.raises(UnknownFlagError, match='flow_sent_upp'):
            Typo().run(dataset)

    def test_unknown_availability_counts_as_available(self, dataset, monkeypatch):
        unknown = Availability(None, 'only a fetch tells')
        monkeypatch.setattr(type(dataset), 'availability', lambda self, flag: unknown)
        assert FlowCount().run(dataset).status is CheckStatus.INFO
        assert DatasetCheck.has_flags(dataset, [FLOW])

    def test_unavailable_fetch_gives_skip(self, without_demand):
        assert FetchesUnavailable().run(without_demand).status is CheckStatus.SKIP

    def test_has_flags(self, dataset, without_demand):
        assert DatasetCheck.has_flags(dataset, [FLOW, DEMAND])
        assert not DatasetCheck.has_flags(without_demand, [FLOW, DEMAND])
        with pytest.raises(UnknownFlagError):
            DatasetCheck.has_flags(dataset, ['Line.Results.flow_sent_upp'])

    def test_helpers(self):
        check = FlowCount()
        details = pd.DataFrame({'x': [1]})
        assert check.result(CheckStatus.FAIL, 's', details).details is details
        assert check.skip('why').summary == 'why'
        assert check.result(CheckStatus.FAIL, 's').details.empty

    def test_compared(self, dataset):
        failed = Tolerance().run(dataset)
        assert failed.status is CheckStatus.FAIL
        assert failed.summary == '120 of 120 line MTUs off.'
        assert len(failed.details) == 120
        assert Tolerance(tol=1).run(dataset).status is CheckStatus.PASS
        empty = Comparison(0, pd.DataFrame())
        assert Tolerance().compared(empty, 'x').status is CheckStatus.SKIP


class TestConstraintCheck:
    def test_pass(self, dataset):
        result = ConstraintCheck(FLOW, min_value=0, max_value=1000, isna_ok=False).run(dataset)
        assert result.status is CheckStatus.PASS
        assert result.check == f'ConstraintCheck[{FLOW}]'
        assert result.summary == f'120 values of {FLOW} meet the rule >= 0, <= 1000, not empty.'

    def test_fail_lists_violations(self, dataset):
        result = ConstraintCheck(FLOW, max_value=600, object_subset=['L3', 'L4']).run(dataset)
        assert result.status is CheckStatus.FAIL
        assert result.summary.startswith('24 of 48 values')
        assert set(result.details['Line']) == {'L4'}
        assert (result.details['value'] == 650).all()

    def test_equal_bounds(self, dataset):
        result = ConstraintCheck(FLOW, min_value=500, max_value=500, object_subset=['L1']).run(dataset)
        assert result.status is CheckStatus.PASS
        assert result.summary.endswith('meet the rule = 500, empty allowed.')
        assert ConstraintCheck(FLOW, min_value=500, max_value=500).run(dataset).status is CheckStatus.FAIL

    def test_object_subset(self, dataset):
        with pytest.raises(ValueError, match='empty'):
            ConstraintCheck(FLOW, min_value=0, object_subset=[])
        with pytest.raises(KeyError):
            ConstraintCheck(FLOW, min_value=0, object_subset=['L99']).run(dataset)

    def test_object_subset_on_series_raises(self, dataset, monkeypatch):
        series = pd.Series([1.0], index=pd.Index(['L1'], name='Line'))
        monkeypatch.setattr(type(dataset), 'fetch', lambda self, flag, **kwargs: series)
        with pytest.raises(ValueError, match='Series'):
            ConstraintCheck(FLOW, min_value=0, object_subset=['L1']).run(dataset)
        assert ConstraintCheck(FLOW, min_value=0).run(dataset).status is CheckStatus.PASS

    def test_no_values_gives_skip(self, dataset, monkeypatch):
        empty = pd.DataFrame(index=pd.Index([], name='time'), columns=pd.Index([], name='Line'), dtype=float)
        monkeypatch.setattr(type(dataset), 'fetch', lambda self, flag, **kwargs: empty)
        assert ConstraintCheck(FLOW, min_value=0).run(dataset).status is CheckStatus.SKIP

    def test_empty_values(self):
        gaps = DummyDataset(CONSTANT.replace(name='gaps', nan_gaps=(('2024-01-01 00:00', '2024-01-01 02:00'),)))
        assert ConstraintCheck(FLOW, min_value=0).run(gaps).status is CheckStatus.PASS
        result = ConstraintCheck(FLOW, min_value=0, isna_ok=False).run(gaps)
        assert result.status is CheckStatus.FAIL
        assert result.details['value'].isna().all()

    def test_name_and_skip(self, without_demand):
        check = ConstraintCheck(DEMAND, min_value=0, name='demand not negative')
        result = check.run(without_demand)
        assert (result.check, result.status) == ('demand not negative', CheckStatus.SKIP)


class TestConstraintCheckBounds:
    @pytest.fixture
    def flows(self, dataset):
        return dataset.fetch(FLOW)

    def test_per_object_series(self, dataset):
        limits = pd.Series({'L1': 500.0, 'L2': 500.0, 'L3': 600.0, 'L4': 700.0, 'L5': 700.0}).rename_axis('Line')
        result = ConstraintCheck(FLOW, max_value=limits).run(dataset)
        assert result.status is CheckStatus.FAIL
        assert set(result.details['Line']) == {'L2'}
        assert result.details.columns.tolist() == ['time', 'Line', 'value', 'max_value']
        assert (result.details['max_value'] == 500).all()
        assert 'break the rule <= max_value per object, empty allowed.' in result.summary

    def test_per_timestep_series(self, dataset, flows):
        limits = pd.Series(700.0, index=flows.index)
        limits.iloc[0] = 600.0
        result = ConstraintCheck(FLOW, min_value=0, max_value=limits).run(dataset)
        assert result.status is CheckStatus.FAIL
        assert result.details['time'].nunique() == 1
        assert set(result.details['Line']) == {'L4', 'L5'}
        assert result.details.columns.tolist() == ['time', 'Line', 'value', 'min_value', 'max_value']
        assert '>= 0, <= max_value per timestep' in result.summary

    def test_dataframe_bound(self, dataset, flows):
        limits = flows + 1
        limits.iloc[3, 2] = 0
        result = ConstraintCheck(FLOW, max_value=limits).run(dataset)
        assert len(result.details) == 1
        assert result.details.loc[0, 'Line'] == 'L3'
        assert '<= max_value per value' in result.summary

    def test_series_flag_with_series_bound(self, dataset, monkeypatch):
        series = pd.Series([1.0, 5.0], index=pd.Index(['L1', 'L2'], name='Line'))
        monkeypatch.setattr(type(dataset), 'fetch', lambda self, flag, **kwargs: series)
        result = ConstraintCheck(FLOW, max_value=pd.Series({'L1': 2.0, 'L2': 2.0})).run(dataset)
        assert result.details[['Line', 'value', 'max_value']].to_dict('records') == [
            {'Line': 'L2', 'value': 5.0, 'max_value': 2.0}
        ]

    def test_missing_label_and_nan_bound_are_not_tested(self, dataset):
        limits = pd.Series({'L1': 0.0, 'L2': np.nan}).rename_axis('Line')
        result = ConstraintCheck(FLOW, max_value=limits).run(dataset)
        assert set(result.details['Line']) == {'L1'}

    def test_decision_by_labels(self, dataset):
        limits = pd.Series({'L1': 0.0})
        assert set(ConstraintCheck(FLOW, max_value=limits).run(dataset).details['Line']) == {'L1'}

    def test_name_decides_before_labels(self, dataset, flows):
        # The labels are neither all lines nor all timestamps; the index name says per timestep.
        limits = pd.Series([0.0, 0.0], index=pd.Index([flows.index[0], 'L1'], name='time'))
        result = ConstraintCheck(FLOW, max_value=limits).run(dataset)
        assert result.details['time'].nunique() == 1
        assert len(result.details) == 5

    def test_ambiguous_or_unmatched_series_raises(self, dataset):
        with pytest.raises(ValueError, match='per object'):
            ConstraintCheck(FLOW, max_value=pd.Series({'X': 0.0})).run(dataset)
        report = DatasetValidator([ConstraintCheck(FLOW, max_value=pd.Series({'X': 0.0}))]).run(dataset)
        assert report.errors

    def test_ambiguous_labels_raise(self, monkeypatch, dataset):
        square = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=['a', 'b'], columns=['a', 'b'])
        monkeypatch.setattr(type(dataset), 'fetch', lambda self, flag, **kwargs: square)
        with pytest.raises(ValueError, match='not both'):
            ConstraintCheck(FLOW, max_value=pd.Series({'a': 0.0})).run(dataset)


class TestConstraintCheckEqual:
    def test_scalar(self, dataset):
        check = ConstraintCheck.equal(FLOW, 500, object_subset=['L1'], name='flow 500', isna_ok=False)
        assert (check.min_value, check.max_value, check.name, check.isna_ok) == (500, 500, 'flow 500', False)
        result = check.run(dataset)
        assert result.status is CheckStatus.PASS
        assert result.summary.endswith('meet the rule = 500, not empty.')

    def test_per_object_series(self, dataset):
        values = pd.Series({'L1': 500.0, 'L2': 500.0, 'L3': 600.0, 'L4': 650.0, 'L5': 700.0}).rename_axis('Line')
        result = ConstraintCheck.equal(FLOW, values).run(dataset)
        assert result.status is CheckStatus.FAIL
        assert set(result.details['Line']) == {'L2'}
        assert 'break the rule = value per object, empty allowed.' in result.summary


class TestConstraintCheckBoundErrors:
    @staticmethod
    def patch_fetch(monkeypatch, dataset, data):
        monkeypatch.setattr(type(dataset), 'fetch', lambda self, flag, **kwargs: data)

    def test_multiindex_columns(self, dataset, monkeypatch):
        columns = pd.MultiIndex.from_tuples([('L1', 'up'), ('L1', 'down'), ('L2', 'up')], names=['line', 'direction'])
        data = pd.DataFrame([[1.0, 2.0, 3.0]] * 2, index=pd.Index([1, 2], name='time'), columns=columns)
        self.patch_fetch(monkeypatch, dataset, data)
        bound = pd.Series({('L1', 'down'): 1.5, ('L2', 'up'): 5.0})
        result = ConstraintCheck(FLOW, max_value=bound).run(dataset)
        assert result.details[['line', 'direction']].drop_duplicates().to_dict('records') == [
            {'line': 'L1', 'direction': 'down'}
        ]
        with pytest.raises(ValueError, match='one level of a MultiIndex'):
            ConstraintCheck(FLOW, max_value=pd.Series({'L1': 0.0}).rename_axis('line')).run(dataset)

    def test_multiindex_index(self, dataset, monkeypatch):
        index = pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1)], names=['scenario', 'time'])
        data = pd.DataFrame({'L1': [1.0, 2.0, 3.0], 'L2': [1.0, 1.0, 1.0]}, index=index)
        self.patch_fetch(monkeypatch, dataset, data)
        bound = pd.Series({('a', 2): 1.5, ('b', 1): 1.5})
        result = ConstraintCheck(FLOW, max_value=bound).run(dataset)
        assert result.details[['scenario', 'time', 'column']].to_dict('records') == [
            {'scenario': 'a', 'time': 2, 'column': 'L1'},
            {'scenario': 'b', 'time': 1, 'column': 'L1'},
        ]
        with pytest.raises(ValueError, match='one level of a MultiIndex'):
            ConstraintCheck(FLOW, max_value=pd.Series({1: 0.0}).rename_axis('time')).run(dataset)

    def test_tz_mismatch_raises(self):
        aware = DummyDataset(CONSTANT.replace(name='aware', tz='Europe/Berlin'))
        naive_times = aware.fetch(FLOW).index.tz_localize(None)
        with pytest.raises(ValueError, match='covers no value'):
            ConstraintCheck(FLOW, max_value=pd.Series(0.0, index=naive_times)).run(aware)
        with pytest.raises(ValueError):
            ConstraintCheck(FLOW, max_value=pd.Series(0.0, index=pd.Index(naive_times, name='other'))).run(aware)

    def test_bound_without_cover_raises(self, dataset):
        with pytest.raises(ValueError, match='covers no value'):
            ConstraintCheck(FLOW, max_value=pd.Series({'X1': 0.0}).rename_axis('Line')).run(dataset)
        with pytest.raises(ValueError, match='covers no value'):
            ConstraintCheck(FLOW, max_value=pd.DataFrame({'X1': [0.0]})).run(dataset)

    def test_extra_labels_and_object_subset(self, dataset):
        bound = pd.Series({'L1': 0.0, 'L2': 0.0, 'X9': 0.0})
        result = ConstraintCheck(FLOW, max_value=bound, object_subset=['L2', 'L3']).run(dataset)
        assert set(result.details['Line']) == {'L2'}

    def test_empty_and_duplicate_bounds_raise(self, dataset):
        with pytest.raises(ValueError, match='empty'):
            ConstraintCheck(FLOW, max_value=pd.Series(dtype=float)).run(dataset)
        with pytest.raises(ValueError, match='duplicate'):
            ConstraintCheck(FLOW, max_value=pd.Series([0.0, 1.0], index=['L1', 'L1'])).run(dataset)

    def test_reserved_and_repeated_level_names_raise(self, dataset, monkeypatch):
        data = pd.DataFrame({'L1': [1.0]}, index=pd.Index([1], name='value'))
        self.patch_fetch(monkeypatch, dataset, data)
        with pytest.raises(ValueError, match='value'):
            ConstraintCheck(FLOW, max_value=0).run(dataset)
        data = pd.DataFrame({'L1': [1.0]}, index=pd.Index([1], name='x')).rename_axis(columns='x')
        self.patch_fetch(monkeypatch, dataset, data)
        with pytest.raises(ValueError, match='same name'):
            ConstraintCheck(FLOW, max_value=0).run(dataset)

    def test_nan_and_bool_scalar_bounds_raise(self):
        with pytest.raises(ValueError, match='NaN'):
            ConstraintCheck(FLOW, min_value=np.nan)
        with pytest.raises(ValueError, match='bool'):
            ConstraintCheck(FLOW, max_value=True)
        with pytest.raises(ValueError, match='bool'):
            ConstraintCheck(FLOW, max_value=np.bool_(False))

    def test_dataframe_bound_on_series_flag_raises(self, dataset, monkeypatch):
        self.patch_fetch(monkeypatch, dataset, pd.Series([1.0], index=pd.Index(['L1'], name='Line')))
        with pytest.raises(ValueError, match='DataFrame bound'):
            ConstraintCheck(FLOW, max_value=pd.DataFrame({'L1': [0.0]})).run(dataset)

    def test_scalar_min_with_series_max(self, dataset):
        limits = pd.Series({'L1': 400.0}).rename_axis('Line')
        result = ConstraintCheck(FLOW, min_value=600, max_value=limits).run(dataset)
        assert set(result.details['Line']) == {'L1', 'L2'}
        assert 'break the rule >= 600, <= max_value per object' in result.summary
        assert result.details.loc[result.details['Line'] == 'L2', 'max_value'].isna().all()

    def test_isna_ok_false_with_nan_bound(self):
        gaps = DummyDataset(CONSTANT.replace(name='gaps', nan_gaps=(('2024-01-01 00:00', '2024-01-01 01:00'),)))
        limits = pd.Series({'L1': np.nan, 'L2': 1000.0}).rename_axis('Line')
        result = ConstraintCheck(FLOW, max_value=limits, isna_ok=False).run(gaps)
        assert len(result.details) == 5
        assert result.details['value'].isna().all()


class TestDatasetValidator:
    def test_run_returns_report_in_order(self, dataset):
        report = DatasetValidator([FlowCount(), ConstraintCheck(FLOW, min_value=0)]).run(dataset)
        assert isinstance(report, ValidationReport)
        assert report.dataset == 'constant'
        assert [r.check for r in report.results] == ['FlowCount', f'ConstraintCheck[{FLOW}]']

    def test_error_is_captured_and_others_run(self, dataset, caplog):
        with caplog.at_level(logging.WARNING, logger='mesqual.validation.validator'):
            report = DatasetValidator([Raises(), FlowCount()]).run(dataset)
        assert report['Raises'].status is CheckStatus.ERROR
        assert report['Raises'].summary == 'ZeroDivisionError: division by zero'
        assert report['FlowCount'].status is CheckStatus.INFO
        assert 'Raises' in caplog.text and 'ZeroDivisionError' in caplog.text

    def test_unaccepted_flag_and_failing_hook_give_error(self, dataset):
        report = DatasetValidator([Typo(), UnmetRaises()]).run(dataset)
        assert report['Typo'].status is CheckStatus.ERROR
        assert report['Typo'].summary.startswith('UnknownFlagError: ')
        assert report['UnmetRaises'].summary == 'RuntimeError: broken hook'

    def test_duplicate_name_raises(self):
        with pytest.raises(ValueError, match='FlowCount'):
            DatasetValidator([FlowCount(), FlowCount()])
        validator = DatasetValidator([ConstraintCheck(FLOW, min_value=0)])
        with pytest.raises(ValueError):
            validator.add(ConstraintCheck(FLOW, max_value=1))
        validator.add(ConstraintCheck(FLOW, max_value=1, name='upper'))
        assert len(validator.checks) == 2

    def test_replace_in_place(self, dataset):
        validator = DatasetValidator([Tolerance(), FlowCount()])
        validator.add(Tolerance(tol=1), replace=True)
        assert [c.name for c in validator.checks] == ['Tolerance', 'FlowCount']
        assert validator.run(dataset)['Tolerance'].status is CheckStatus.PASS
        validator.add(Raises(), replace=True)
        assert [c.name for c in validator.checks] == ['Tolerance', 'FlowCount', 'Raises']

    def test_remove(self):
        validator = DatasetValidator([FlowCount(), Raises()])
        validator.remove('FlowCount')
        assert [c.name for c in validator.checks] == ['Raises']
        with pytest.raises(KeyError):
            validator.remove('FlowCount')

    def test_rejects_other_types(self):
        with pytest.raises(TypeError):
            DatasetValidator([object()])


class TestValidationReport:
    @pytest.fixture
    def report(self):
        details = pd.DataFrame({'line': [f'L{i}' for i in range(5)], 'note': ['a|b', 'x\ny', None, 1.5, 2]})
        return ValidationReport(
            'base',
            (
                CheckResult('ok', CheckStatus.PASS, 'fine'),
                CheckResult('bad', CheckStatus.FAIL, '5 rows off', details),
                CheckResult('boom', CheckStatus.ERROR, 'KeyError: x'),
                CheckResult('later', CheckStatus.SKIP, 'no flag'),
            ),
        )

    def test_lookup_and_status_lists(self, report):
        assert report['bad'].summary == '5 rows off'
        with pytest.raises(KeyError):
            report['missing']
        assert [r.check for r in report.failed] == ['bad']
        assert [r.check for r in report.errors] == ['boom']
        assert not report.passed
        assert ValidationReport('x', (report['ok'], report['later'])).passed

    def test_to_frame(self, report):
        frame = report.to_frame()
        assert frame.columns.tolist() == ['check', 'status', 'summary', 'details']
        assert frame['status'].tolist() == ['PASS', 'FAIL', 'ERROR', 'SKIP']
        assert frame['details'].tolist() == [0, 5, 0, 0]

    def test_to_markdown(self, report, tmp_path):
        path = tmp_path / 'report.md'
        text = report.to_markdown(path, max_rows=2)
        assert path.read_text() == text
        assert text.startswith('# Validation of base\n\n| check | status | summary |\n')
        assert '| bad | FAIL | 5 rows off |' in text
        assert '## bad\n\nFAIL: 5 rows off\n' in text
        assert '| L0 | a\\|b |' in text and '| L1 | x y |' in text
        assert 'L2' not in text
        assert '3 more rows.' in text
        assert '## ok' not in text

    def test_to_markdown_keeps_details_index(self):
        details = pd.DataFrame({'value': [1.0]}, index=pd.Index(['L1'], name='line'))
        text = ValidationReport('x', (CheckResult('c', CheckStatus.INFO, 's', details),)).to_markdown()
        assert '| line | value |' in text and '| L1 | 1.0 |' in text

    def test_to_markdown_array_cells_and_header_pipes(self):
        details = pd.DataFrame({'a|b': [np.array([1, 2])], 'c': [np.nan]})
        text = ValidationReport('x', (CheckResult('c', CheckStatus.INFO, 's', details),)).to_markdown()
        assert '| a\\|b | c |' in text
        assert '| [1 2] |  |' in text
