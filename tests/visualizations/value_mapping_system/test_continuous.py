import numpy as np
import pandas as pd
import pytest

from mesqual.visualizations.value_mapping_system import (
    SegmentedContinuousColorscale,
    SegmentedContinuousLineWidthMapping,
)


@pytest.fixture
def widths() -> SegmentedContinuousLineWidthMapping:
    return SegmentedContinuousLineWidthMapping(
        segments={(0, 10): [1.0, 3.0], (20, 30): 5.0},
        default_output=0.5,
    )


@pytest.mark.parametrize('value', [None, np.nan, pd.NA, pd.NaT])
def test_missing_value_gives_default_output(widths, value):
    assert widths(value) == 0.5


def test_value_between_segments_gives_default_output(widths):
    assert widths(15) == 0.5


def test_value_that_is_not_a_number_gives_default_output(widths):
    assert widths('abc') == 0.5


def test_value_outside_the_range_gets_the_edge_output(widths):
    assert widths(-5) == 1.0
    assert widths(100) == 5.0


def test_value_in_a_segment_is_interpolated(widths):
    assert widths(5) == pytest.approx(2.0)
    assert widths.default_output == 0.5


def test_colorscale_factory_passes_default_output():
    scale = SegmentedContinuousColorscale.single_segment_autoscale_factory_from_array(
        values=[0, 1, np.nan], colorscale=['#000000', '#ffffff'], default_output='#cccccc'
    )
    assert scale(None) == '#cccccc'
    assert scale(1) == '#ffffff'


def test_line_width_factory_passes_default_output():
    mapping = SegmentedContinuousLineWidthMapping.single_segment_autoscale_factory_from_array(
        values=[0, 10], output_range=(1, 5), default_output=0.5
    )
    assert mapping(np.nan) == 0.5
    assert mapping(10) == 5


def test_colormap_name_gives_the_colors_of_the_colormap():
    scale = SegmentedContinuousColorscale(segments={(0, 1): 'viridis'})
    assert scale(0) == '#440154'
    assert scale(1) == '#fde725'


def test_single_color_string_gives_that_color():
    scale = SegmentedContinuousColorscale(segments={(0, 1): 'red'})
    assert scale(0.5) == '#ff0000'


def test_unknown_string_raises():
    with pytest.raises(ValueError, match='neither a valid colormap nor a valid color'):
        SegmentedContinuousColorscale(segments={(0, 1): 'no-such-colormap'})


def test_colorscale_factory_default_is_viridis():
    scale = SegmentedContinuousColorscale.single_segment_autoscale_factory_from_array(values=[0, 10])
    assert scale(0) == '#440154'


def test_colorscale_does_not_change_the_given_segments():
    segments = {(0, 1): 'viridis'}
    SegmentedContinuousColorscale(segments=segments)
    assert segments == {(0, 1): 'viridis'}


@pytest.mark.parametrize('cls', [SegmentedContinuousColorscale, SegmentedContinuousLineWidthMapping])
def test_factory_rejects_unknown_arguments(cls):
    with pytest.raises(TypeError, match='nan_fallback'):
        cls.single_segment_autoscale_factory_from_array(values=[0, 1], nan_fallback=0)


def test_empty_segments_raise():
    with pytest.raises(ValueError, match='at least one segment'):
        SegmentedContinuousLineWidthMapping(segments={})


def test_input_value_range_spans_all_segments(widths):
    assert widths.input_value_range == (0.0, 30.0)


@pytest.mark.parametrize('value, expected_range', [(50.0, (45.0, 55.0)), (-20.0, (-22.0, -18.0)), (0.0, (-1.0, 1.0))])
def test_factory_widens_the_range_of_a_constant_array(value, expected_range):
    mapping = SegmentedContinuousLineWidthMapping.single_segment_autoscale_factory_from_array(
        values=[value, value, np.nan], output_range=(1, 5)
    )
    assert mapping.input_value_range == pytest.approx(expected_range)
    assert mapping(value) == pytest.approx(3.0)


@pytest.mark.parametrize(
    'value, midpoint, expected_range', [(0, 0, (-1.0, 1.0)), (3, 3, (2.7, 3.3)), (5, 0, (-5.0, 5.0))]
)
def test_colorscale_factory_widens_a_constant_array_only_if_it_equals_the_midpoint(value, midpoint, expected_range):
    scale = SegmentedContinuousColorscale.single_segment_autoscale_factory_from_array(
        values=[value, value], colorscale=['#000000', '#ffffff'], continuous_midpoint=midpoint
    )
    assert scale.input_value_range == pytest.approx(expected_range)


@pytest.mark.parametrize(
    'values, trim_percentile',
    [
        ([np.inf, np.inf], None),
        ([0.0, np.inf], None),
        ([-np.inf, 5.0], None),
        ([1.0, 2.0, np.inf], (5, 95)),
        ([1.7e308, 1.7e308], None),
        ([-1e308, 1e308], None),
    ],
)
def test_factory_raises_for_a_range_that_is_not_finite(values, trim_percentile):
    # pytest turns warnings into errors, so this also checks that numpy emits no warning.
    with pytest.raises(ValueError, match='or its width is not finite'):
        SegmentedContinuousLineWidthMapping.single_segment_autoscale_factory_from_array(
            values=values, output_range=(1, 5), trim_percentile=trim_percentile
        )


def test_trim_percentile_cuts_off_an_infinite_value():
    mapping = SegmentedContinuousLineWidthMapping.single_segment_autoscale_factory_from_array(
        values=list(range(100)) + [np.inf], output_range=(1, 5), trim_percentile=(5, 95)
    )
    assert mapping.input_value_range == pytest.approx((5.0, 95.0))


def test_colorscale_factory_widens_a_constant_array_without_midpoint():
    scale = SegmentedContinuousColorscale.single_segment_autoscale_factory_from_array(values=[7, 7])
    assert scale.input_value_range == pytest.approx((6.3, 7.7))


def test_factory_widens_a_trimmed_range_of_one_value():
    mapping = SegmentedContinuousLineWidthMapping.single_segment_autoscale_factory_from_array(
        values=[1, 5, 5, 5, 5, 5, 5, 5, 5, 9], output_range=(1, 5), trim_percentile=(20, 80)
    )
    assert mapping.input_value_range == pytest.approx((4.5, 5.5))


@pytest.mark.parametrize('values', [[], [np.nan, None]])
def test_factory_raises_for_empty_or_all_missing_values(values):
    with pytest.raises(ValueError, match='empty or all missing'):
        SegmentedContinuousColorscale.single_segment_autoscale_factory_from_array(values=values)
