import pytest

from mesqual.flag import FrameSchema, Level


def test_unknown_role_raises():
    with pytest.raises(ValueError, match='unknown role'):
        Level('Node', 'node')


def test_levels_must_be_level_objects():
    with pytest.raises(TypeError):
        FrameSchema(index=('time',))


def test_time_levels_and_text():
    schema = FrameSchema(index=[Level('run', 'dimension'), Level('time', 'time')], columns=[Level('Node', 'object')])
    assert schema.time_levels == (Level('time', 'time'),)
    assert str(schema) == 'index (run: dimension, time: time), columns (Node: object)'
