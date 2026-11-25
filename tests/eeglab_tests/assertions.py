"""Structure comparisons used by the original suite's near.m assertions."""

import numpy as np

from tests.eeglab_tests import assert_matlab_equal, assert_matlab_near


def matlab_field_concat(structures, field):
    """Evaluate MATLAB's [structures.field], including horizontal concatenation."""
    if isinstance(structures, dict):
        return structures[field]
    values = list(structures[field].ravel(order="F"))
    if not values:
        return np.empty((0, 0))
    if all(isinstance(value, str) for value in values):
        return "".join(values)
    if all(isinstance(value, dict) for value in values):
        fields = list(values[0])
        return np.array(
            [[tuple(value[field] for field in fields) for value in values]], dtype=[(f, object) for f in fields]
        )
    arrays = [np.atleast_2d(value) for value in values if np.asarray(value).size]
    return np.concatenate(arrays, axis=1) if arrays else np.empty((0, 0))


def assert_matlab_struct_near(first, second, recursionlevel=5):
    """Port near.m's depth-limited numeric/structure/isequal branches exactly."""
    assert recursionlevel >= 1
    first_fields = tuple(first) if isinstance(first, dict) else getattr(getattr(first, "dtype", None), "names", None)
    second_fields = (
        tuple(second) if isinstance(second, dict) else getattr(getattr(second, "dtype", None), "names", None)
    )
    first_empty = False if isinstance(first, dict) else np.asarray(first).size == 0
    second_empty = False if isinstance(second, dict) else np.asarray(second).size == 0
    assert first_empty == second_empty
    if first_fields is not None and second_fields is not None:
        assert set(first_fields) == set(second_fields)
        # near.m initializes result=0 and only changes it inside its field loop.
        assert first_fields
        for field in first_fields:
            assert_matlab_struct_near(
                matlab_field_concat(first, field), matlab_field_concat(second, field), recursionlevel - 1
            )
    elif np.asarray(first).dtype.kind in "iufc" and np.asarray(second).dtype.kind in "iufc":
        assert_matlab_near(second, first)
    else:
        assert_matlab_equal(first, second)
