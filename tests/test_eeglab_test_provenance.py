from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from tests.eeglab_tests import (
    EEGLAB_TESTS_COMMIT,
    EEGLAB_TESTS_EEGLAB_COMMIT,
    EEGLAB_TESTS_REPOSITORY,
    STALE_EEGLAB_TESTS_REPOSITORY,
    assert_matlab_near,
    eeglab_test,
    load_matlab_test_fixture,
    upstream_references,
)


def test_current_eeglab_test_suite_is_pinned_without_the_stale_repository() -> None:
    assert EEGLAB_TESTS_REPOSITORY == "https://github.com/sccn/eeglab_tests.git"
    assert EEGLAB_TESTS_COMMIT == "ff605546f3f70868916fb8d49c007472b3257b50"
    assert EEGLAB_TESTS_EEGLAB_COMMIT == "8ac485f654d6bbb1a6acb8dc9ef3f2eaf3d409ba"
    assert STALE_EEGLAB_TESTS_REPOSITORY not in {
        EEGLAB_TESTS_REPOSITORY,
        EEGLAB_TESTS_COMMIT,
        EEGLAB_TESTS_EEGLAB_COMMIT,
    }


def test_eeglab_test_records_multiple_upstream_scenarios() -> None:
    @eeglab_test("regression_tests/t_statcond.m", "test_1")
    @eeglab_test("regression_tests/t_statcond.m", "test_2")
    def translated_test() -> None:
        pass

    assert [(reference.source, reference.test) for reference in upstream_references(translated_test)] == [
        ("regression_tests/t_statcond.m", "test_2"),
        ("regression_tests/t_statcond.m", "test_1"),
    ]
    assert any(marker.name == "parity" for marker in getattr(translated_test, "pytestmark"))


@pytest.mark.parametrize(
    ("source", "test"),
    [
        ("/tmp/t_statcond.m", "test_1"),
        ("../t_statcond.m", "test_1"),
        ("regression_tests/t_statcond.py", "test_1"),
        ("regression_tests/t_statcond.m", ""),
        ("regression_tests/t_statcond.m", "class::test_1"),
    ],
)
def test_eeglab_test_rejects_invalid_provenance(source: str, test: str) -> None:
    with pytest.raises(ValueError):
        eeglab_test(source, test)


def test_eeglab_test_rejects_duplicate_references() -> None:
    decorator = eeglab_test("regression_tests/t_statcond.m", "test_1")

    @decorator
    def translated_test() -> None:
        pass

    with pytest.raises(ValueError, match="duplicate EEGLAB test reference"):
        decorator(translated_test)


def test_load_matlab_test_fixture_preserves_shape_and_dtype(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixture.mat"
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    savemat(fixture_path, {"values": expected})

    loaded = load_matlab_test_fixture(fixture_path)

    assert set(loaded) == {"values"}
    assert loaded["values"].shape == (2, 3)
    assert loaded["values"].dtype == np.float32
    np.testing.assert_array_equal(loaded["values"], expected)


def test_load_native_matlab_fixture_preserves_numeric_classes_and_structs(eeglab_matlab_engine, tmp_path):
    fixture_path = tmp_path / "native_fixture.mat"
    quoted_path = str(fixture_path).replace("'", "''")
    eeglab_matlab_engine.eval(
        "eegprep_fixture.integral_double = double([1 2 3]);"
        "eegprep_fixture.single_values = single([1; 2]);"
        "eegprep_fixture.integer_values = int16([1 -2]);"
        "eegprep_fixture.complex_values = complex(single([1 2]), single([3 -4]));"
        "eegprep_fixture.logical_values = logical([1 0]);"
        "eegprep_fixture.scalar_struct = struct('values', {[1 2]}, 'cells', {{[3; 4], complex(1, 2)}});"
        "eegprep_fixture.struct_array = repmat(eegprep_fixture.scalar_struct, 1, 2);"
        f"save('{quoted_path}', '-struct', 'eegprep_fixture', '-v7');"
        "clear eegprep_fixture;",
        nargout=0,
    )
    loaded = load_matlab_test_fixture(fixture_path)
    assert loaded["integral_double"].dtype == np.float64
    assert loaded["integral_double"].shape == (1, 3)
    assert loaded["single_values"].dtype == np.float32
    assert loaded["single_values"].shape == (2, 1)
    assert loaded["integer_values"].dtype == np.int16
    assert loaded["complex_values"].dtype == np.complex64
    np.testing.assert_array_equal(loaded["complex_values"], [[1 + 3j, 2 - 4j]])
    assert loaded["logical_values"].dtype == np.bool_
    scalar = loaded["scalar_struct"]
    assert scalar.dtype.names == ("values", "cells")
    assert scalar.shape == (1, 1)
    assert scalar["values"][0, 0].dtype == np.float64
    cells = scalar["cells"][0, 0]
    assert cells.dtype == object
    assert cells.shape == (1, 2)
    assert cells[0, 0].dtype == np.float64
    assert cells[0, 0].shape == (2, 1)
    np.testing.assert_array_equal(cells[0, 1], [[1 + 2j]])
    assert loaded["struct_array"].shape == (1, 2)
    assert loaded["struct_array"].dtype.names == scalar.dtype.names


def test_matlab_near_keeps_absolute_tolerance_and_nan_semantics():
    assert_matlab_near([[0.0001, np.nan, np.inf]], [[0, np.nan, np.inf]])
    with pytest.raises(AssertionError):
        assert_matlab_near([[100000.001]], [[100000.0]])
    with pytest.raises(AssertionError):
        assert_matlab_near([[np.nan]], [[0.0]])


def test_matlab_near_rejects_broadcasting_rows_into_columns():
    with pytest.raises(AssertionError, match="\\(2, 1\\) != \\(1, 2\\)"):
        assert_matlab_near([[1], [2]], [[1, 2]])
