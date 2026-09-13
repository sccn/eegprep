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
