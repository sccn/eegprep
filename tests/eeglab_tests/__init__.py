"""Shared provenance helpers for ports of the current EEGLAB test suite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Callable, TypeVar
import warnings

import pytest
import numpy as np
from numpy.exceptions import ComplexWarning
from scipy.io import loadmat


EEGLAB_TESTS_REPOSITORY = "https://github.com/sccn/eeglab_tests.git"
EEGLAB_TESTS_COMMIT = "ff605546f3f70868916fb8d49c007472b3257b50"
EEGLAB_TESTS_EEGLAB_COMMIT = "8ac485f654d6bbb1a6acb8dc9ef3f2eaf3d409ba"
STALE_EEGLAB_TESTS_REPOSITORY = "https://github.com/sccn/eeglab-testcases.git"
MATLAB_TEST_EPSILON = 0.0001

_TestCallable = TypeVar("_TestCallable", bound=Callable)
_REFERENCE_ATTRIBUTE = "__eeglab_test_references__"


@dataclass(frozen=True)
class EeglabTestReference:
    """Identify one MATLAB test scenario ported by a Python test."""

    source: str
    test: str
    suite_commit: str = EEGLAB_TESTS_COMMIT
    eeglab_commit: str = EEGLAB_TESTS_EEGLAB_COMMIT


def eeglab_test(source: str, test: str) -> Callable[[_TestCallable], _TestCallable]:
    """Attach current-suite provenance and the parity marker to a pytest test."""
    reference = _validated_reference(source, test)

    def decorate(test_function: _TestCallable) -> _TestCallable:
        references = upstream_references(test_function)
        if reference in references:
            raise ValueError(f"duplicate EEGLAB test reference: {source}::{test}")
        setattr(test_function, _REFERENCE_ATTRIBUTE, (*references, reference))
        return pytest.mark.parity(test_function)

    return decorate


def upstream_references(test_function: Callable) -> tuple[EeglabTestReference, ...]:
    """Return the upstream MATLAB tests represented by a Python test."""
    return tuple(getattr(test_function, _REFERENCE_ATTRIBUTE, ()))


def load_matlab_test_fixture(file: str | Path) -> dict[str, Any]:
    """Load a MATLAB v4-v7.2 fixture, preserving MATLAB classes and shapes."""
    loaded = loadmat(file, struct_as_record=True, squeeze_me=False)
    # MATLAB can store integral-valued doubles using integer MAT storage. The
    # typed read restores their declared class but drops complex imaginary data.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        typed = loadmat(file, struct_as_record=True, squeeze_me=False, mat_dtype=True)
    return {
        key: _fixture_numeric_classes(value, typed[key]) for key, value in loaded.items() if not key.startswith("__")
    }


def _fixture_numeric_classes(raw, typed):
    if raw.dtype.names:
        for index in np.ndindex(raw.shape):
            for field in raw.dtype.names:
                raw[field][index] = _fixture_numeric_classes(raw[field][index], typed[field][index])
        return raw
    if raw.dtype == object:
        for index in np.ndindex(raw.shape):
            raw[index] = _fixture_numeric_classes(raw[index], typed[index])
        return raw
    return raw if np.iscomplexobj(raw) else typed


def assert_matlab_near(actual, expected):
    """Preserve numeric near.m/tc_epsilon.m assertions from the source suite."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, f"{actual.shape} != {expected.shape}"
    within = ((expected - MATLAB_TEST_EPSILON <= actual) & (actual <= expected + MATLAB_TEST_EPSILON)) | (
        np.isnan(actual) & np.isnan(expected)
    )
    assert np.all(within), f"Values differ by more than {MATLAB_TEST_EPSILON}:\n{actual}\n!=\n{expected}"


def _validated_reference(source: str, test: str) -> EeglabTestReference:
    source_path = PurePosixPath(source)
    if source_path.is_absolute() or ".." in source_path.parts or source_path.suffix != ".m":
        raise ValueError(f"EEGLAB test source must be a relative .m path: {source!r}")
    if not test or "::" in test:
        raise ValueError(f"EEGLAB test name must be a non-empty MATLAB test name: {test!r}")
    return EeglabTestReference(source=source_path.as_posix(), test=test)


__all__ = [
    "EEGLAB_TESTS_COMMIT",
    "EEGLAB_TESTS_EEGLAB_COMMIT",
    "EEGLAB_TESTS_REPOSITORY",
    "STALE_EEGLAB_TESTS_REPOSITORY",
    "EeglabTestReference",
    "eeglab_test",
    "load_matlab_test_fixture",
    "upstream_references",
]
