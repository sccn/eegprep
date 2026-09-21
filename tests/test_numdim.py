"""Parity tests for numdim - effective number of sources (EEGLAB numdim.m).

numdim estimates a lower bound on the number of discrete sources via the
eigenvalue entropy of the channel second-order matrix ``A @ A.T / 100``. The
measure is scale-invariant (eigenvalues are normalized), so expectations below
are closed-form / property-based.

Expected values match the MATLAB reference tests in tests/matlab/test_numdim.m
(confirmed on MATLAB R2025a). Regenerate ground truth with
``matlab -batch "addpath('.../functions/miscfunc'); numdim(<input>)"``.
"""

from __future__ import annotations

import numpy as np
import pytest

from eegprep.functions.miscfunc.numdim import numdim
from tests.eeglab_tests import eeglab_test

pytestmark = pytest.mark.parity


def test_single_channel_is_one():
    # One channel -> one normalized eigenvalue -> entropy 0 -> lambda == 1.
    rng = np.random.default_rng(0)
    a = rng.random((1, 50)) + 0.5
    np.testing.assert_allclose(numdim(a), 1.0, atol=1e-10)


def test_two_channel_orthogonal_is_nchan():
    # A @ A.T = 2*I -> equal eigenvalues -> lambda == nchan == 2.
    a = np.array([[1.0, 1.0], [1.0, -1.0]])
    np.testing.assert_allclose(numdim(a), 2.0, atol=1e-10)


def test_hadamard_equal_energy_is_nchan():
    # Hadamard(4): orthogonal equal-norm rows -> A @ A.T = 4*I -> lambda == 4.
    h4 = np.array(
        [
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(numdim(h4), 4.0, atol=1e-9)


def test_rank_deficient_is_approx_one():
    # Identical channels -> rank-1 -> one dominant eigenvalue -> ~1 effective dim.
    a = np.ones((3, 10))
    np.testing.assert_allclose(numdim(a), 1.0, atol=1e-6)


def test_full_rank_matches_matlab():
    # Full-rank deterministic integer matrix: identical in numpy and MATLAB, so
    # pin to the exact MATLAB R2025a value (confirmed via matlab -batch). The
    # result is real and strictly between 1 and nchan (=3).
    a = np.array(
        [
            [2.0, 1.0, 0.0, 1.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
        ]
    )
    v = numdim(a)
    assert np.isreal(v)
    np.testing.assert_allclose(v, 2.147746217856, rtol=1e-6)


@eeglab_test(
    "unittesting_miscfunc/numdim/miscfunc_numdim_wrapperTest.m",
    "test_pass_general",
)
def test_current_eeglab_suite_matches_closed_form_entropy():
    data = np.array([[2.0, 1.0], [-1.0, -2.0]])
    expected = 1 / (0.1**0.1) / (0.9**0.9)

    np.testing.assert_allclose(numdim(data), expected, rtol=1e-12)


@eeglab_test(
    "unittesting_miscfunc/numdim/miscfunc_numdim_wrapperTest.m",
    "test_pass_row_vector",
)
def test_current_eeglab_suite_row_vector_is_one_dimensional():
    assert numdim(np.array([[1.0, 2.0, 0.0, -6.0]])) == pytest.approx(1.0)


@eeglab_test(
    "unittesting_miscfunc/numdim/miscfunc_numdim_wrapperTest.m",
    "test_pass_column_vector",
)
def test_current_eeglab_suite_preserves_column_vector_degeneracy():
    # The upstream test allows NaN or 1 because numdim.m evaluates 0*log(0)
    # literally. The analytic entropy limit is one effective source.
    result = numdim(np.array([[1.0], [2.0], [0.0], [-6.0]]))

    assert result == pytest.approx(1.0)


@eeglab_test(
    "unittesting_miscfunc/numdim/miscfunc_numdim_wrapperTest.m",
    "test_pass_singular_matrix",
)
def test_current_eeglab_suite_singular_matrix_scenario_is_executable():
    # The upstream assertion is commented out, but its intended result and
    # the analytic entropy of a rank-one matrix are both one.
    result = numdim(np.array([[2.0, 2.0], [-1.0, -1.0]]))

    assert result == pytest.approx(1.0)
