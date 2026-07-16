"""Parity tests for spher - sphering matrix (EEGLAB spher.m).

``spher(data) = 2 * inv(sqrtm(cov(data.T)))``. The result is symmetric and
whitens the channel covariance so that ``S @ C @ S.T == 4*I`` (the factor is 2).
Single-channel data has the closed form ``2 / sqrt(var(data))``. Expectations
mirror tests/matlab/test_spher.m (confirmed on MATLAB R2025a).
"""

from __future__ import annotations

import numpy as np
import pytest

from eegprep.functions.sigprocfunc.spher import spher

pytestmark = pytest.mark.parity

# Deterministic full-rank 3-channel x 6-sample data (matches the MATLAB test).
DATA = np.array(
    [
        [2.0, 1.0, 0.0, 1.0, 3.0, 2.0],
        [0.0, 2.0, 1.0, 3.0, 1.0, 0.0],
        [1.0, 0.0, 3.0, 1.0, 2.0, 1.0],
    ]
)


def test_single_channel_closed_form():
    # 1 channel: cov(data.T) = var(data) (N-1) = 2.5 -> 2/sqrt(2.5).
    data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    np.testing.assert_allclose(np.squeeze(spher(data)), 2 / np.sqrt(2.5), rtol=1e-10)


def test_symmetric():
    # Sphering matrix of a symmetric PSD covariance is symmetric.
    s = spher(DATA)
    np.testing.assert_allclose(s, s.T, atol=1e-9)


def test_whitens_covariance_to_four_i():
    # S = 2*inv(sqrtm(C)) -> S @ C @ S.T = 4*I.
    s = spher(DATA)
    c = np.cov(DATA)  # rowvar=True, ddof=1 == MATLAB cov(data.T)
    np.testing.assert_allclose(s @ c @ s.T, 4 * np.eye(3), atol=1e-6)
