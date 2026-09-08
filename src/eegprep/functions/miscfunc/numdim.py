"""EEGLAB-compatible estimate of the effective number of sources (numdim)."""

from __future__ import annotations

from typing import Any

import numpy as np


def numdim(data: Any) -> float:
    """Estimate a lower bound on the number of discrete sources in the data.

    Ports EEGLAB's ``numdim`` (Wackermann's measure): the effective
    dimensionality is the exponential of the Shannon entropy of the normalized
    eigenvalue spectrum of the channel second-order matrix ``data @ data.T / 100``.
    The measure is invariant to overall scaling of the data.

    Args:
        data: 2-D array shaped ``(channels, samples)``.

    Returns:
        The estimated effective number of sources, a real scalar.
    """
    a = np.asarray(data, dtype=float).T  # MATLAB: a = a';
    b = a.T @ a / 100.0  # MATLAB: b = a'*a/100;
    eigenvalues = np.linalg.eigvals(b)  # MATLAB: [v d] = eig(b);
    weights = (eigenvalues / np.sum(eigenvalues)).astype(complex)
    # Complex log mirrors MATLAB's real(exp(...)): tiny/negative eigenvalues
    # from finite precision contribute ~0 instead of producing NaN.
    lambda_ = np.exp(-np.sum(weights * np.log(weights)))
    return float(np.real(lambda_))


__all__ = ["numdim"]
