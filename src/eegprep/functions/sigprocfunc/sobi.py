"""Second-order blind identification, ported from EEGLAB ``sobi.m``."""

from __future__ import annotations

import numpy as np


def sobi(data, n: int | None = None, p: int | None = None) -> np.ndarray:
    """Return the mixing matrix whose columns are the component scalp maps.

    ``data`` is ``(channels, samples)`` or ``(channels, samples, trials)``.
    ``n`` is accepted for the EEGLAB signature; the decomposition follows
    ``sobi.m`` and returns one component per channel. ``p`` is the number of
    lagged covariance matrices (default ``min(100, ceil(samples / 3))``).
    """
    samples = np.asarray(data, dtype=float)
    if samples.ndim == 2:
        samples = samples[:, :, None]
    if samples.ndim != 3:
        raise ValueError("sobi: data must be channels by samples, or channels by samples by trials.")
    channels, n_samples, n_trials = samples.shape
    if n_samples < 3:
        raise ValueError("sobi: at least three samples are required to form a lagged covariance.")
    source_count = channels if n is None else int(n)
    if source_count < 1:
        raise ValueError("sobi: the number of sources must be at least 1.")
    lagged = p if p is not None else min(100, int(np.ceil(n_samples / 3)))
    if lagged < 1:
        raise ValueError("sobi: the number of lagged covariances must be at least 1.")

    flattened = samples.reshape(channels, -1, order="F")
    flattened = flattened - flattened.mean(axis=1, keepdims=True)
    samples = flattened.reshape(channels, n_samples, n_trials, order="F")
    _singular, scales, vectors = np.linalg.svd(flattened.T, full_matrices=False)
    whitener = np.linalg.pinv(np.diag(scales)) @ vectors.T
    samples = np.einsum("ij,jkl->ikl", whitener, samples)

    covariances = np.zeros((channels, lagged * channels), dtype=float)
    for lag_index in range(lagged):
        lag = lag_index + 2
        lagged_cov = np.zeros((channels, channels), dtype=float)
        for trial in range(n_trials):
            epoch = samples[:, :, trial]
            lagged_cov += epoch[:, lag - 1 :] @ epoch[:, : n_samples - lag + 1].T / (n_samples - lag + 1) / n_trials
        start = lag_index * channels
        covariances[:, start : start + channels] = np.linalg.norm(lagged_cov, ord="fro") * lagged_cov

    rotation = _joint_diagonalize(covariances, lagged, tolerance=1.0 / np.sqrt(n_samples) / 100.0)
    return np.linalg.pinv(whitener) @ rotation


def _joint_diagonalize(covariances: np.ndarray, lags: int, *, tolerance: float) -> np.ndarray:
    channels = covariances.shape[0]
    rotation = np.eye(channels, dtype=complex)
    matrices = covariances.astype(complex)
    changed = True
    while changed:
        changed = False
        for first in range(channels - 1):
            for second in range(first + 1, channels):
                left = matrices[first, first::channels][:lags] - matrices[second, second::channels][:lags]
                both = matrices[first, second::channels][:lags] + matrices[second, first::channels][:lags]
                difference = matrices[second, first::channels][:lags] - matrices[first, second::channels][:lags]
                givens = np.vstack([left, both, 1j * difference])
                _values, vectors = np.linalg.eigh(np.real(givens @ givens.conj().T))
                angles = vectors[:, -1]
                angles = np.sign(angles[0]) * angles
                cosine = np.sqrt(0.5 + angles[0] / 2.0)
                sine = 0.5 * (angles[1] - 1j * angles[2]) / cosine
                if abs(sine) <= tolerance:
                    continue
                changed = True
                first_columns = matrices[:, first::channels].copy()
                second_columns = matrices[:, second::channels].copy()
                matrices[:, first::channels] = cosine * first_columns + sine * second_columns
                matrices[:, second::channels] = cosine * second_columns - np.conj(sine) * first_columns
                first_row = matrices[first, :].copy()
                second_row = matrices[second, :].copy()
                matrices[first, :] = cosine * first_row + np.conj(sine) * second_row
                matrices[second, :] = cosine * second_row - sine * first_row
                stored = rotation[:, first].copy()
                rotation[:, first] = cosine * rotation[:, first] + sine * rotation[:, second]
                rotation[:, second] = cosine * rotation[:, second] - np.conj(sine) * stored
    return np.real_if_close(rotation, tol=1000).real
