"""JADE ICA for real signals, ported from EEGLAB ``jader.m`` (``jadeR``)."""

from __future__ import annotations

import numpy as np


def jader(data, m: int | None = None) -> np.ndarray:
    """Return the separating matrix ``B`` such that ``B @ data`` estimates the sources.

    ``data`` is ``(sensors, samples)``. ``m`` sources are kept (default: every sensor).
    Rows are ordered by decreasing mixing-column norm and signed so the first
    column of ``B`` is non-negative, as in ``jadeR``.
    """
    samples = np.asarray(data, dtype=float)
    if samples.ndim != 2:
        raise ValueError("jader: data must be a sensors-by-samples array.")
    sensors, n_samples = samples.shape
    sources = sensors if m is None else int(m)
    if sources > sensors:
        raise ValueError("jader: do not ask for more sources than sensors.")
    if n_samples < 2:
        raise ValueError("jader: at least two samples are required.")
    centered = samples - samples.mean(axis=1, keepdims=True)
    product = (centered @ centered.T) / n_samples
    lower = np.tril(product)
    covariance = lower + np.tril(lower, -1).T
    eigenvalues, eigenvectors = _matlab_eig(covariance)
    chosen = np.arange(sensors - sources, sensors)
    scales = np.sqrt(np.maximum(eigenvalues[chosen], 0.0))
    basis = eigenvectors[:, chosen]
    whitener = np.diag(1.0 / scales) @ basis.T
    whitened_inverse = basis @ np.diag(scales)
    whitened = whitener @ centered

    cumulant_count = sources * (sources + 1) // 2
    cumulants = np.zeros((sources, sources * cumulant_count))
    identity = np.eye(sources)
    column = 0
    scale = np.ones(sources) / n_samples
    for first in range(sources):
        first_row = whitened[first, :]
        weighted = (scale[:, None] * (first_row * first_row)[None, :]) * whitened
        cumulants[:, column : column + sources] = (
            weighted @ whitened.T - identity - 2.0 * np.outer(identity[:, first], identity[:, first])
        )
        column += sources
        for second in range(first):
            second_row = whitened[second, :]
            weighted = (scale[:, None] * (first_row * second_row)[None, :]) * whitened
            cross = (
                weighted @ whitened.T
                - np.outer(identity[:, first], identity[:, second])
                - np.outer(identity[:, second], identity[:, first])
            )
            cumulants[:, column : column + sources] = np.sqrt(2.0) * cross
            column += sources

    _values, rotation = _matlab_eig(cumulants[:, :sources])
    for start in range(0, sources * cumulant_count, sources):
        cumulants[:, start : start + sources] = cumulants[:, start : start + sources] @ rotation
    cumulants = rotation.T @ cumulants

    threshold = 1.0 / np.sqrt(n_samples) / 100.0
    changed = True
    while changed:
        changed = False
        for first in range(sources - 1):
            for second in range(first + 1, sources):
                first_index = slice(first, None, sources)
                second_index = slice(second, None, sources)
                pair = np.vstack(
                    [
                        cumulants[first, first_index] - cumulants[second, second_index],
                        cumulants[first, second_index] + cumulants[second, first_index],
                    ]
                )
                gram = pair @ pair.T
                diagonal = gram[0, 0] - gram[1, 1]
                off = gram[0, 1] + gram[1, 0]
                theta = 0.5 * np.arctan2(off, diagonal + np.sqrt(diagonal * diagonal + off * off))
                if abs(theta) <= threshold:
                    continue
                changed = True
                cosine = np.cos(theta)
                sine = np.sin(theta)
                givens = np.array([[cosine, -sine], [sine, cosine]])
                rotation[:, [first, second]] = rotation[:, [first, second]] @ givens
                cumulants[[first, second], :] = givens.T @ cumulants[[first, second], :]
                first_columns = cumulants[:, first_index].copy()
                second_columns = cumulants[:, second_index].copy()
                cumulants[:, first_index] = cosine * first_columns + sine * second_columns
                cumulants[:, second_index] = -sine * first_columns + cosine * second_columns

    separating = rotation.T @ whitener
    mixing = whitened_inverse @ rotation
    order = np.argsort(np.sum(mixing * mixing, axis=0))
    separating = separating[order, :][::-1, :]
    signs = np.sign(np.sign(separating[:, 0]) + 0.1)
    return np.diag(signs) @ separating


def _matlab_eig(matrix):
    """Match MATLAB ``eig``: the symmetric solver only when ``A`` equals ``A.'``.

    The first JADE cumulant block is not exactly symmetric, so MATLAB takes the
    general unsorted path. Real parts are kept only when that path's result is real.
    """
    array = np.asarray(matrix)
    if array.shape[0] == array.shape[1] and np.array_equal(array, array.T):
        return np.linalg.eigh(array)
    values, vectors = np.linalg.eig(array)
    if not np.iscomplexobj(values) and not np.iscomplexobj(vectors):
        return values, vectors
    real_values = np.real_if_close(values, tol=1000)
    real_vectors = np.real_if_close(vectors, tol=1000)
    if not np.iscomplexobj(real_values) and not np.iscomplexobj(real_vectors):
        return np.asarray(real_values, dtype=float), np.asarray(real_vectors, dtype=float)
    return values, vectors
