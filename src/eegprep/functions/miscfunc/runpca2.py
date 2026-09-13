"""Covariance-based PCA and whitening."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.miscfunc.misc import finite_matmul
from eegprep.functions.miscfunc.runpca import _component_count, _pca_matrix


def runpca2(
    data: Any,
    n_components: int | None = None,
    symmetric: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whiten channel-major data from its channel covariance matrix.

    Args:
        data: Channels by observations matrix.
        n_components: Number of covariance eigenvectors to retain.
        symmetric: Use EEGLAB's compact eigenvector form. ``False`` is accepted
            only for a full-rank decomposition and returns a symmetric mixing
            operator.

    Returns:
        Whitened components, mixing matrix, and descending covariance standard
        deviations.
    """
    matrix = _pca_matrix(data)
    centered = matrix - np.mean(matrix, axis=1, keepdims=True)
    channels, frames = centered.shape
    count = _component_count(n_components, channels, channels)
    if not symmetric and count != channels:
        raise ValueError("symmetric=False requires all channel components")

    covariance = finite_matmul(centered, centered.T) / frames
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0, None)
    eigenvectors = eigenvectors[:, order]
    eigenvectors = _canonical_eigenvector_signs(eigenvectors)
    scales = np.sqrt(eigenvalues)

    if symmetric:
        selected_vectors = eigenvectors[:, :count]
        selected_scales = scales[:count]
        inverse_scales = np.divide(
            1.0,
            selected_scales,
            out=np.zeros_like(selected_scales),
            where=selected_scales > _rank_tolerance(selected_scales, covariance.shape[0]),
        )
        components = inverse_scales[:, np.newaxis] * finite_matmul(selected_vectors.T, centered)
        mixing = selected_vectors * selected_scales
        return components, mixing, scales

    inverse_scales = np.divide(
        1.0,
        scales,
        out=np.zeros_like(scales),
        where=scales > _rank_tolerance(scales, covariance.shape[0]),
    )
    whitened = inverse_scales[:, np.newaxis] * finite_matmul(eigenvectors.T, centered)
    components = finite_matmul(eigenvectors, whitened)
    mixing = finite_matmul(finite_matmul(eigenvectors, np.diag(scales)), eigenvectors.T)
    return components, mixing, scales


def _canonical_eigenvector_signs(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.copy()
    for component in range(vectors.shape[1]):
        anchor = int(np.argmax(np.abs(vectors[:, component])))
        if vectors[anchor, component] < 0:
            vectors[:, component] *= -1
    return vectors


def _rank_tolerance(scales: np.ndarray, dimension: int) -> float:
    maximum = float(np.max(scales, initial=0))
    return np.finfo(float).eps * max(1, dimension) * maximum


__all__ = ["runpca2"]
