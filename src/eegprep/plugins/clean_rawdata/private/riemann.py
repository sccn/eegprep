"""Manifold operations used by Riemannian ASR.

EEGLAB's ``asr_process_r`` / ``rasr_nonlinear_eigenspace`` call Manopt for two
operations. Both reduce to dense linear algebra for the calls ASR actually
makes:

* The Karcher mean is only ever taken of two SPD covariances with equal
  weight, which is the affine-invariant geodesic midpoint.
* The nonlinear eigenspace is requested for every component. ``Gr(n, n)`` is a
  single point, so Manopt's trust-region solver returns its initial basis: the
  eigenvectors of ``L + alpha * diag(L \\ 1)``. The squared row norms of the
  orthogonal polar factor used to build that operator are exactly 1.
"""

import numpy as np

from eegprep.functions.miscfunc.misc import finite_matmul
from eegprep.functions.miscfunc.pinv import matlab_pinv_rcond

from .covariance import cov_sqrtm, cov_rsqrtm


def spd_geodesic_midpoint(left, right):
    """Affine-invariant geodesic midpoint of two symmetric positive definite matrices.

    This is the Riemannian center of mass of ``left`` and ``right`` under the
    metric Manopt's ``sympositivedefinitefactory`` uses for
    ``positive_definite_karcher_mean``.
    """
    left = np.real(np.asarray(left, dtype=float))
    right = np.real(np.asarray(right, dtype=float))
    left = 0.5 * (left + left.T)
    right = 0.5 * (right + right.T)
    left_sqrt = cov_sqrtm(left)
    left_inv_sqrt = cov_rsqrtm(left)
    inner = finite_matmul(finite_matmul(left_inv_sqrt, right), left_inv_sqrt)
    inner = 0.5 * (inner + inner.T)
    midpoint = finite_matmul(finite_matmul(left_sqrt, cov_sqrtm(inner)), left_sqrt)
    return 0.5 * (midpoint + midpoint.T)


def rasr_nonlinear_eigenspace(operator, k, alpha=1.0):
    """Eigenvectors and diagonal eigenvalues for Riemannian ASR.

    Args:
        operator: Square matrix, the covariance or mixing matrix ``L``.
        k: Number of eigenvectors to return. ASR passes ``k = n``.
        alpha: Scalar in the nonlinear eigenproblem. ASR leaves this at 1.

    Returns:
        tuple: ``(vectors, values)`` where ``values`` is a diagonal matrix, matching
        ``[Xsol, S0] = rasr_nonlinear_eigenspace(L, k)``.
    """
    matrix = np.real(np.asarray(operator, dtype=float))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('rasr_nonlinear_eigenspace requires a square matrix.')
    dimension = matrix.shape[0]
    component_count = int(k)
    if component_count < 1 or component_count > dimension:
        raise ValueError(f'k must be between 1 and {dimension}.')
    matrix = 0.5 * (matrix + matrix.T)
    row_energy = np.ones(dimension) if component_count == dimension else _polar_row_energy(dimension, component_count)
    solved = _solve_spd(matrix, row_energy)
    shifted = matrix + float(alpha) * np.diag(np.real(solved))
    shifted = 0.5 * (shifted + shifted.T)
    eigenvalues, eigenvectors = np.linalg.eigh(shifted)
    if component_count < dimension:
        chosen = np.argsort(np.abs(eigenvalues), kind='mergesort')[-component_count:]
        eigenvalues = eigenvalues[chosen]
        eigenvectors = eigenvectors[:, chosen]
    return np.real(eigenvectors), np.diag(np.real(eigenvalues))


def asr_subspace_reconstruction(mixing, basis, keep):
    """Reconstruction matrix ``M * pinv(keep .* (V' * M)) * V'``.

    Column sign flips of ``basis`` cancel, matching the eigenvector-sign
    freedom in ``eig`` / ``eigs``.
    """
    keep_mask = np.asarray(keep, dtype=bool)[np.newaxis, :]
    projected = finite_matmul(np.asarray(basis).T, mixing)
    masked = np.asarray(keep_mask * projected.T, dtype=np.float64)
    inverse = np.linalg.pinv(masked.T, rtol=matlab_pinv_rcond(masked.T))
    return np.real(finite_matmul(finite_matmul(mixing, inverse), np.asarray(basis).T))


def _solve_spd(matrix, rhs):
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def _polar_row_energy(dimension, component_count):
    """Squared row norms of a polar factor with ``component_count < dimension``.

    ASR does not take this branch: it requests every component, and those row
    norms are identically 1. The seed is local; it is not MATLAB ``rng(0)``.
    """
    raw = np.random.default_rng(0).standard_normal((dimension, component_count))
    left, _, right = np.linalg.svd(raw, full_matrices=False)
    polar = left @ right
    return np.sum(polar**2, axis=1)
