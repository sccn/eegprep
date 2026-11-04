"""Moore-Penrose pseudoinverse with MATLAB's singular-value cutoff."""

import numpy as np


def pinv(A: np.ndarray, tol: float | None = None) -> np.ndarray:
    """Moore-Penrose pseudoinverse using MATLAB's singular-value cutoff.

    The singular-value decomposition is computed once. Singular values at or
    below ``max(size(A)) * eps(norm(A, 2))`` are treated as zero. Non-finite
    input returns an all-NaN matrix of the transposed shape.

    Args:
        A: Matrix with shape ``(m, n)``. Cast to float64.
        tol: Absolute singular-value cutoff, matching MATLAB ``pinv(A, tol)``.
            Replaces the default cutoff. Singular values at or below ``tol``
            are treated as zero.

    Returns:
        Pseudoinverse with shape ``(n, m)``.
    """
    matrix = np.asarray(A, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError('pinv requires a 2-D array.')
    rows, columns = matrix.shape
    if rows == 0 or columns == 0:
        return np.zeros((columns, rows), dtype=np.float64)
    if not np.isfinite(matrix).all():
        return np.full((columns, rows), np.nan, dtype=np.float64)

    left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    cutoff = float(tol) if tol is not None else float(max(rows, columns) * np.spacing(singular_values[0]))
    factors = np.zeros_like(singular_values)
    np.divide(1.0, singular_values, out=factors, where=singular_values > cutoff)
    return (right.T * factors) @ left.T
