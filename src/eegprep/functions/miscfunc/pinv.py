"""Moore-Penrose pseudoinverse with MATLAB's singular-value cutoff."""

import numpy as np


def matlab_pinv_rcond(matrix: np.ndarray, tol: float | None = None) -> float:
    """Relative cutoff for ``np.linalg.pinv`` matching MATLAB ``pinv``.

    MATLAB's absolute cutoff is ``max(size(A)) * eps(norm(A, 2))``.
    ``np.spacing`` is ``eps(x)``. ``np.linalg.pinv`` drops singular values
    at or below ``rcond`` times the largest singular value, so this returns
    that absolute cutoff divided by the 2-norm. ``tol`` replaces the default
    absolute cutoff. A zero matrix returns 0.

    Args:
        matrix: Real 2-D matrix whose cutoff is required.
        tol: Absolute singular-value cutoff. ``None`` uses MATLAB's default.

    Returns:
        Non-negative ``rcond`` argument for ``np.linalg.pinv``.
    """
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError('pinv requires a 2-D array.')
    largest = float(np.linalg.norm(array, ord=2))
    if largest == 0.0:
        return 0.0
    absolute = float(max(array.shape) * np.spacing(largest)) if tol is None else float(tol)
    return absolute / largest


def pinv(A: np.ndarray, tol: float | None = None) -> np.ndarray:
    """Moore-Penrose pseudoinverse using MATLAB's singular-value cutoff.

    Args:
        A: Matrix with shape ``(m, n)``. Cast to float64.
        tol: Absolute singular-value cutoff, matching MATLAB ``pinv(A, tol)``.
            Singular values at or below ``tol`` are treated as zero. The
            default is ``max(size(A)) * eps(norm(A, 2))``.

    Returns:
        Pseudoinverse with shape ``(n, m)``.
    """
    matrix = np.asarray(A, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError('pinv requires a 2-D array.')
    if matrix.size == 0:
        return np.zeros((matrix.shape[1], matrix.shape[0]), dtype=np.float64)
    return np.linalg.pinv(matrix, rtol=matlab_pinv_rcond(matrix, tol))
