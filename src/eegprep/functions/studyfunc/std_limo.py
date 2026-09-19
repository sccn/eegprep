"""Mass-univariate first-level models for LIMO-compatible workflows."""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
from scipy import stats


LIMO_METHODS = {"OLS", "WLS", "IRLS"}
_TUKEY_TUNING = 4.685


def std_limo(
    data: Any,
    design: Any,
    *,
    method: str = "OLS",
    parameter_names: list[str] | None = None,
    times: Any = None,
) -> dict[str, Any]:
    """Fit a first-level mass-univariate linear model.

    Args:
        data: EEG observations with trials on the final axis. Channel-by-time-
            by-trial arrays are the usual input, but any feature dimensions are
            accepted.
        design: Two-dimensional trial-by-parameter design matrix.
        method: ``"OLS"``, ``"WLS"``, or Tukey-bisquare ``"IRLS"``.
        parameter_names: Optional names for the columns of ``design``.
        times: Optional time coordinates retained in the result.

    Returns:
        A dictionary containing beta estimates, fitted values, residuals,
        inferential statistics, the design matrix, and robust weights.
    """
    values = np.asarray(data, dtype=float)
    matrix = np.asarray(design, dtype=float)
    if values.ndim < 2:
        raise ValueError("data must have one or more feature dimensions and a final trial axis")
    if matrix.ndim != 2:
        raise ValueError("design must be a two-dimensional trial-by-parameter matrix")
    if values.shape[-1] != matrix.shape[0]:
        raise ValueError("the data trial axis must match the number of design rows")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("design must contain only finite values")
    method_name = str(method).upper()
    if method_name not in LIMO_METHODS:
        raise ValueError("method must be 'OLS', 'WLS', or 'IRLS'")
    rank = int(np.linalg.matrix_rank(matrix))
    if matrix.shape[0] <= rank:
        raise ValueError("the first-level model needs more trials than independent design columns")
    names = parameter_names or [f"parameter_{index}" for index in range(1, matrix.shape[1] + 1)]
    if len(names) != matrix.shape[1]:
        raise ValueError("parameter_names must have one entry per design column")

    feature_shape = values.shape[:-1]
    response = np.moveaxis(values, -1, 0).reshape(matrix.shape[0], -1)
    if not np.all(np.isfinite(response)):
        raise ValueError("std_limo requires finite first-level data")
    if method_name == "OLS":
        fit = _fit_global_weights(matrix, response, np.ones(matrix.shape[0], dtype=float))
        reduction = 0
    elif method_name == "WLS":
        initial = _fit_global_weights(matrix, response, np.ones(matrix.shape[0], dtype=float))
        weights, reduction = _observation_weights(initial["residuals"], matrix)
        fit = _fit_global_weights(matrix, response, weights)
    else:
        fit = _fit_irls(matrix, response)
        reduction = 0

    result = {
        "kind": "first_level",
        "method": method_name,
        "design": matrix,
        "parameter_names": list(names),
        "rank": rank,
        "df_residual": int(matrix.shape[0] - rank),
        "betas": _reshape_parameters(fit["betas"], feature_shape),
        "fitted": _reshape_trials(fit["fitted"], feature_shape),
        "residuals": _reshape_trials(fit["residuals"], feature_shape),
        "r2": np.asarray(fit["r2"]).reshape(feature_shape),
        "sigma2": np.asarray(fit["sigma2"]).reshape(feature_shape),
        "stderr": _reshape_parameters(fit["stderr"], feature_shape),
        "t": _reshape_parameters(fit["t"], feature_shape),
        "p": _reshape_parameters(fit["p"], feature_shape),
        "weights": _reshape_weights(fit["weights"], feature_shape),
        "weight_reduction": reduction,
        "converged": bool(fit["converged"]),
    }
    if times is not None:
        coordinates = np.asarray(times, dtype=float).ravel()
        if values.ndim < 3 or coordinates.size != values.shape[-2]:
            raise ValueError("times must match the penultimate data axis")
        result["times"] = coordinates
    return result


def _fit_global_weights(design: np.ndarray, response: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    row_scale = weights[:, None]
    weighted_design = design * row_scale
    betas = np.linalg.pinv(weighted_design) @ (response * row_scale)
    fitted = np.einsum("np,pf->nf", design, betas)
    residuals = response - fitted
    rank = int(np.linalg.matrix_rank(weighted_design))
    df = design.shape[0] - rank
    weighted_sse = np.sum(row_scale**2 * residuals**2, axis=0)
    sigma2 = weighted_sse / df
    covariance = np.linalg.pinv(weighted_design.T @ weighted_design)
    stderr = np.sqrt(np.maximum(covariance.diagonal()[:, None] * sigma2[None, :], 0.0))
    fit = _finish_fit(response, betas, fitted, residuals, sigma2, stderr, weights[:, None], df)
    fit["converged"] = True
    return fit


def _fit_irls(design: np.ndarray, response: np.ndarray) -> dict[str, Any]:
    rank = int(np.linalg.matrix_rank(design))
    df = design.shape[0] - rank
    betas = np.linalg.pinv(design) @ response
    inverse = np.linalg.pinv(np.einsum("np,nq->pq", design, design))
    projected = np.einsum("np,pq->nq", design, inverse)
    leverage = np.einsum("np,np->n", projected, design)
    adjustment = 1.0 / np.sqrt(np.maximum(1.0 - leverage, np.finfo(float).eps))
    adjustment[~np.isfinite(adjustment)] = 1.0
    old_error = 1.0
    new_error = 10.0
    weights = np.ones_like(response)
    converged = False
    for _iteration in range(100):
        if abs(old_error - new_error) <= 1e-4:
            converged = True
            break
        old_error = new_error
        residuals = response - np.einsum("np,pf->nf", design, betas)
        adjusted = residuals * adjustment[:, None]
        scale = np.maximum(np.median(np.abs(adjusted), axis=0) / 0.6745, 1e-5)
        standardized = adjusted / (_TUKEY_TUNING * scale[None, :])
        weights = np.sqrt(_tukey_weights(standardized))
        for feature in range(response.shape[1]):
            weighted_design = design * weights[:, feature, None]
            betas[:, feature] = np.linalg.pinv(weighted_design) @ (response[:, feature] * weights[:, feature])
        new_error = float(np.sum(residuals**2))
    if not converged:
        warnings.warn("LIMO IRLS did not converge after 100 iterations", RuntimeWarning, stacklevel=2)

    fitted = np.einsum("np,pf->nf", design, betas)
    residuals = response - fitted
    sigma2 = np.empty(response.shape[1], dtype=float)
    stderr = np.empty_like(betas)
    for feature in range(response.shape[1]):
        weighted_design = design * weights[:, feature, None]
        variance = float(np.sum((weights[:, feature] * residuals[:, feature]) ** 2) / df)
        covariance = np.linalg.pinv(weighted_design.T @ weighted_design)
        sigma2[feature] = variance
        stderr[:, feature] = np.sqrt(np.maximum(covariance.diagonal() * variance, 0.0))
    fit = _finish_fit(response, betas, fitted, residuals, sigma2, stderr, weights, df)
    fit["converged"] = converged
    return fit


def _finish_fit(
    response: np.ndarray,
    betas: np.ndarray,
    fitted: np.ndarray,
    residuals: np.ndarray,
    sigma2: np.ndarray,
    stderr: np.ndarray,
    weights: np.ndarray,
    df: int,
) -> dict[str, Any]:
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = np.divide(betas, stderr, out=np.zeros_like(betas), where=stderr > 0)
    exact = (stderr == 0) & (betas != 0)
    t_values[exact] = np.copysign(np.inf, betas[exact])
    p_values = 2.0 * stats.t.sf(np.abs(t_values), df)
    centered = response - np.mean(response, axis=0, keepdims=True)
    total = np.sum(centered**2, axis=0)
    error = np.sum(residuals**2, axis=0)
    r2 = np.divide(total - error, total, out=np.zeros_like(total), where=total > 0)
    return {
        "betas": betas,
        "fitted": fitted,
        "residuals": residuals,
        "sigma2": sigma2,
        "stderr": stderr,
        "t": t_values,
        "p": p_values,
        "r2": r2,
        "weights": weights,
    }


def _observation_weights(residuals: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, int]:
    inverse = np.linalg.pinv(np.einsum("np,nq->pq", design, design))
    projected = np.einsum("np,pq->nq", design, inverse)
    leverage = np.clip(np.einsum("np,np->n", projected, design), 0.0, 1.0)
    adjusted = residuals / np.sqrt(np.maximum(1.0 - leverage[:, None], np.finfo(float).eps))
    scales = np.maximum(np.median(np.abs(adjusted), axis=0) / 0.6745, 1e-5)
    standardized = adjusted / (_TUKEY_TUNING * scales[None, :])
    weights, reduction = _pcout_weights(standardized)
    if np.count_nonzero(weights) <= np.linalg.matrix_rank(design):
        return np.ones_like(weights), reduction
    return weights, reduction


def _pcout_weights(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Return LIMO PCOut location/scatter weights without its plotting path."""
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median[None, :]), axis=0)
    retained = mad > 1e-6
    if not np.any(retained):
        raise ValueError("WLS cannot be computed because all adjusted residual dimensions are constant")
    data = values[:, retained]
    median = np.median(data, axis=0)
    mad = np.median(np.abs(data - median[None, :]), axis=0) * 1.4826
    robust = (data - median[None, :]) / mad[None, :]
    centered = robust - np.mean(robust, axis=0, keepdims=True)
    _left, singular, right = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2 / max(1, data.shape[0] - 1)
    cumulative = np.cumsum(variance) / np.sum(variance)
    above = np.flatnonzero(cumulative > 0.99)
    components = int(above[0] + 1) if above.size else singular.size
    reduction = int(data.shape[1] - components)
    projected = robust @ right[:components].T
    projected_median = np.median(projected, axis=0)
    projected_mad = np.median(np.abs(projected - projected_median[None, :]), axis=0) * 1.4826
    projected_mad = np.maximum(projected_mad, np.finfo(float).eps)
    scaled = (projected - projected_median[None, :]) / projected_mad[None, :]

    kurtosis = np.abs(np.mean(scaled**4, axis=0) - 3.0)
    if float(np.sum(kurtosis)) <= np.finfo(float).eps:
        kurtosis = np.ones_like(kurtosis)
    location_norm = np.sqrt(np.sum((scaled * (kurtosis / np.sum(kurtosis))[None, :]) ** 2, axis=1))
    chi_median = float(np.sqrt(stats.chi2.ppf(0.5, components)))
    location_distance = location_norm * chi_median / max(float(np.median(location_norm)), np.finfo(float).eps)
    location_weight = _translated_biweight(location_distance)

    scatter_norm = np.sqrt(np.sum(scaled**2, axis=1))
    scatter_distance = scatter_norm * chi_median / max(float(np.median(scatter_norm)), np.finfo(float).eps)
    lower = float(np.sqrt(stats.chi2.ppf(0.25, components)))
    upper = float(np.sqrt(stats.chi2.ppf(0.99, components)))
    scatter_weight = np.square(1.0 - np.square((scatter_distance - lower) / (upper - lower)))
    scatter_weight[scatter_distance < lower] = 1.0
    scatter_weight[scatter_distance > upper] = 0.0
    offset = 0.25
    return (location_weight + offset) * (scatter_weight + offset) / (1.0 + offset) ** 2, reduction


def _translated_biweight(distance: np.ndarray) -> np.ndarray:
    lower = _matlab_quantile(distance, 1.0 / 3.0)
    mad = float(np.median(np.abs(distance - np.median(distance))))
    upper = float(np.median(distance) + 2.5 * mad * 1.4826)
    if upper <= lower:
        return np.ones_like(distance)
    weights = np.square(1.0 - np.square((distance - lower) / (upper - lower)))
    weights[distance < lower] = 1.0
    weights[distance > upper] = 0.0
    return weights


def _matlab_quantile(values: np.ndarray, probability: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float).ravel())
    position = ordered.size * probability + 0.5
    if position <= 1.0:
        return float(ordered[0])
    if position >= ordered.size:
        return float(ordered[-1])
    lower = int(np.floor(position))
    fraction = position - lower
    return float(ordered[lower - 1] * (1.0 - fraction) + ordered[lower] * fraction)


def _tukey_weights(standardized: np.ndarray) -> np.ndarray:
    scaled = np.asarray(standardized, dtype=float)
    weights = np.zeros_like(scaled)
    inside = np.abs(scaled) < 1.0
    weights[inside] = (1.0 - scaled[inside] ** 2) ** 2
    return weights


def _reshape_parameters(values: np.ndarray, feature_shape: tuple[int, ...]) -> np.ndarray:
    return np.moveaxis(values.reshape((values.shape[0], *feature_shape)), 0, -1)


def _reshape_trials(values: np.ndarray, feature_shape: tuple[int, ...]) -> np.ndarray:
    return np.moveaxis(values.reshape((values.shape[0], *feature_shape)), 0, -1)


def _reshape_weights(values: np.ndarray, feature_shape: tuple[int, ...]) -> np.ndarray:
    if values.shape[1] == 1:
        return values[:, 0]
    return np.moveaxis(values.reshape((values.shape[0], *feature_shape)), 0, -1)


__all__ = ["LIMO_METHODS", "std_limo"]
