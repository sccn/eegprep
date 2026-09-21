"""Second-level statistics for EEGPrep-owned LIMO-compatible models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from eegprep.functions.studyfunc._limo_io import save_limo_result
from eegprep.functions.studyfunc.std_readfilelimo import std_readfilelimo


def std_limoresults(
    source: Any,
    analysis: str = "one sample t-test",
    *,
    contrast: Any = None,
    parameter: int | None = None,
    data2: Any = None,
    regressors: Any = None,
    groups: Any = None,
    covariates: Any = None,
    estimator: str = "mean",
    variances: Any = None,
    alpha: float = 0.05,
    nboot: int = 0,
    tfce: int = 0,
    outputfile: str | Path | None = None,
) -> dict[str, Any]:
    """Compute contrasts, group tests, regressions, or central tendencies.

    Statistical tests operate across the first (subject) axis. ``source`` may
    contain arrays, in-memory first-level model dictionaries, or paths created
    by :func:`pop_limo`. Parameter indices follow EEGLAB's 1-based convention.
    """
    if nboot or tfce:
        raise NotImplementedError("LIMO bootstrap and TFCE correction are not implemented in EEGPrep")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be between zero and one")
    name = _analysis_name(analysis)
    if name == "load":
        return std_readfilelimo(source)
    if name == "contrast":
        result = _contrasts(source, contrast=contrast, parameter=parameter)
    elif name == "one sample t-test":
        values = _subject_values(source, contrast=contrast, parameter=parameter)
        result = _one_sample(values)
    elif name == "paired t-test":
        first = _subject_values(source, contrast=contrast, parameter=parameter)
        second = _subject_values(data2, contrast=contrast, parameter=parameter)
        result = _paired(first, second)
    elif name == "two samples t-test":
        first = _subject_values(source, contrast=contrast, parameter=parameter)
        second = _subject_values(data2, contrast=contrast, parameter=parameter)
        result = _two_sample(first, second)
    elif name in {"n-ways anova", "one-way anova"}:
        result = _one_way_anova(source)
    elif name == "ancova":
        result = _ancova(_array(source), groups, covariates)
    elif name == "repeated measures anova":
        result = _repeated_measures(_array(source))
    elif name == "regression":
        result = _regression(_array(source), regressors)
    elif name == "central tendency":
        result = _central_tendency(_array(source), estimator=estimator, variances=variances, alpha=float(alpha))
    else:
        raise ValueError(f"Unknown LIMO analysis: {analysis!r}")
    result["kind"] = "second_level"
    result["analysis"] = name
    result["alpha"] = float(alpha)
    if outputfile is not None:
        result["file"] = str(save_limo_result(result, outputfile))
    return result


def _analysis_name(value: str) -> str:
    text = " ".join(str(value).lower().replace("_", " ").replace("-", " ").split())
    aliases = {
        "one sample": "one sample t-test",
        "one sample t test": "one sample t-test",
        "paired": "paired t-test",
        "paired t test": "paired t-test",
        "two sample": "two samples t-test",
        "two samples t test": "two samples t-test",
        "two sample t test": "two samples t-test",
        "anova": "one-way anova",
        "n ways anova": "n-ways anova",
        "repeated measures": "repeated measures anova",
    }
    return aliases.get(text, text)


def _contrasts(source: Any, *, contrast: Any, parameter: int | None) -> dict[str, Any]:
    models = _models(source)
    if not models:
        raise ValueError("contrast analysis requires one or more first-level models")
    estimates = []
    contrast_matrix = _contrast_matrix(contrast, parameter, models[0])
    names = list(models[0].get("parameter_names") or [])
    for model in models:
        betas = np.asarray(model.get("betas"), dtype=float)
        if betas.ndim < 1 or betas.shape[-1] != contrast_matrix.shape[1]:
            raise ValueError("all first-level beta arrays must share the contrast parameter axis")
        if list(model.get("parameter_names") or []) != names:
            raise ValueError("all first-level models must share parameter_names")
        estimates.append(np.einsum("...p,cp->c...", betas, contrast_matrix))
    return {
        "contrast": contrast_matrix,
        "parameter_names": names,
        "estimates": np.stack(estimates, axis=0),
        "dataset_indices": np.asarray([int(model.get("dataset_index", 0)) for model in models], dtype=int),
    }


def _contrast_matrix(contrast: Any, parameter: int | None, model: dict[str, Any]) -> np.ndarray:
    count = np.asarray(model.get("betas")).shape[-1]
    if parameter is not None:
        index = int(parameter)
        if index < 1 or index > count:
            raise ValueError(f"parameter must be 1-based and within 1..{count}")
        matrix = np.zeros((1, count), dtype=float)
        matrix[0, index - 1] = 1.0
        return matrix
    if contrast is None:
        raise ValueError("specify a contrast or a 1-based parameter index")
    matrix = np.asarray(contrast, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[1] != count:
        raise ValueError(f"contrast must have {count} columns")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("contrast must contain finite values")
    return matrix


def _subject_values(source: Any, *, contrast: Any, parameter: int | None) -> np.ndarray:
    if _looks_like_models(source):
        estimates = _contrasts(source, contrast=contrast, parameter=parameter)["estimates"]
        return estimates[:, 0] if estimates.shape[1] == 1 else estimates
    if isinstance(source, dict) and "estimates" in source:
        estimates = np.asarray(source["estimates"], dtype=float)
        return estimates[:, 0] if estimates.ndim > 1 and estimates.shape[1] == 1 else estimates
    return _array(source)


def _one_sample(values: np.ndarray) -> dict[str, Any]:
    if values.shape[0] < 2:
        raise ValueError("one-sample inference requires at least two subjects")
    count = np.sum(np.isfinite(values), axis=0)
    mean = np.nanmean(values, axis=0)
    standard_error = np.nanstd(values, axis=0, ddof=1) / np.sqrt(count)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = np.divide(mean, standard_error, out=np.zeros_like(mean), where=standard_error > 0)
    exact = (standard_error == 0) & (mean != 0)
    t_values[exact] = np.copysign(np.inf, mean[exact])
    p_values = 2.0 * stats.t.sf(np.abs(t_values), count - 1)
    return {
        "mean": mean,
        "t": t_values,
        "p": p_values,
        "df": count - 1,
        "n": count,
    }


def _paired(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    if first.shape != second.shape:
        raise ValueError("paired samples must have identical shapes")
    result = _one_sample(first - second)
    result["difference"] = result.pop("mean")
    return result


def _two_sample(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    if first.shape[1:] != second.shape[1:]:
        raise ValueError("the two samples must share feature dimensions")
    if first.shape[0] < 2 or second.shape[0] < 2:
        raise ValueError("two-sample inference requires at least two subjects per group")
    test = stats.ttest_ind(first, second, axis=0, equal_var=False, nan_policy="omit")
    var1 = np.nanvar(first, axis=0, ddof=1)
    var2 = np.nanvar(second, axis=0, ddof=1)
    n1 = np.sum(np.isfinite(first), axis=0)
    n2 = np.sum(np.isfinite(second), axis=0)
    term1 = var1 / n1
    term2 = var2 / n2
    with np.errstate(divide="ignore", invalid="ignore"):
        df = (term1 + term2) ** 2 / (term1**2 / (n1 - 1) + term2**2 / (n2 - 1))
    return {
        "difference": np.nanmean(first, axis=0) - np.nanmean(second, axis=0),
        "t": np.asarray(test.statistic),
        "p": np.asarray(test.pvalue),
        "df": df,
        "n1": n1,
        "n2": n2,
    }


def _regression(values: np.ndarray, regressors: Any) -> dict[str, Any]:
    predictors = np.asarray(regressors, dtype=float)
    if predictors.ndim == 1:
        predictors = predictors[:, None]
    if predictors.ndim != 2 or predictors.shape[0] != values.shape[0]:
        raise ValueError("regressors must be a subject-by-predictor matrix")
    if not np.all(np.isfinite(predictors)) or not np.all(np.isfinite(values)):
        raise ValueError("regression requires finite data and regressors")
    design = np.column_stack((predictors, np.ones(predictors.shape[0], dtype=float)))
    rank = int(np.linalg.matrix_rank(design))
    if design.shape[0] <= rank:
        raise ValueError("regression needs more subjects than independent design columns")
    shape = values.shape[1:]
    response = values.reshape(values.shape[0], -1)
    inverse = np.linalg.pinv(design.T @ design)
    beta = np.linalg.pinv(design) @ response
    residual = response - design @ beta
    df = design.shape[0] - rank
    sigma2 = np.sum(residual**2, axis=0) / df
    stderr = np.sqrt(np.maximum(inverse.diagonal()[:, None] * sigma2[None, :], 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = np.divide(beta, stderr, out=np.zeros_like(beta), where=stderr > 0)
    exact = (stderr == 0) & (beta != 0)
    t_values[exact] = np.copysign(np.inf, beta[exact])
    p_values = 2.0 * stats.t.sf(np.abs(t_values), df)
    total = np.sum((response - np.mean(response, axis=0, keepdims=True)) ** 2, axis=0)
    r2 = np.divide(total - np.sum(residual**2, axis=0), total, out=np.zeros_like(total), where=total > 0)
    return {
        "design": design,
        "betas": np.moveaxis(beta.reshape((beta.shape[0], *shape)), 0, -1),
        "stderr": np.moveaxis(stderr.reshape((stderr.shape[0], *shape)), 0, -1),
        "t": np.moveaxis(t_values.reshape((t_values.shape[0], *shape)), 0, -1),
        "p": np.moveaxis(p_values.reshape((p_values.shape[0], *shape)), 0, -1),
        "r2": r2.reshape(shape),
        "df": df,
    }


def _one_way_anova(source: Any) -> dict[str, Any]:
    if not isinstance(source, (list, tuple)):
        raise TypeError("one-way ANOVA source must be a sequence of group arrays")
    samples = [_array(item) for item in source]
    if len(samples) < 2 or any(sample.shape[0] < 2 for sample in samples):
        raise ValueError("one-way ANOVA requires at least two groups with two subjects each")
    if len({sample.shape[1:] for sample in samples}) != 1:
        raise ValueError("all ANOVA groups must share feature dimensions")
    test = stats.f_oneway(*samples, axis=0, nan_policy="omit")
    return {
        "f": np.asarray(test.statistic),
        "p": np.asarray(test.pvalue),
        "df_between": len(samples) - 1,
        "df_within": sum(np.sum(np.isfinite(sample), axis=0) for sample in samples) - len(samples),
        "group_means": np.stack([np.nanmean(sample, axis=0) for sample in samples]),
    }


def _ancova(values: np.ndarray, groups: Any, covariates: Any) -> dict[str, Any]:
    labels = np.asarray(groups)
    covars = np.asarray(covariates, dtype=float)
    if covars.ndim == 1:
        covars = covars[:, None]
    if labels.ndim != 1 or labels.size != values.shape[0] or covars.shape[0] != values.shape[0]:
        raise ValueError("groups and covariates must have one row per subject")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(covars)):
        raise ValueError("ANCOVA requires finite data and covariates")
    levels = list(dict.fromkeys(labels.tolist()))
    if len(levels) < 2:
        raise ValueError("ANCOVA requires at least two groups")
    dummy = np.column_stack([labels == level for level in levels[1:]]).astype(float)
    full = np.column_stack((covars, dummy, np.ones(values.shape[0], dtype=float)))
    reduced = np.column_stack((covars, np.ones(values.shape[0], dtype=float)))
    response = values.reshape(values.shape[0], -1)
    full_residual = response - full @ (np.linalg.pinv(full) @ response)
    reduced_residual = response - reduced @ (np.linalg.pinv(reduced) @ response)
    full_sse = np.sum(full_residual**2, axis=0)
    reduced_sse = np.sum(reduced_residual**2, axis=0)
    df_group = len(levels) - 1
    df_error = values.shape[0] - np.linalg.matrix_rank(full)
    if df_error <= 0:
        raise ValueError("ANCOVA needs more subjects than independent design columns")
    with np.errstate(divide="ignore", invalid="ignore"):
        f_values = ((reduced_sse - full_sse) / df_group) / (full_sse / df_error)
    f_values = np.maximum(f_values, 0.0).reshape(values.shape[1:])
    return {
        "f": f_values,
        "p": stats.f.sf(f_values, df_group, df_error),
        "df_group": df_group,
        "df_error": int(df_error),
        "levels": [str(level) for level in levels],
    }


def _repeated_measures(values: np.ndarray) -> dict[str, Any]:
    if values.ndim < 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("repeated-measures ANOVA requires subject-by-condition data")
    if not np.all(np.isfinite(values)):
        raise ValueError("repeated-measures ANOVA requires a complete finite array")
    n_subjects, n_conditions = values.shape[:2]
    grand = np.mean(values, axis=(0, 1))
    condition_mean = np.mean(values, axis=0)
    subject_mean = np.mean(values, axis=1)
    total = np.sum((values - grand) ** 2, axis=(0, 1))
    condition_ss = n_subjects * np.sum((condition_mean - grand) ** 2, axis=0)
    subject_ss = n_conditions * np.sum((subject_mean - grand) ** 2, axis=0)
    error_ss = np.maximum(total - condition_ss - subject_ss, 0.0)
    df_condition = n_conditions - 1
    df_error = (n_subjects - 1) * df_condition
    with np.errstate(divide="ignore", invalid="ignore"):
        f_values = (condition_ss / df_condition) / (error_ss / df_error)
    return {
        "f": f_values,
        "p": stats.f.sf(f_values, df_condition, df_error),
        "df_condition": df_condition,
        "df_error": df_error,
        "condition_means": condition_mean,
    }


def _central_tendency(values: np.ndarray, *, estimator: str, variances: Any, alpha: float) -> dict[str, Any]:
    if values.shape[0] < 2:
        raise ValueError("central tendency requires at least two subjects")
    name = " ".join(str(estimator).lower().split())
    if name == "mean":
        center = np.nanmean(values, axis=0)
        count = np.sum(np.isfinite(values), axis=0)
        se = stats.sem(values, axis=0, nan_policy="omit")
    elif name == "weighted mean":
        if variances is None:
            raise ValueError("weighted mean requires subject-level variances")
        variance = np.asarray(variances, dtype=float)
        if variance.shape != values.shape or not np.all(np.isfinite(variance)) or np.any(variance <= 0):
            raise ValueError("variances must be finite, positive, and match the data shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("weighted mean requires finite subject values")
        weights = 1.0 / variance
        center = np.sum(weights * values, axis=0) / np.sum(weights, axis=0)
        se = np.sqrt(1.0 / np.sum(weights, axis=0))
        count = np.sum(np.isfinite(values), axis=0)
    else:
        raise ValueError("estimator must be 'mean' or 'weighted mean'")
    critical = stats.t.ppf(1.0 - alpha / 2.0, np.maximum(count - 1, 1))
    return {"estimate": center, "se": se, "ci": np.stack((center - critical * se, center + critical * se)), "n": count}


def _models(source: Any) -> list[dict[str, Any]]:
    if isinstance(source, dict) and "models" in source:
        source = source["models"] or source.get("files") or []
    if isinstance(source, (str, Path, dict)):
        source = [source]
    if not isinstance(source, (list, tuple)):
        return []
    loaded = [std_readfilelimo(item) for item in source]
    if any(not isinstance(item, dict) or item.get("kind") != "first_level" for item in loaded):
        raise ValueError("all contrast inputs must be EEGPrep first-level LIMO models")
    return loaded


def _looks_like_models(source: Any) -> bool:
    if isinstance(source, dict):
        return source.get("kind") == "first_level" or "models" in source
    if isinstance(source, (str, Path)):
        return True
    if isinstance(source, (list, tuple)) and source:
        return isinstance(source[0], (dict, str, Path))
    return False


def _array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        raise ValueError("LIMO second-level data must have a subject axis")
    return array


__all__ = ["std_limoresults"]
