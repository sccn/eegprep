"""Build LIMO-compatible STUDY design matrices."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.popfunc._pop_utils import is_on, parse_key_value_args
from eegprep.functions.studyfunc._study_utils import build_python_call, equal_value, trialinfo_rows, value_matches
from eegprep.functions.studyfunc.pop_listfactors import pop_listfactors


def std_limodesign(
    factors: Any,
    trialinfo: Any,
    *args: Any,
    splitreg: str | bool = "off",
    interaction: str | bool = "off",
    desconly: str | bool = "off",
    filepath: str = "",
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Create LIMO-compatible categorical and continuous design matrices."""
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    splitreg = options.pop("splitreg", splitreg)
    interaction = options.pop("interaction", interaction)
    desconly = options.pop("desconly", desconly)
    filepath = str(options.pop("filepath", filepath) or "")
    if options:
        raise ValueError(f"Unknown std_limodesign option(s): {', '.join(sorted(options))}")

    rows = trialinfo_rows(trialinfo)
    factor_rows = _factor_rows(factors)
    categorical = _categorical_options(factor_rows)
    continuous = _continuous_options(factor_rows)
    categorical_design = _categorical_design(categorical, interaction=is_on(interaction))
    continuous_design = _continuous_design(continuous, categorical, splitreg=is_on(splitreg))
    limodesign = {"categorical": categorical_design, "continuous": continuous_design}

    if is_on(desconly):
        cat_matrix = np.empty((0, 0), dtype=float)
        cont_matrix = np.empty((0, 0), dtype=float)
    else:
        cat_matrix = _categorical_matrix(rows, categorical_design)
        cont_matrix = _continuous_matrix(rows, continuous_design, splitreg=is_on(splitreg))
        if filepath:
            _save_design_files(Path(filepath), cat_matrix, cont_matrix)
    command = build_python_call(
        ("catMat", "contMat", "limodesign"),
        "std_limodesign",
        "factors",
        "trialinfo",
        splitreg="on" if is_on(splitreg) else "off",
        interaction="on" if is_on(interaction) else "off",
        desconly="on" if is_on(desconly) else "off",
        filepath=filepath,
    )
    result = (cat_matrix, cont_matrix, limodesign)
    return (*result, command) if return_com else result


def _factor_rows(factors: Any) -> list[dict[str, Any]]:
    if isinstance(factors, dict) and ("design" in factors or "variable" in factors):
        factors = pop_listfactors(factors, constant="off")
    if isinstance(factors, np.ndarray):
        factors = factors.tolist()
    if isinstance(factors, dict):
        factors = [factors]
    if not isinstance(factors, list):
        return []
    return [factor for factor in factors if isinstance(factor, dict) and str(factor.get("label") or "") != "constant"]


def _categorical_options(factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for factor in factors:
        if str(factor.get("vartype") or "categorical").lower() != "categorical":
            continue
        label = str(factor.get("label") or "")
        if not label:
            continue
        existing = next((entry for entry in output if entry["label"] == label), None)
        if existing is None:
            existing = {"label": label, "values": []}
            output.append(existing)
        if "value" in factor:
            values = existing.get("values")
            if not isinstance(values, list):
                values = []
                existing["values"] = values
            value = factor.get("value")
            if not any(_equal(value, current) for current in values):
                values.append(value)
    return output


def _continuous_options(factors: list[dict[str, Any]]) -> list[str]:
    output = []
    for factor in factors:
        if str(factor.get("vartype") or "").lower() != "continuous":
            continue
        label = str(factor.get("label") or "")
        if label and label not in output:
            output.append(label)
    return output


def _categorical_design(categorical: list[dict[str, Any]], *, interaction: bool) -> list[list[list[Any]]]:
    options = [[[(entry["label"], value)] for value in entry["values"]] for entry in categorical]
    if interaction and len(options) > 1:
        return [[sum((list(item) for item in combo), []) for combo in product(*options)]]
    return options


def _continuous_design(
    continuous: list[str], categorical: list[dict[str, Any]], *, splitreg: bool
) -> list[list[tuple[str, Any]]]:
    base = [[(label, "")] for label in continuous]
    if not splitreg or not categorical:
        return base
    categorical_options = _categorical_design(categorical, interaction=True)
    if not categorical_options:
        return base
    output = []
    for continuous_option in base:
        for categorical_option in categorical_options[0]:
            output.append([*continuous_option, *categorical_option])
    return output


def _categorical_matrix(rows: list[dict[str, Any]], design: list[list[list[Any]]]) -> np.ndarray:
    if not design:
        return np.empty((len(rows), 0), dtype=float)
    matrix = np.full((len(rows), len(design)), np.nan, dtype=float)
    for column, options in enumerate(design):
        for level_index, option in enumerate(options, start=1):
            hits = _matching_rows(rows, option)
            matrix[hits, column] = float(level_index)
    return matrix


def _continuous_matrix(
    rows: list[dict[str, Any]], design: list[list[tuple[str, Any]]], *, splitreg: bool
) -> np.ndarray:
    if not design:
        return np.empty((len(rows), 0), dtype=float)
    fill = np.nan if splitreg else 0.0
    matrix = np.full((len(rows), len(design)), fill, dtype=float)
    for column, option in enumerate(design):
        continuous_label = str(option[0][0])
        hits = _matching_rows(rows, option[1:])
        for row_index, row in enumerate(rows):
            if not hits[row_index]:
                continue
            value = row.get(continuous_label)
            if value not in (None, ""):
                matrix[row_index, column] = float(value)
        if splitreg:
            finite = np.isfinite(matrix[:, column])
            if np.any(finite):
                values = matrix[finite, column]
                if values.size < 2:
                    matrix[finite, column] = 0.0
                else:
                    std = float(np.nanstd(values, ddof=1))
                    matrix[finite, column] = 0.0 if std == 0 else (values - float(np.nanmean(values))) / std
            matrix[~finite, column] = 0.0
    return matrix


def _matching_rows(rows: list[dict[str, Any]], option: list[tuple[str, Any]]) -> np.ndarray:
    if not option:
        return np.ones(len(rows), dtype=bool)
    hits = np.ones(len(rows), dtype=bool)
    for label, expected in option:
        if expected == "":
            hits &= np.asarray([row.get(label) not in (None, "") for row in rows], dtype=bool)
        else:
            hits &= np.asarray([_value_matches(row.get(label), expected) for row in rows], dtype=bool)
    return hits


def _value_matches(value: Any, expected: Any) -> bool:
    return value_matches(value, expected)


def _equal(left: Any, right: Any) -> bool:
    return equal_value(left, right)


def _save_design_files(path: Path, cat_matrix: np.ndarray, cont_matrix: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if cat_matrix.size:
        np.savetxt(path / "categorical_variables.txt", cat_matrix, fmt="%.12g")
    if cont_matrix.size:
        np.savetxt(path / "continuous_variables.txt", cont_matrix, fmt="%.12g")


__all__ = ["std_limodesign"]
