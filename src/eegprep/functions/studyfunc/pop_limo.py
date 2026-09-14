"""Run first-level LIMO-compatible models for a STUDY."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
import re
from typing import Any

import numpy as np

from eegprep.functions.popfunc._pop_utils import is_on, parse_key_value_args
from eegprep.functions.popfunc.plot_utils import component_activations
from eegprep.functions.studyfunc._limo_io import save_limo_result
from eegprep.functions.studyfunc._study_utils import (
    build_python_call,
    ensure_study,
    trialinfo_rows,
    value_matches,
)
from eegprep.functions.studyfunc.std_checkset import std_checkset
from eegprep.functions.studyfunc.std_limo import std_limo
from eegprep.functions.studyfunc.std_maketrialinfo import std_maketrialinfo


def pop_limo(
    STUDY: dict[str, Any],
    ALLEEG: list[dict[str, Any]] | None,
    *args: Any,
    method: str = "OLS",
    measure: str = "daterp",
    timelim: Any = None,
    splitreg: str | bool = "off",
    interaction: str | bool = "off",
    erase: str | bool = "on",
    outputdir: str | Path | None = None,
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Fit first-level models for every selected STUDY dataset.

    The active ``STUDY.design`` supplies categorical levels, continuous
    regressors, and subject selection. Results are returned in memory and can
    optionally be written as safe, versioned ``.npz`` files.
    """
    mode, option_args = _mode_and_args(args)
    options = parse_key_value_args(option_args, kwargs, lowercase_kwargs=True)
    method = str(options.pop("method", method)).upper()
    measure = str(options.pop("measure", measure)).lower()
    timelim = options.pop("timelim", timelim)
    splitreg = options.pop("splitreg", splitreg)
    interaction = options.pop("interaction", interaction)
    erase = options.pop("erase", erase)
    outputdir = options.pop("outputdir", outputdir)
    nboot = int(options.pop("nboot", 0) or 0)
    tfce = int(options.pop("tfce", 0) or 0)
    if options:
        raise ValueError(f"Unknown pop_limo option(s): {', '.join(sorted(options))}")
    if measure not in {"daterp", "erp"}:
        raise NotImplementedError("pop_limo currently fits epoched time-domain data only (measure='daterp')")
    if nboot or tfce:
        raise NotImplementedError("first-level LIMO bootstrap and TFCE inference are not implemented in EEGPrep")

    study = ensure_study(STUDY)
    _active_design(study)
    study, datasets = std_checkset(study, ALLEEG)
    study, generated_trialinfo = std_maketrialinfo(study, datasets)
    design = _active_design(study)
    selected_subjects = {str(value) for value in design.get("cases", {}).get("value", [])}
    destination = Path(outputdir).expanduser().resolve() if outputdir else None
    models = []
    paths = []
    dataset_indices = []
    for dataset_index, (eeg, fallback_rows) in enumerate(zip(datasets, generated_trialinfo), start=1):
        info = study["datasetinfo"][dataset_index - 1]
        subject = str(info.get("subject") or eeg.get("subject") or f"S{dataset_index}")
        if selected_subjects and subject not in selected_subjects:
            continue
        data, times = _model_data(eeg, mode, timelim)
        rows = trialinfo_rows(info.get("trialinfo")) or fallback_rows
        rows = _enrich_rows(rows, info, data.shape[-1])
        matrix, names, keep = _design_matrix(
            design,
            rows,
            splitreg=is_on(splitreg),
            interaction=is_on(interaction),
        )
        model = std_limo(data[..., keep], matrix, method=method, parameter_names=names, times=times)
        model["dataset_index"] = dataset_index
        model["subject"] = subject
        model["measure"] = "daterp"
        models.append(model)
        dataset_indices.append(dataset_index)
        if destination is not None:
            path = destination / f"{_safe_name(subject)}_{dataset_index}_limo_{method.lower()}.npz"
            if path.exists() and not is_on(erase):
                raise FileExistsError(f"LIMO model already exists: {path}; use erase='on' to replace it")
            paths.append(str(save_limo_result(model, path)))
    if not models:
        raise ValueError("the active STUDY design selected no datasets")

    model_files = {
        "models": models,
        "files": paths,
        "mat": paths,
        "Beta": paths,
        "dataset_indices": dataset_indices,
        "method": method,
        "measure": "daterp",
    }
    study["limo"] = {
        "design": int(study.get("currentdesign") or 1),
        "method": method,
        "measure": "daterp",
        "model_files": paths,
        "dataset_indices": dataset_indices,
        "chanloc": deepcopy(datasets[dataset_indices[0] - 1].get("chanlocs") or []),
        "unsupported": ["MATLAB .mat interchange", "bootstrap", "TFCE", "LIMO plotting"],
    }
    study["saved"] = "no"
    command = build_python_call(
        ("STUDY", "ALLEEG", "model_files"),
        "pop_limo",
        "STUDY",
        "ALLEEG",
        method=method,
        measure="daterp",
        timelim=timelim,
        splitreg="on" if is_on(splitreg) else "off",
        interaction="on" if is_on(interaction) else "off",
        erase="on" if is_on(erase) else "off",
        outputdir=str(destination) if destination else None,
    )
    result = (study, datasets, model_files)
    return (*result, command) if return_com else result


def _mode_and_args(args: tuple[Any, ...]) -> tuple[str, tuple[Any, ...]]:
    if args and isinstance(args[0], str) and args[0].lower() in {"dat", "data", "channels", "ica", "components"}:
        return args[0].lower(), args[1:]
    return "dat", args


def _active_design(study: dict[str, Any]) -> dict[str, Any]:
    designs = study.get("design") or []
    index = int(study.get("currentdesign") or 1)
    if index < 1 or index > len(designs):
        raise ValueError("pop_limo requires a valid active STUDY design")
    design = designs[index - 1]
    if not isinstance(design, dict):
        raise ValueError("the active STUDY design is invalid")
    return design


def _model_data(eeg: dict[str, Any], mode: str, timelim: Any) -> tuple[np.ndarray, np.ndarray]:
    data = component_activations(eeg) if mode in {"ica", "components"} else np.asarray(eeg.get("data"), dtype=float)
    if data.ndim != 3 or int(eeg.get("trials", data.shape[-1]) or 1) <= 1:
        raise ValueError("pop_limo requires epoched channel-by-time-by-trial data")
    times = np.asarray(eeg.get("times", []), dtype=float).ravel()
    if times.size != data.shape[1]:
        srate = float(eeg.get("srate", 1.0) or 1.0)
        xmin = float(eeg.get("xmin", 0.0) or 0.0)
        times = (np.arange(data.shape[1], dtype=float) / srate + xmin) * 1000.0
    if timelim is None:
        return data, times
    bounds = np.asarray(timelim, dtype=float).ravel()
    if bounds.size != 2 or bounds[0] > bounds[1]:
        raise ValueError("timelim must contain increasing [start, stop] milliseconds")
    keep = (times >= bounds[0]) & (times <= bounds[1])
    if not np.any(keep):
        raise ValueError("timelim does not overlap the EEG time axis")
    return data[:, keep, :], times[keep]


def _enrich_rows(rows: list[dict[str, Any]], info: dict[str, Any], trials: int) -> list[dict[str, Any]]:
    if len(rows) != trials:
        rows = [{} for _index in range(trials)]
    constants = {
        key: value
        for key, value in info.items()
        if key not in {"index", "filename", "filepath", "comps", "trialinfo"} and value not in (None, "")
    }
    return [{**constants, **row} for row in rows]


def _design_matrix(
    design: dict[str, Any], rows: list[dict[str, Any]], *, splitreg: bool, interaction: bool
) -> tuple[np.ndarray, list[str], np.ndarray]:
    categorical: list[tuple[str, list[Any]]] = []
    continuous: list[str] = []
    for variable in design.get("variable") or []:
        if not isinstance(variable, dict):
            continue
        label = str(variable.get("label") or "")
        if not label:
            continue
        if str(variable.get("vartype") or "categorical").lower() == "continuous":
            continuous.append(label)
        else:
            categorical.append((label, list(variable.get("value") or [])))
    columns: list[np.ndarray] = []
    names: list[str] = []
    main_effects: list[list[np.ndarray]] = []
    for label, levels in categorical:
        effects = []
        for level in levels:
            column = np.asarray([value_matches(row.get(label), level) for row in rows], dtype=float)
            columns.append(column)
            effects.append(column)
            names.append(f"{label}={_level_name(level)}")
        main_effects.append(effects)
    if interaction and len(main_effects) > 1:
        for combination in product(*main_effects):
            columns.append(np.prod(np.vstack(combination), axis=0))
        for levels in product(*(levels for _label, levels in categorical)):
            names.append(
                ":".join(f"{label}={_level_name(level)}" for (label, _values), level in zip(categorical, levels))
            )
    joint_conditions = list(product(*main_effects)) if main_effects else []
    joint_names = list(product(*(levels for _label, levels in categorical))) if categorical else []
    for label in continuous:
        values = np.asarray([_numeric_value(row.get(label)) for row in rows], dtype=float)
        finite = np.isfinite(values)
        if np.count_nonzero(finite) > 1:
            scale = float(np.std(values[finite], ddof=1))
            values[finite] = 0.0 if scale == 0 else (values[finite] - float(np.mean(values[finite]))) / scale
        if splitreg and joint_conditions:
            for condition, level_names in zip(joint_conditions, joint_names):
                membership = np.prod(np.vstack(condition), axis=0).astype(bool)
                column = np.zeros_like(values)
                column[membership] = values[membership]
                columns.append(column)
                suffix = ":".join(
                    f"{factor_label}={_level_name(level)}"
                    for (factor_label, _levels), level in zip(categorical, level_names)
                )
                names.append(f"{label}|{suffix}")
        else:
            columns.append(values)
            names.append(label)
    columns.append(np.ones(len(rows), dtype=float))
    names.append("constant")
    matrix = np.column_stack(columns)
    keep = np.all(np.isfinite(matrix), axis=1)
    if not np.any(keep):
        raise ValueError("no trials have complete values for the active LIMO design")
    matrix = matrix[keep]
    nonzero = np.any(np.abs(matrix) > np.finfo(float).eps, axis=0)
    matrix = matrix[:, nonzero]
    names = [name for name, retain in zip(names, nonzero) if retain]
    return matrix, names, keep


def _numeric_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _level_name(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "&".join(str(item) for item in value)
    return str(value)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "subject"


__all__ = ["pop_limo"]
