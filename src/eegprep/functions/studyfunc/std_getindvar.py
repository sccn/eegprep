"""Inspect independent variables available in a STUDY."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._study_utils import (
    RESERVED_VARIABLE_FIELDS,
    _empty_value,
    ensure_study,
    equal_value,
    trialinfo_rows,
)


DATASETINFO_EXCLUDE = RESERVED_VARIABLE_FIELDS | {"ncomps", "subject"}


def std_getindvar(
    STUDY: dict[str, Any],
    mode: str = "both",
    scandesign: bool | int = False,
) -> tuple[list[str], list[list[Any]], list[list[list[Any]]], list[str]]:
    """Return STUDY factor names, values, subject groupings, and pairing flags."""
    study = ensure_study(STUDY)
    mode = str(mode or "both").lower()
    if mode not in {"datinfo", "trialinfo", "both"}:
        raise ValueError("mode must be 'datinfo', 'trialinfo', or 'both'")
    factors: list[str] = []
    factorvals: list[list[Any]] = []
    subjects: list[list[list[Any]]] = []
    paired: list[str] = []
    infos = [info for info in study.get("datasetinfo") or [] if isinstance(info, dict)]
    if mode in {"datinfo", "both"}:
        _append_dataset_factors(infos, factors, factorvals, subjects, paired)
    if mode in {"trialinfo", "both"}:
        _append_trial_factors(infos, factors, factorvals, subjects, paired)
    if scandesign:
        _append_design_values(study, factors, factorvals)
    return factors, factorvals, subjects, paired


def _append_dataset_factors(
    infos: list[dict[str, Any]],
    factors: list[str],
    factorvals: list[list[Any]],
    subjects: list[list[list[Any]]],
    paired: list[str],
) -> None:
    labels = []
    for info in infos:
        for label, value in info.items():
            if label not in DATASETINFO_EXCLUDE and not _empty_value(value) and label not in labels:
                labels.append(label)
    for label in labels:
        values = _unique_values(info.get(label) for info in infos)
        if len(values) <= 1:
            continue
        factors.append(label)
        factorvals.append(values)
        value_subjects = [
            _unique_values(info.get("subject") for info in infos if _equal_value(info.get(label), value))
            for value in values
        ]
        subjects.append(value_subjects)
        paired.append("on" if _paired_subject_sets(value_subjects) else "off")


def _append_trial_factors(
    infos: list[dict[str, Any]],
    factors: list[str],
    factorvals: list[list[Any]],
    subjects: list[list[list[Any]]],
    paired: list[str],
) -> None:
    labels = []
    for info in infos:
        for row in trialinfo_rows(info.get("trialinfo")):
            for label, value in row.items():
                if not _empty_value(value) and label not in labels:
                    labels.append(label)
    for label in labels:
        values = _unique_values(
            row.get(label) for info in infos for row in trialinfo_rows(info.get("trialinfo")) if label in row
        )
        if len(values) <= 1 or label in factors:
            continue
        factors.append(label)
        factorvals.append(values)
        value_subjects = [
            _unique_values(
                info.get("subject")
                for info in infos
                if any(_equal_value(row.get(label), value) for row in trialinfo_rows(info.get("trialinfo")))
            )
            for value in values
        ]
        subjects.append(value_subjects)
        paired.append("on" if _paired_subject_sets(value_subjects) else "off")


def _append_design_values(study: dict[str, Any], factors: list[str], factorvals: list[list[Any]]) -> None:
    for design in study.get("design") or []:
        if not isinstance(design, dict):
            continue
        for variable in design.get("variable") or []:
            if not isinstance(variable, dict) or str(variable.get("vartype") or "categorical") != "categorical":
                continue
            label = str(variable.get("label") or "")
            if not label:
                continue
            if label not in factors:
                factors.append(label)
                factorvals.append([])
            index = factors.index(label)
            for value in variable.get("value") or []:
                if not any(_equal_value(value, existing) for existing in factorvals[index]):
                    factorvals[index].append(value)


def _unique_values(values: Any) -> list[Any]:
    output: list[Any] = []
    for value in values:
        if _empty_value(value) or any(_equal_value(value, existing) for existing in output):
            continue
        output.append(value)
    try:
        return sorted(output)
    except TypeError:
        return output


def _same_subjects(value_subjects: list[list[Any]]) -> bool:
    if not value_subjects:
        return True
    first = value_subjects[0]
    return all(subjects == first for subjects in value_subjects)


def _paired_subject_sets(value_subjects: list[list[Any]]) -> bool:
    return bool(value_subjects) and all(value_subjects) and _same_subjects(value_subjects)


def _equal_value(left: Any, right: Any) -> bool:
    return equal_value(left, right)


__all__ = ["std_getindvar"]
