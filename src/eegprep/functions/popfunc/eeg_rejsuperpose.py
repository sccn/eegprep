"""Superpose EEGLAB-style rejection marks."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.popfunc._rejection import copy_eeg


def eeg_rejsuperpose(
    EEG: dict[str, Any],
    typerej: int | bool,
    Rmanual: int | bool,
    Rthres: int | bool,
    Rconst: int | bool,
    Rent: int | bool,
    Rkurt: int | bool,
    Rfreq: int | bool,
    Rothertype: int | bool,
    *,
    return_com: bool = False,
):
    """Combine stored rejection marks into ``EEG.reject.rejglobal``.

    ``typerej`` follows EEGLAB: ``1`` means raw data marks and ``0`` means ICA
    activity marks. ``Rothertype`` includes both raw and ICA mark families.
    """
    out = copy_eeg(EEG)
    trials = int(out.get("trials", 1) or 1)
    reject = out.setdefault("reject", {})
    row_count = int(out.get("nbchan", 0) or 0) if int(bool(typerej)) else _component_count(out)
    rejglobal = np.zeros(trials, dtype=bool)
    rejglobal_e = np.zeros((row_count, trials), dtype=bool)

    is_data_rejection = int(bool(typerej))
    include_other = int(bool(Rothertype))

    if is_data_rejection or include_other:
        rejglobal, rejglobal_e = _add_family(
            reject,
            "",
            rejglobal,
            rejglobal_e,
            include_rows=bool(is_data_rejection),
            manual=Rmanual,
            thres=Rthres,
            const=Rconst,
            ent=Rent,
            kurt=Rkurt,
            freq=Rfreq,
        )
    if not is_data_rejection or include_other:
        rejglobal, rejglobal_e = _add_family(
            reject,
            "ica",
            rejglobal,
            rejglobal_e,
            include_rows=not bool(is_data_rejection),
            manual=Rmanual,
            thres=Rthres,
            const=Rconst,
            ent=Rent,
            kurt=Rkurt,
            freq=Rfreq,
        )

    reject["rejglobal"] = rejglobal
    reject["rejglobalE"] = rejglobal_e
    command = (
        "EEG = eeg_rejsuperpose(EEG, "
        f"{int(bool(typerej))}, {int(bool(Rmanual))}, {int(bool(Rthres))}, "
        f"{int(bool(Rconst))}, {int(bool(Rent))}, {int(bool(Rkurt))}, "
        f"{int(bool(Rfreq))}, {int(bool(Rothertype))});"
    )
    return (out, command) if return_com else out


def _add_family(
    reject: dict[str, Any],
    prefix: str,
    rejglobal: np.ndarray,
    rejglobal_e: np.ndarray,
    *,
    include_rows: bool,
    manual: int | bool,
    thres: int | bool,
    const: int | bool,
    ent: int | bool,
    kurt: int | bool,
    freq: int | bool,
) -> tuple[np.ndarray, np.ndarray]:
    pairs = (
        ("rejmanual", manual),
        ("rejthresh", thres),
        ("rejconst", const),
        ("rejjp", ent),
        ("rejkurt", kurt),
        ("rejfreq", freq),
    )
    for name, enabled in pairs:
        if not int(bool(enabled)):
            continue
        rejglobal = _or_vector(rejglobal, reject.get(f"{prefix}{name}"))
        if include_rows:
            rejglobal_e = _or_matrix(rejglobal_e, reject.get(f"{prefix}{name}E"))
    return rejglobal, rejglobal_e


def _or_vector(dest: np.ndarray, source: Any) -> np.ndarray:
    if source is None or np.asarray(source).size == 0:
        return dest
    values = np.asarray(source, dtype=bool).ravel()
    out = dest.copy()
    out[: min(out.size, values.size)] |= values[: out.size]
    return out


def _or_matrix(dest: np.ndarray, source: Any) -> np.ndarray:
    if source is None or np.asarray(source).size == 0:
        return dest
    values = np.asarray(source, dtype=bool)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    out = dest.copy()
    rows = min(out.shape[0], values.shape[0])
    cols = min(out.shape[1], values.shape[1])
    out[:rows, :cols] |= values[:rows, :cols]
    return out


def _component_count(EEG: dict[str, Any]) -> int:
    weights = np.asarray(EEG.get("icaweights", []))
    if weights.ndim == 2 and weights.size:
        return int(weights.shape[0])
    winv = np.asarray(EEG.get("icawinv", []))
    if winv.ndim == 2 and winv.size:
        return int(winv.shape[1])
    return 0
