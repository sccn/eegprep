"""Shared EEGBrowser integration for rejection pop functions."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc._rejection import (
    copy_eeg,
    one_based_indices,
    parse_numeric_sequence,
    reject_field_names,
    rejection_data,
    update_reject_fields,
)
from eegprep.functions.popfunc.pop_eegplot import (
    DEFAULT_REJECTION_COLORS,
    MANUAL_REJECTION_COLOR,
    REJECTION_FAMILIES,
    pad_rejection_rows,
    rejection_row_count,
)
from eegprep.functions.popfunc.pop_rejepoch import pop_rejepoch
from eegprep.functions.sigprocfunc.eegplot import eegplot, eegplot2trial, trial2eegplot


# Autorej shares manual-color browser marks and is not superposed as a separate EEGLAB family.
_AUTO_REJECTION_COLOR = MANUAL_REJECTION_COLOR


def run_epoched_rejection(
    EEG: dict[str, Any],
    icacomp: int | bool,
    elecrange: Any,
    locthresh: Any,
    globthresh: Any,
    superpose: int | bool,
    reject: int | bool,
    vistype: int,
    *,
    marks_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    kind: str,
    stats_local_field: str,
    stats_global_field: str,
    stats_local_field_ica: str,
    stats_global_field_ica: str,
    error_message: str,
    command_fn: Callable[[list[int]], str],
    display: bool = False,
    command_callback: Any | None = None,
    show: bool = True,
) -> tuple[dict[str, Any], list[float], list[float], list[int], str]:
    """Run a local/global epoched-rejection method and store marks and stats.

    Shared scaffold for the per-method ``pop_*`` epoched rejection wrappers
    (kurtosis, joint probability, ...). The method-specific pieces are supplied
    by ``marks_fn``, ``kind``, the stats field names, the error message, and
    ``command_fn``, which builds the history string from the normalized
    1-based electrode/component range.
    """
    out = copy_eeg(EEG)
    data, row_count = rejection_data(out, icacomp)
    if int(out.get("trials", data.shape[2]) or data.shape[2]) <= 1:
        raise ValueError(error_message)
    elecrange = one_based_indices(elecrange, limit=row_count, default_all=True)
    marks, marks_e, local_scores, global_scores = marks_fn(data, elecrange, locthresh, globthresh)
    out.setdefault("stats", {})
    if int(bool(icacomp)):
        out["stats"][stats_local_field] = local_scores
        out["stats"][stats_global_field] = global_scores
    else:
        out["stats"][stats_local_field_ica] = local_scores
        out["stats"][stats_global_field_ica] = global_scores
    update_reject_fields(out, icacomp=icacomp, kind=kind, reject=marks, reject_e=marks_e)
    rejected = (np.flatnonzero(marks) + 1).tolist()
    command = command_fn(elecrange)
    if display:
        open_epoched_rejection_browser(
            out,
            data=data,
            icacomp=icacomp,
            elecrange=elecrange,
            kind=kind,
            superpose=superpose,
            reject=reject,
            command=command,
            command_callback=command_callback,
            show=show,
        )
    elif int(bool(reject)) and rejected:
        out = pop_rejepoch(out, rejected, 0)
    return (
        out,
        parse_numeric_sequence(locthresh, dtype=float),
        parse_numeric_sequence(globthresh, dtype=float),
        rejected,
        command,
    )


def vistype_from_gui(value: Any) -> int:
    """Map an EEGLAB visualization-mode popup value to a vistype flag."""
    if isinstance(value, str):
        return 0 if value.strip().lower() in {"rejecttrials", "reject trials", "0"} else 1
    try:
        return 0 if int(value) == 1 else 1
    except (TypeError, ValueError):
        return 1


def open_epoched_rejection_browser(
    EEG: dict[str, Any],
    *,
    data: np.ndarray,
    icacomp: int | bool,
    elecrange: list[int],
    kind: str,
    superpose: int | bool,
    reject: int | bool,
    command: str,
    command_callback: Any | None = None,
    show: bool = True,
) -> Any:
    """Open EEGBrowser with method-specific epoched rejection marks."""
    row_count = int(data.shape[0])
    selected_rows = _selected_row_indices(elecrange, row_count)
    selected_data = np.asarray(data, dtype=float)[selected_rows, :, :]
    ensure_rejection_defaults(EEG)
    winrej = initial_epoched_rejection_winrej(
        EEG,
        icacomp=icacomp,
        elecrange=elecrange,
        kind=kind,
        superpose=superpose,
        row_count=row_count,
    )

    def _accept(winrej_rows: np.ndarray) -> None:
        eeg_out, accept_command = apply_epoched_rejection_browser(
            EEG,
            winrej_rows,
            icacomp=icacomp,
            elecrange=elecrange,
            kind=kind,
            superpose=superpose,
            reject=reject,
            command=command,
            row_count=row_count,
            return_com=True,
        )
        if command_callback is not None:
            command_callback(eeg_out, accept_command)

    options: dict[str, Any] = {
        "srate": EEG.get("srate", 256),
        "limits": [
            float(EEG.get("xmin", 0.0) or 0.0) * 1000.0,
            float(EEG.get("xmax", 0.0) or 0.0) * 1000.0,
        ],
        "events": EEG.get("event", []),
        "winlength": 5,
        "xgrid": "off",
        "winrej": winrej,
        "wincolor": _kind_color(EEG, kind),
        "butlabel": "Reject" if int(bool(reject)) else "Update Marks",
        "show": show,
    }
    if show:
        options["command_callback"] = _accept
    if int(bool(icacomp)):
        options["eloc_file"] = _selected_chanlocs(EEG, selected_rows)
        options["title"] = f"Scroll channel activities -- eegplot() -- {EEG.get('setname', '')}".rstrip()
    else:
        options["component"] = True
        options["title"] = f"Scroll component activities -- eegplot() -- {EEG.get('setname', '')}".rstrip()
    return eegplot(selected_data, **options)


def initial_epoched_rejection_winrej(
    EEG: dict[str, Any],
    *,
    icacomp: int | bool,
    elecrange: list[int],
    kind: str,
    superpose: int | bool,
    row_count: int,
) -> np.ndarray:
    """Build selected-channel ``winrej`` rows for a rejection-family browser."""
    reject = EEG.get("reject") or {}
    trials = int(EEG.get("trials", 1) or 1)
    pnts = int(EEG.get("pnts", np.asarray(EEG.get("data")).shape[1]))
    selected_rows = _selected_row_indices(elecrange, row_count)
    rows: list[np.ndarray] = []
    current_rows = _family_rows(
        reject,
        kind,
        icacomp=icacomp,
        trials=trials,
        row_count=row_count,
        pnts=pnts,
        selected_rows=selected_rows,
        color=_kind_color(EEG, kind),
    )
    if int(superpose) == 2:
        for family in _displayed_families(reject):
            family_kind = f"rej{family}"
            if family_kind == kind:
                continue
            family_rows = _family_rows(
                reject,
                family_kind,
                icacomp=icacomp,
                trials=trials,
                row_count=row_count,
                pnts=pnts,
                selected_rows=selected_rows,
                color=_family_color(reject, family),
            )
            if family_rows.size:
                rows.append(family_rows)
    if int(superpose) == 1:
        old_rows = _family_rows(
            reject,
            kind,
            icacomp=icacomp,
            trials=trials,
            row_count=row_count,
            pnts=pnts,
            selected_rows=selected_rows,
            color=_lighter_color(_kind_color(EEG, kind)),
        )
        if old_rows.size:
            rows.append(old_rows)
    if current_rows.size:
        rows.append(current_rows)
    return np.vstack(rows) if rows else np.zeros((0, 5 + len(selected_rows)), dtype=float)


def apply_epoched_rejection_browser(
    EEG: dict[str, Any],
    winrej: Any,
    *,
    icacomp: int | bool,
    elecrange: list[int],
    kind: str,
    superpose: int | bool,
    reject: int | bool,
    command: str,
    row_count: int | None = None,
    return_com: bool = False,
) -> Any:
    """Apply accepted EEGBrowser rejection rows to method-specific fields."""
    out = copy_eeg(EEG)
    ensure_rejection_defaults(out)
    trials = int(out.get("trials", 1) or 1)
    pnts = int(out.get("pnts", np.asarray(out.get("data")).shape[1]))
    if row_count is None:
        row_count = rejection_row_count(out, icacomp)
    selected_rows = _selected_row_indices(elecrange, row_count)
    rows = _as_winrej_rows(winrej)
    if int(superpose) == 2:
        _apply_superposed_family_rows(
            out,
            rows,
            icacomp=icacomp,
            kind=kind,
            trials=trials,
            row_count=row_count,
            selected_rows=selected_rows,
            pnts=pnts,
        )
        trial_marks, row_marks = _all_rows_to_trial_marks(rows, pnts, trials, row_count, selected_rows)
    else:
        trial_marks, selected_marks = eegplot2trial(rows, pnts, trials, _kind_color(out, kind), None)
        row_marks = _expand_selected_row_marks(selected_marks, row_count, selected_rows, trials)
        update_reject_fields(out, icacomp=icacomp, kind=kind, reject=trial_marks, reject_e=row_marks)
        _store_manual_marks(out, trial_marks, row_marks, icacomp=icacomp)
    if int(bool(reject)) and np.asarray(trial_marks, dtype=bool).any():
        out, _reject_command = pop_rejepoch(out, np.asarray(trial_marks, dtype=bool), 1, return_com=True)
    return (out, command) if return_com else out


def ensure_rejection_defaults(EEG: dict[str, Any]) -> None:
    """Ensure EEGLAB-compatible rejection color fields exist."""
    reject = EEG.setdefault("reject", {})
    reject.setdefault("rejmanualcol", np.asarray(MANUAL_REJECTION_COLOR, dtype=float))
    for family, color in DEFAULT_REJECTION_COLORS.items():
        reject.setdefault(f"rej{family}col", np.asarray(color, dtype=float))


def _apply_superposed_family_rows(
    EEG: dict[str, Any],
    rows: np.ndarray,
    *,
    icacomp: int | bool,
    kind: str,
    trials: int,
    row_count: int,
    selected_rows: np.ndarray,
    pnts: int,
) -> None:
    reject = EEG.setdefault("reject", {})
    families = set(_displayed_families(reject))
    families.add(_family_from_kind(kind))
    for family in sorted(families):
        family_kind = f"rej{family}"
        trial_marks, selected_marks = eegplot2trial(rows, pnts, trials, _family_color(reject, family), None)
        row_marks = _expand_selected_row_marks(selected_marks, row_count, selected_rows, trials)
        update_reject_fields(EEG, icacomp=icacomp, kind=family_kind, reject=trial_marks, reject_e=row_marks)


def _all_rows_to_trial_marks(
    rows: np.ndarray,
    pnts: int,
    trials: int,
    row_count: int,
    selected_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    trial_marks, selected_marks = eegplot2trial(rows, pnts, trials, None, None)
    return trial_marks, _expand_selected_row_marks(selected_marks, row_count, selected_rows, trials)


def _family_rows(
    reject: dict[str, Any],
    kind: str,
    *,
    icacomp: int | bool,
    trials: int,
    row_count: int,
    pnts: int,
    selected_rows: np.ndarray,
    color: tuple[float, float, float],
) -> np.ndarray:
    field, field_e = reject_field_names(icacomp, kind)
    marks = _trial_marks(reject.get(field), trials)
    row_marks = pad_rejection_rows(reject.get(field_e), row_count, trials)[selected_rows, :]
    return trial2eegplot(marks, row_marks, pnts, color)


def _store_manual_marks(
    EEG: dict[str, Any],
    trial_marks: np.ndarray,
    row_marks: np.ndarray,
    *,
    icacomp: int | bool,
) -> None:
    field, field_e = reject_field_names(icacomp, "rejmanual")
    reject = EEG.setdefault("reject", {})
    current = _trial_marks(reject.get(field), np.asarray(trial_marks).size)
    current_e = pad_rejection_rows(reject.get(field_e), row_marks.shape[0], np.asarray(trial_marks).size)
    reject[field] = current | np.asarray(trial_marks, dtype=bool)
    reject[field_e] = current_e | np.asarray(row_marks, dtype=bool)
    reject.setdefault("rejmanualcol", np.asarray(MANUAL_REJECTION_COLOR, dtype=float))


def _selected_row_indices(elecrange: list[int], row_count: int) -> np.ndarray:
    selected = np.asarray(elecrange, dtype=int).ravel() - 1
    if selected.size == 0:
        selected = np.arange(row_count, dtype=int)
    if np.any(selected < 0) or np.any(selected >= row_count):
        raise ValueError("Index out of range")
    return selected


def _expand_selected_row_marks(
    selected_marks: np.ndarray,
    row_count: int,
    selected_rows: np.ndarray,
    trials: int,
) -> np.ndarray:
    out = np.zeros((row_count, trials), dtype=bool)
    marks = np.asarray(selected_marks, dtype=bool)
    if marks.ndim == 1 and marks.size:
        marks = marks.reshape(1, -1)
    if marks.ndim != 2 or marks.size == 0:
        return out
    rows = min(selected_rows.size, marks.shape[0])
    cols = min(trials, marks.shape[1])
    out[selected_rows[:rows], :cols] = marks[:rows, :cols]
    return out


def _trial_marks(value: Any, trials: int) -> np.ndarray:
    out = np.zeros(trials, dtype=bool)
    marks = np.asarray(value if value is not None else [], dtype=bool).ravel()
    out[: min(trials, marks.size)] = marks[:trials]
    return out


def _displayed_families(reject: dict[str, Any]) -> tuple[str, ...]:
    disprej = reject.get("disprej")
    if disprej is not None and np.asarray(disprej, dtype=object).size:
        return tuple(str(item) for item in np.asarray(disprej, dtype=object).ravel() if str(item) in REJECTION_FAMILIES)
    return tuple(family for family in REJECTION_FAMILIES if _has_family_marks(reject, family))


def _has_family_marks(reject: dict[str, Any], family: str) -> bool:
    return any(np.asarray(reject.get(field, [])).size for field in (f"rej{family}", f"icarej{family}"))


def _kind_color(EEG: dict[str, Any], kind: str) -> tuple[float, float, float]:
    reject = EEG.get("reject") or {}
    family = _family_from_kind(kind)
    if family == "auto":
        return _AUTO_REJECTION_COLOR
    return _family_color(reject, family)


def _family_color(reject: dict[str, Any], family: str) -> tuple[float, float, float]:
    default = DEFAULT_REJECTION_COLORS.get(family, MANUAL_REJECTION_COLOR)
    values = np.asarray(reject.get(f"rej{family}col", default), dtype=float).ravel()
    if values.size < 3:
        return default
    return tuple(float(item) for item in values[:3])


def _family_from_kind(kind: str) -> str:
    family = kind[3:] if kind.startswith("rej") else kind
    return family or "manual"


def _lighter_color(color: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(min(float(component) + 0.15, 1.0) for component in color)


def _selected_chanlocs(EEG: dict[str, Any], selected_rows: np.ndarray) -> list[dict[str, Any]]:
    chanlocs = chanlocs_as_list(EEG.get("chanlocs", []))
    return [chanlocs[index] for index in selected_rows if index < len(chanlocs)]


def _as_winrej_rows(winrej: Any) -> np.ndarray:
    rows = np.asarray(winrej if winrej is not None else [], dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.size == 0:
        return np.zeros((0, 5), dtype=float)
    return rows
