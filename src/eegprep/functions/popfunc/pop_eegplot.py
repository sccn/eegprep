"""EEGLAB-style wrapper for the scrolling EEG browser."""

from __future__ import annotations

from typing import Any

from eegprep.functions.popfunc._plot_utils import history_command
import numpy as np

from eegprep.functions.popfunc._rejection import copy_eeg
from eegprep.functions.popfunc.eeg_eegrej import eeg_eegrej
from eegprep.functions.popfunc.pop_rejepoch import pop_rejepoch
from eegprep.functions.sigprocfunc.eegplot import (
    DEFAULT_WINREJ_COLOR,
    eegplot,
    eegplot2event,
    eegplot2trial,
    normalize_winrej,
    trial2eegplot,
    winrej_to_array,
)


MANUAL_REJECTION_COLOR = (1.0, 0.9, 0.9)
DEFAULT_REJECTION_COLORS = {
    "manual": (1.0000, 1.0000, 0.7830),
    "thresh": (0.8487, 1.0000, 0.5008),
    "const": (0.6940, 1.0000, 0.7008),
    "jp": (1.0000, 0.6991, 0.7537),
    "kurt": (0.6880, 0.7042, 1.0000),
    "freq": (0.9596, 0.7193, 1.0000),
}
REJECTION_FAMILIES = ("manual", "thresh", "const", "jp", "kurt", "freq")
CONTINUOUS_MANUAL_WINREJ = "rejmanualwinrej"
CONTINUOUS_ICA_MANUAL_WINREJ = "icarejmanualwinrej"


def pop_eegplot(
    EEG: dict[str, Any] | None = None,
    icacomp: int = 1,
    superpose: int = 0,
    reject: int = 1,
    topcommand: Any = None,
    *args: Any,
    return_com: bool = False,
    show: bool = True,
    **kwargs: Any,
):
    """Inspect EEG channel or component activity in the scrolling browser.

    The browser edits marks in window-local state. Data or rejection fields are
    changed only when the Accept/Reject callback runs.
    """
    del topcommand
    if EEG is None:
        raise ValueError("pop_eegplot requires a non-empty EEG dataset")
    if icacomp not in {0, 1}:
        raise ValueError("icacomp must be 1 for data channels or 0 for components")
    if int(bool(icacomp)) == 0:
        _require_ica(EEG)
    options = dict(kwargs)
    options.setdefault("srate", EEG.get("srate", 256))
    options.setdefault(
        "limits", [float(EEG.get("xmin", 0.0) or 0.0) * 1000.0, float(EEG.get("xmax", 0.0) or 0.0) * 1000.0]
    )
    options.setdefault("events", EEG.get("event", []))
    accept_callback = options.pop("command_callback", None)
    options.setdefault("wincolor", _manual_color(EEG))
    options.setdefault("butlabel", "Reject" if int(bool(reject)) else "Update Marks")
    trials = int(EEG.get("trials", 1) or 1)
    if trials > 1:
        options.setdefault("winrej", _initial_epoch_winrej(EEG, icacomp, superpose))
    else:
        options.setdefault("winrej", _initial_continuous_winrej(EEG, icacomp))

    def _accept(winrej: np.ndarray) -> None:
        eeg_out, accept_command = apply_eegplot_rejections(
            EEG,
            winrej,
            icacomp=icacomp,
            superpose=superpose,
            reject=reject,
            return_com=True,
        )
        if accept_callback is not None:
            accept_callback(eeg_out, accept_command)

    if return_com and show and accept_callback is None:
        raise ValueError(
            "pop_eegplot(return_com=True) opens a non-blocking browser; pass command_callback to receive "
            "accepted browser results, or use apply_eegplot_rejections for programmatic rejection."
        )
    if show:
        options["command_callback"] = _accept
    if icacomp == 1:
        options.setdefault("eloc_file", EEG.get("chanlocs", []))
        options.setdefault("title", f"Scroll channel activities -- eegplot() -- {EEG.get('setname', '')}".rstrip())
        window = eegplot(EEG, *args, show=show, **options)
    else:
        options.setdefault("component", True)
        options.setdefault("title", f"Scroll component activities -- eegplot() -- {EEG.get('setname', '')}".rstrip())
        window = eegplot(EEG, *args, show=show, **options)
    command = history_command("pop_eegplot", icacomp, superpose, reject)
    if return_com and show:
        return (None, command)
    return (EEG, command) if return_com else window


def eegplot_accept_creates_dataset(input_eeg: dict[str, Any], output_eeg: dict[str, Any], reject: int = 1) -> bool:
    """Return whether accepted ``pop_eegplot`` output should be a new dataset.

    Rejection accepts that remove continuous samples or epochs create a new
    EEGLAB-style dataset. Mark-only updates and no-op rejects update in place.
    For example, ``(pnts=100, pnts=80, reject=1)`` returns true, while
    ``(pnts=100, pnts=100, reject=1)`` and any ``reject=0`` call return false.
    """
    if not int(bool(reject)):
        return False
    input_trials = int(input_eeg.get("trials", 1) or 1)
    output_trials = int(output_eeg.get("trials", 1) or 1)
    if input_trials > 1:
        return output_trials < input_trials
    return _dataset_pnts(output_eeg) < _dataset_pnts(input_eeg)


def apply_eegplot_rejections(
    EEG: dict[str, Any],
    winrej: Any,
    *,
    icacomp: int = 1,
    superpose: int = 0,
    reject: int = 1,
    return_com: bool = False,
):
    """Apply accepted EEGBrowser marks to an EEG dataset."""
    out = copy_eeg(EEG)
    command = history_command("pop_eegplot", icacomp, superpose, reject)
    rows = _as_winrej_rows(winrej)

    trials = int(out.get("trials", 1) or 1)
    if trials <= 1:
        if int(bool(reject)):
            if rows.size:
                out = eeg_eegrej(out, eegplot2event(rows, -1))
                _clear_continuous_marks(out, icacomp=icacomp)
        else:
            _store_continuous_marks(out, rows, icacomp=icacomp)
        return (out, command) if return_com else out

    pnts = int(out.get("pnts", np.asarray(out.get("data")).shape[1]))
    if rows.size:
        trial_marks, row_marks = eegplot2trial(rows, pnts, trials, _manual_color(out), None)
        store_superpose = superpose
    else:
        trial_marks = np.zeros(trials, dtype=bool)
        row_marks = np.zeros((rejection_row_count(out, icacomp), trials), dtype=bool)
        store_superpose = 0
    _store_epoch_marks(out, trial_marks, row_marks, icacomp=icacomp, superpose=store_superpose)
    if int(bool(reject)) and trial_marks.any():
        out, _reject_command = pop_rejepoch(out, trial_marks, 1, return_com=True)
    return (out, command) if return_com else out


def _initial_epoch_winrej(EEG: dict[str, Any], icacomp: int, superpose: int) -> np.ndarray:
    reject = EEG.get("reject") or {}
    trials = int(EEG.get("trials", 1) or 1)
    pnts = int(EEG.get("pnts", np.asarray(EEG.get("data")).shape[1]))
    row_count = rejection_row_count(EEG, icacomp)
    manual, manual_e = _reject_arrays(reject, "rejmanual", trials, row_count, icacomp=icacomp)
    if int(superpose) == 0:
        return trial2eegplot(manual, manual_e, pnts, _manual_color(EEG))
    if int(superpose) == 2:
        return _superposed_epoch_winrej(EEG, reject, icacomp, trials, row_count, pnts, manual, manual_e)
    rows = []
    old_color = tuple(min(component + 0.15, 1.0) for component in _manual_color(EEG))
    old, old_e = _reject_arrays(reject, "rejglobal", trials, row_count, icacomp=1)
    old_rows = trial2eegplot(old, old_e, pnts, old_color)
    if old_rows.size:
        rows.append(old_rows)
    manual_rows = trial2eegplot(manual, manual_e, pnts, _manual_color(EEG))
    if manual_rows.size:
        rows.append(manual_rows)
    return np.vstack(rows) if rows else np.zeros((0, 5 + row_count), dtype=float)


def _initial_continuous_winrej(EEG: dict[str, Any], icacomp: int) -> np.ndarray:
    row_count = rejection_row_count(EEG, icacomp)
    rows = _as_winrej_rows((EEG.get("reject") or {}).get(_continuous_mark_field(icacomp), []))
    if rows.size == 0:
        return np.zeros((0, 5 + row_count), dtype=float)
    total_samples = _continuous_total_samples(EEG)
    try:
        return winrej_to_array(normalize_winrej(rows, row_count, total_samples), row_count)
    except ValueError as exc:
        field = _continuous_mark_field(icacomp)
        raise ValueError(f"EEG.reject.{field} contains invalid eegplot winrej rows") from exc


def _superposed_epoch_winrej(
    EEG: dict[str, Any],
    reject: dict[str, Any],
    icacomp: int,
    trials: int,
    row_count: int,
    pnts: int,
    manual: np.ndarray,
    manual_e: np.ndarray,
) -> np.ndarray:
    rows = []
    manual_color = _manual_color(EEG)
    for family in _displayed_rejection_families(reject):
        color = _reject_color(reject, family, DEFAULT_REJECTION_COLORS.get(family, manual_color))
        if tuple(color) == tuple(manual_color):
            continue
        marks, marks_e = _reject_arrays(reject, f"rej{family}", trials, row_count, icacomp=icacomp)
        family_rows = trial2eegplot(marks, marks_e, pnts, color)
        if family_rows.size:
            rows.append(family_rows)
    manual_rows = trial2eegplot(manual, manual_e, pnts, manual_color)
    if manual_rows.size:
        rows.append(manual_rows)
    return np.vstack(rows) if rows else np.zeros((0, 5 + row_count), dtype=float)


def _store_epoch_marks(
    EEG: dict[str, Any],
    trial_marks: np.ndarray,
    row_marks: np.ndarray,
    *,
    icacomp: int,
    superpose: int,
) -> None:
    reject = EEG.setdefault("reject", {})
    field = "rejmanual" if int(bool(icacomp)) else "icarejmanual"
    field_e = f"{field}E"
    trials = int(EEG.get("trials", 1) or 1)
    row_count = rejection_row_count(EEG, icacomp)
    current, current_e = _reject_arrays(reject, "rejmanual", trials, row_count, icacomp=icacomp)
    if int(superpose):
        trial_marks = np.asarray(trial_marks, dtype=bool) | current
        row_marks = pad_rejection_rows(row_marks, row_count, trials) | current_e
    reject[field] = np.asarray(trial_marks, dtype=bool)
    reject[field_e] = pad_rejection_rows(row_marks, row_count, trials)
    reject.setdefault("rejmanualcol", np.asarray(MANUAL_REJECTION_COLOR, dtype=float))


def _store_continuous_marks(EEG: dict[str, Any], rows: np.ndarray, *, icacomp: int) -> None:
    reject = EEG.setdefault("reject", {})
    row_count = rejection_row_count(EEG, icacomp)
    if rows.size == 0:
        reject[_continuous_mark_field(icacomp)] = np.zeros((0, 5 + row_count), dtype=float)
    else:
        total_samples = _continuous_total_samples(EEG)
        reject[_continuous_mark_field(icacomp)] = winrej_to_array(
            normalize_winrej(rows, row_count, total_samples), row_count
        )
    reject.setdefault("rejmanualcol", np.asarray(MANUAL_REJECTION_COLOR, dtype=float))


def _clear_continuous_marks(EEG: dict[str, Any], *, icacomp: int) -> None:
    reject = EEG.setdefault("reject", {})
    reject[_continuous_mark_field(icacomp)] = np.zeros((0, 5 + rejection_row_count(EEG, icacomp)), dtype=float)


def _continuous_mark_field(icacomp: int) -> str:
    return CONTINUOUS_MANUAL_WINREJ if int(bool(icacomp)) else CONTINUOUS_ICA_MANUAL_WINREJ


def _continuous_total_samples(EEG: dict[str, Any]) -> int:
    data = np.asarray(EEG.get("data"))
    if data.ndim >= 2:
        return int(data.shape[1])
    return int(EEG.get("pnts", 0) or 0)


def _as_winrej_rows(winrej: Any) -> np.ndarray:
    rows = np.asarray(winrej if winrej is not None else [], dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    return rows


def _reject_arrays(
    reject: dict[str, Any],
    kind: str,
    trials: int,
    row_count: int,
    *,
    icacomp: int,
) -> tuple[np.ndarray, np.ndarray]:
    prefix = "" if int(bool(icacomp)) else "ica"
    marks = np.asarray(reject.get(f"{prefix}{kind}", []), dtype=bool).ravel()
    out = np.zeros(trials, dtype=bool)
    out[: min(trials, marks.size)] = marks[:trials]
    row_marks = pad_rejection_rows(np.asarray(reject.get(f"{prefix}{kind}E", []), dtype=bool), row_count, trials)
    return out, row_marks


def pad_rejection_rows(values: np.ndarray, row_count: int, trials: int) -> np.ndarray:
    """Zero-pad/crop a row-mask array to ``(row_count, trials)``."""
    out = np.zeros((row_count, trials), dtype=bool)
    arr = np.asarray(values, dtype=bool)
    if arr.ndim == 1 and arr.size:
        arr = arr.reshape(1, -1)
    if arr.ndim == 2:
        rows = min(row_count, arr.shape[0])
        cols = min(trials, arr.shape[1])
        out[:rows, :cols] = arr[:rows, :cols]
    return out


def rejection_row_count(EEG: dict[str, Any], icacomp: int) -> int:
    """Return the number of channel or component rows for rejection marks."""
    if int(bool(icacomp)):
        return int(EEG.get("nbchan", np.asarray(EEG.get("data")).shape[0]) or 0)
    weights = np.asarray(EEG.get("icaweights", []))
    return int(weights.shape[0]) if weights.ndim == 2 else 0


def _require_ica(EEG: dict[str, Any]) -> None:
    if _nonempty_array(EEG.get("icaact")):
        return
    if _nonempty_array(EEG.get("icaweights")) and _nonempty_array(EEG.get("icasphere")):
        return
    raise ValueError("You must run ICA before browsing component activations with pop_eegplot.")


def _nonempty_array(value: Any) -> bool:
    return value is not None and np.asarray(value).size > 0


def _dataset_pnts(EEG: dict[str, Any]) -> int:
    if "pnts" in EEG:
        return int(EEG.get("pnts", 0) or 0)
    data = np.asarray(EEG.get("data"))
    if data.ndim >= 2:
        return int(data.shape[1])
    return 0


def _displayed_rejection_families(reject: dict[str, Any]) -> tuple[str, ...]:
    disprej = reject.get("disprej")
    if disprej is None or np.asarray(disprej, dtype=object).size == 0:
        return tuple(family for family in REJECTION_FAMILIES if _has_rejection_family(reject, family))
    values = np.asarray(disprej, dtype=object).ravel().tolist()
    return tuple(str(value) for value in values if str(value) in REJECTION_FAMILIES)


def _has_rejection_family(reject: dict[str, Any], family: str) -> bool:
    data_marks = reject.get(f"rej{family}")
    component_marks = reject.get(f"icarej{family}")
    return (data_marks is not None and np.asarray(data_marks).size > 0) or (
        component_marks is not None and np.asarray(component_marks).size > 0
    )


def _manual_color(EEG: dict[str, Any]) -> tuple[float, float, float]:
    reject = EEG.get("reject") or {}
    return _reject_color(reject, "manual", MANUAL_REJECTION_COLOR)


def _reject_color(
    reject: dict[str, Any], family: str, default: tuple[float, float, float]
) -> tuple[float, float, float]:
    color = reject.get(f"rej{family}col", default)
    values = np.asarray(color if color is not None else DEFAULT_WINREJ_COLOR, dtype=float).ravel()
    if values.size < 3:
        return default
    return tuple(float(item) for item in values[:3])


__all__ = ["apply_eegplot_rejections", "eegplot_accept_creates_dataset", "pop_eegplot"]
