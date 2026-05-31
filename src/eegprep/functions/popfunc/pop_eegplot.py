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
    trial2eegplot,
)


MANUAL_REJECTION_COLOR = (1.0, 0.9, 0.9)


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
        return (None, "") if return_com else None
    if icacomp not in {0, 1}:
        raise ValueError("icacomp must be 1 for data channels or 0 for components")
    options = dict(kwargs)
    options.setdefault("srate", EEG.get("srate", 256))
    options.setdefault(
        "limits", [float(EEG.get("xmin", 0.0) or 0.0) * 1000.0, float(EEG.get("xmax", 0.0) or 0.0) * 1000.0]
    )
    options.setdefault("events", EEG.get("event", []))
    accept_callback = options.pop("command_callback", None)
    options.setdefault("wincolor", _manual_color(EEG))
    options.setdefault("butlabel", "Reject" if int(bool(reject)) else "Update Marks")
    if int(EEG.get("trials", 1) or 1) > 1:
        options.setdefault("winrej", _initial_epoch_winrej(EEG, icacomp, superpose))

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
    return (EEG, command) if return_com else window


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
    rows = np.asarray(winrej if winrej is not None else [], dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)

    trials = int(out.get("trials", 1) or 1)
    if trials <= 1:
        if rows.size and int(bool(reject)):
            out = eeg_eegrej(out, eegplot2event(rows, -1))
        return (out, command) if return_com else out

    pnts = int(out.get("pnts", np.asarray(out.get("data")).shape[1]))
    if rows.size:
        trial_marks, row_marks = eegplot2trial(rows, pnts, trials, _manual_color(out), None)
        store_superpose = superpose
    else:
        trial_marks = np.zeros(trials, dtype=bool)
        row_marks = np.zeros((_row_count(out, icacomp), trials), dtype=bool)
        store_superpose = 0
    _store_epoch_marks(out, trial_marks, row_marks, icacomp=icacomp, superpose=store_superpose)
    if int(bool(reject)) and trial_marks.any():
        out, _reject_command = pop_rejepoch(out, trial_marks, 1, return_com=True)
    return (out, command) if return_com else out


def _initial_epoch_winrej(EEG: dict[str, Any], icacomp: int, superpose: int) -> np.ndarray:
    reject = EEG.get("reject") or {}
    trials = int(EEG.get("trials", 1) or 1)
    pnts = int(EEG.get("pnts", np.asarray(EEG.get("data")).shape[1]))
    row_count = _row_count(EEG, icacomp)
    manual, manual_e = _reject_arrays(reject, "rejmanual", trials, row_count, icacomp=icacomp)
    if int(superpose) == 0:
        return trial2eegplot(manual, manual_e, pnts, _manual_color(EEG))
    rows = []
    old_color = tuple(min(component + 0.15, 1.0) for component in _manual_color(EEG))
    old, old_e = _reject_arrays(reject, "rejglobal", trials, row_count, icacomp=1)
    old_rows = trial2eegplot(old, old_e, pnts, old_color)
    if old_rows.size:
        rows.append(old_rows)
    # TODO: Overlay EEGLAB's method-specific families (rejthresh, rejconst,
    # rejjp, rejkurt, rejfreq) once Phase 2/visual parity work owns the full
    # rejection-color legend surface.
    manual_rows = trial2eegplot(manual, manual_e, pnts, _manual_color(EEG))
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
    row_count = _row_count(EEG, icacomp)
    current, current_e = _reject_arrays(reject, "rejmanual", trials, row_count, icacomp=icacomp)
    if int(superpose):
        trial_marks = np.asarray(trial_marks, dtype=bool) | current
        row_marks = _pad_rows(row_marks, row_count, trials) | current_e
    reject[field] = np.asarray(trial_marks, dtype=bool)
    reject[field_e] = _pad_rows(row_marks, row_count, trials)
    reject.setdefault("rejmanualcol", np.asarray(MANUAL_REJECTION_COLOR, dtype=float))


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
    row_marks = _pad_rows(np.asarray(reject.get(f"{prefix}{kind}E", []), dtype=bool), row_count, trials)
    return out, row_marks


def _pad_rows(values: np.ndarray, row_count: int, trials: int) -> np.ndarray:
    out = np.zeros((row_count, trials), dtype=bool)
    arr = np.asarray(values, dtype=bool)
    if arr.ndim == 1 and arr.size:
        arr = arr.reshape(1, -1)
    if arr.ndim == 2:
        rows = min(row_count, arr.shape[0])
        cols = min(trials, arr.shape[1])
        out[:rows, :cols] = arr[:rows, :cols]
    return out


def _row_count(EEG: dict[str, Any], icacomp: int) -> int:
    if int(bool(icacomp)):
        return int(EEG.get("nbchan", np.asarray(EEG.get("data")).shape[0]) or 0)
    weights = np.asarray(EEG.get("icaweights", []))
    return int(weights.shape[0]) if weights.ndim == 2 else 0


def _manual_color(EEG: dict[str, Any]) -> tuple[float, float, float]:
    reject = EEG.get("reject") or {}
    color = reject.get("rejmanualcol", MANUAL_REJECTION_COLOR)
    values = np.asarray(color if color is not None else DEFAULT_WINREJ_COLOR, dtype=float).ravel()
    if values.size < 3:
        return MANUAL_REJECTION_COLOR
    return tuple(float(item) for item in values[:3])


__all__ = ["apply_eegplot_rejections", "pop_eegplot"]
