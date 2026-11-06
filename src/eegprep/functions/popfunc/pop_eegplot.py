"""EEGLAB-style wrapper for the scrolling EEG browser."""

from __future__ import annotations

from typing import Any

from eegprep.functions.adminfunc.vararg2str import vararg2str
from eegprep.functions.popfunc.plot_utils import history_command
import numpy as np

from eegprep.functions.popfunc._eegplot_rejection import (
    DEFAULT_REJECTION_COLORS,
    MANUAL_REJECTION_COLOR,
    displayed_rejection_families,
    manual_rejection_color,
    pad_rejection_rows,
    rejection_family_color,
    rejection_row_count,
)
from eegprep.functions.popfunc._rejection import copy_eeg
from eegprep.functions.popfunc.eeg_eegrej import eeg_eegrej
from eegprep.functions.popfunc.pop_rejepoch import pop_rejepoch
from eegprep.functions.sigprocfunc.eegplot import (
    eegplot,
    eegplot2event,
    eegplot2trial,
    normalize_winrej,
    trial2eegplot,
    winrej_to_array,
)


CONTINUOUS_MANUAL_WINREJ = "rejmanualwinrej"
CONTINUOUS_ICA_MANUAL_WINREJ = "icarejmanualwinrej"
_REJECT_OMITTED = object()
_CONTINUOUS_REJECT_INSTRUCTIONS = (
    "Mark stretches of continuous data for rejection\n"
    "by dragging the left mouse button. Click on marked\n"
    'stretches to unmark. When done,press "REJECT" to\n'
    "excise marked stretches (Note: Leaves rejection\n"
    "boundary markers in the event table)."
)
_EPOCH_REJECT_QUESTION = "Are you sure, you want to reject the labeled trials ?"
_EPOCH_REJECT_TITLE = "Reject pre-labelled epochs -- pop_rejepoch()"


def pop_eegplot(
    EEG: dict[str, Any] | None = None,
    icacomp: int = 1,
    superpose: int = 0,
    reject: Any = _REJECT_OMITTED,
    topcommand: Any = None,
    *args: Any,
    return_com: bool = False,
    show: bool = True,
    **kwargs: Any,
):
    """Inspect EEG channel or component activity in the scrolling browser.

    The browser edits marks in window-local state. Data or rejection fields are
    changed only when the Accept/Reject callback runs. Omitting ``reject`` on
    continuous data shows the rejection instruction dialog before the browser
    opens. Pass ``reject`` explicitly, or ``show=False``, to skip that dialog.
    """
    del topcommand
    if EEG is None:
        raise ValueError("pop_eegplot requires a non-empty EEG dataset")
    reject_omitted = reject is _REJECT_OMITTED
    if reject_omitted:
        reject = 1
    reject = int(reject)
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
    options.setdefault("wincolor", manual_rejection_color(EEG))
    options.setdefault("butlabel", "Reject" if int(bool(reject)) else "Update Marks")
    trials = int(EEG.get("trials", 1) or 1)
    if trials > 1:
        options.setdefault("winrej", _initial_epoch_winrej(EEG, icacomp, superpose))
    else:
        options.setdefault("winrej", _initial_continuous_winrej(EEG, icacomp))

    def _accept(winrej: np.ndarray) -> None:
        if _epoch_reject_pending(EEG, winrej) and not _confirm_epoch_rejection():
            return
        eeg_out, accept_command = apply_eegplot_rejections(
            EEG,
            winrej,
            icacomp=icacomp,
            superpose=superpose,
            reject=reject,
            return_com=True,
        )
        if accept_callback is not None and accept_command:
            accept_callback(eeg_out, accept_command)

    if return_com and show and accept_callback is None:
        raise ValueError(
            "pop_eegplot(return_com=True) opens a non-blocking browser; pass command_callback to receive "
            "accepted browser results, or use apply_eegplot_rejections for programmatic rejection."
        )
    if show and reject_omitted and reject and trials <= 1 and not _confirm_continuous_reject_instructions():
        return (None, "") if return_com else None
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
            regions = eegplot2event(rows, -1) if rows.size else np.zeros((0, 0))
            if regions.size:
                out = eeg_eegrej(out, regions)
                _clear_continuous_marks(out, icacomp=icacomp)
                command = f"EEG = eeg_eegrej( EEG, {vararg2str([regions])});"
            else:
                command = ""
        else:
            _store_continuous_marks(out, rows, icacomp=icacomp)
        return (out, command) if return_com else out

    pnts = int(out.get("pnts", np.asarray(out.get("data")).shape[1]))
    if rows.size:
        trial_marks, row_marks = eegplot2trial(rows, pnts, trials, manual_rejection_color(out), None)
        store_superpose = superpose
    else:
        trial_marks = np.zeros(trials, dtype=bool)
        row_marks = np.zeros((rejection_row_count(out, icacomp), trials), dtype=bool)
        store_superpose = 0
    _store_epoch_marks(out, trial_marks, row_marks, icacomp=icacomp, superpose=store_superpose)
    if int(bool(reject)):
        command = _epoch_reject_command(trial_marks)
        if trial_marks.any():
            out, _reject_command = pop_rejepoch(out, trial_marks, 1, return_com=True)
    return (out, command) if return_com else out


def _epoch_reject_command(trial_marks: np.ndarray) -> str:
    indices = np.flatnonzero(np.asarray(trial_marks, dtype=bool)) + 1
    if indices.size == 0:
        return ""
    return f"EEG = pop_rejepoch( EEG, {vararg2str([indices, 0])});"


def _epoch_reject_pending(EEG: dict[str, Any], winrej: Any) -> bool:
    if int(EEG.get("trials", 1) or 1) <= 1:
        return False
    rows = _as_winrej_rows(winrej)
    if rows.size == 0:
        return False
    trials = int(EEG.get("trials", 1) or 1)
    pnts = int(EEG.get("pnts", np.asarray(EEG.get("data")).shape[1]))
    trial_marks, _row_marks = eegplot2trial(rows, pnts, trials, manual_rejection_color(EEG), None)
    return bool(np.asarray(trial_marks).any())


def _confirm_continuous_reject_instructions() -> bool:
    return _ask_dialog("Warning", _CONTINUOUS_REJECT_INSTRUCTIONS, ("Cancel", "Continue"), "Continue") == "Continue"


def _confirm_epoch_rejection() -> bool:
    return _ask_dialog("Question", _EPOCH_REJECT_QUESTION, ("NO", "YES"), "YES", title=_EPOCH_REJECT_TITLE) == "YES"


def _ask_dialog(icon_name: str, text: str, labels: tuple[str, str], default: str, *, title: str | None = None) -> str:
    try:
        from PySide6 import QtCore, QtWidgets
    except ImportError as exc:
        raise RuntimeError(
            "pop_eegplot could not open its dialog because PySide6 is not installed. "
            "Install it with `pip install eegprep[gui]`."
        ) from exc
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle(title or icon_name)
    dialog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
    layout = QtWidgets.QVBoxLayout(dialog)
    message = QtWidgets.QLabel(text, dialog)
    message.setObjectName("eegplot_ask_text")
    message.setTextFormat(QtCore.Qt.TextFormat.PlainText)
    layout.addWidget(message)
    clicked: dict[str, str] = {"label": ""}
    button_row = QtWidgets.QHBoxLayout()
    button_row.addStretch(1)
    button_by_label: dict[str, QtWidgets.QPushButton] = {}
    for label in labels:
        button = QtWidgets.QPushButton(label, dialog)
        button_by_label[label] = button
        button_row.addWidget(button)

        def choose(_checked: bool = False, label: str = label) -> None:
            clicked["label"] = label
            dialog.accept()

        button.clicked.connect(choose)
    layout.addLayout(button_row)
    button_by_label[default].setDefault(True)
    dialog.exec()
    return clicked["label"]


def _initial_epoch_winrej(EEG: dict[str, Any], icacomp: int, superpose: int) -> np.ndarray:
    reject = EEG.get("reject") or {}
    trials = int(EEG.get("trials", 1) or 1)
    pnts = int(EEG.get("pnts", np.asarray(EEG.get("data")).shape[1]))
    row_count = rejection_row_count(EEG, icacomp)
    manual, manual_e = _reject_arrays(reject, "rejmanual", trials, row_count, icacomp=icacomp)
    if int(superpose) == 0:
        return trial2eegplot(manual, manual_e, pnts, manual_rejection_color(EEG))
    if int(superpose) == 2:
        return _superposed_epoch_winrej(EEG, reject, icacomp, trials, row_count, pnts, manual, manual_e)
    rows = []
    old_color = tuple(min(component + 0.15, 1.0) for component in manual_rejection_color(EEG))
    old, old_e = _reject_arrays(reject, "rejglobal", trials, row_count, icacomp=1)
    old_rows = trial2eegplot(old, old_e, pnts, old_color)
    if old_rows.size:
        rows.append(old_rows)
    manual_rows = trial2eegplot(manual, manual_e, pnts, manual_rejection_color(EEG))
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
    manual_color = manual_rejection_color(EEG)
    for family in displayed_rejection_families(reject):
        color = rejection_family_color(reject, family, DEFAULT_REJECTION_COLORS.get(family, manual_color))
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


__all__ = ["apply_eegplot_rejections", "eegplot_accept_creates_dataset", "pop_eegplot"]
