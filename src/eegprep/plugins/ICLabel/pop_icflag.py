"""EEGLAB-style pop wrapper for ICLabel component flagging."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.plugins.ICLabel.eeg_icalabelstat import eeg_icalabelstat
from eegprep.plugins.ICLabel.eeg_icflag import eeg_icflag


ICLABEL_CLASSES = ("Brain", "Muscle", "Eye", "Heart", "Line Noise", "Channel Noise", "Other")
DEFAULT_ICFLAG_THRESHOLDS = np.array(
    [
        [np.nan, np.nan],
        [0.9, 1.0],
        [0.9, 1.0],
        [np.nan, np.nan],
        [np.nan, np.nan],
        [np.nan, np.nan],
        [np.nan, np.nan],
    ],
    dtype=float,
)


def pop_icflag(
    EEG: dict | list[dict],
    thresholds: Any = None,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
):
    """Flag ICLabel-classified components for later rejection.

    Args:
        EEG: EEG dictionary, or list of EEG dictionaries.
        thresholds: 7-by-2 threshold matrix in ICLabel class order.
        gui: Force or suppress the GUI threshold dialog.
        renderer: Optional GUI renderer used by tests.
        return_com: Return ``(EEG, command)`` when true.

    Returns:
        dict or tuple: Updated EEG, and optionally the history command.
    """
    if EEG is None:
        return (None, "") if return_com else None
    if gui is None:
        gui = thresholds is None
    if gui:
        if thresholds is None and not isinstance(EEG, list):
            _require_iclabel(EEG)
            eeg_icalabelstat(EEG)
        result = _run_gui(renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        thresholds = result["thresholds"]
    if thresholds is None:
        thresholds = DEFAULT_ICFLAG_THRESHOLDS
    thresholds = _normalize_thresholds(thresholds)

    if isinstance(EEG, list):
        output = [_flag_dataset(item, thresholds) for item in EEG]
        command = _history_command(thresholds)
        return (output, command) if return_com else output

    output = _flag_dataset(EEG, thresholds)
    command = _history_command(thresholds)
    return (output, command) if return_com else output


def _flag_dataset(EEG: dict, thresholds: np.ndarray) -> dict:
    _require_iclabel(EEG)
    return eeg_icflag(copy.deepcopy(EEG), thresholds)


def pop_icflag_dialog_spec() -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_icflag``."""
    controls: list[ControlSpec] = [
        ControlSpec("text", "Select range for flagging component for rejection", font_weight="bold"),
        ControlSpec("spacer"),
        ControlSpec("text", "Min"),
        ControlSpec("text", "Max"),
    ]
    geometry: list[tuple[float, ...] | tuple[int, ...]] = [(1,), (2.0, 0.4, 0.4)]
    for index, label in enumerate(ICLABEL_CLASSES):
        controls.extend(
            [
                ControlSpec("text", f'Probability range for "{label}"'),
                ControlSpec("edit", tag=f"min_{index}", value=_threshold_text(DEFAULT_ICFLAG_THRESHOLDS[index, 0])),
                ControlSpec("edit", tag=f"max_{index}", value=_threshold_text(DEFAULT_ICFLAG_THRESHOLDS[index, 1])),
            ]
        )
        geometry.append((2.0, 0.4, 0.4))

    return DialogSpec(
        title="Flag components using ICLabel -- pop_icflag()",
        function_name="pop_icflag",
        eeglab_source="plugins/ICLabel/pop_icflag.m",
        geometry=tuple(geometry),
        size=(467, 440),
        content_margins=(23, 37, 23, 13),
        row_spacing=16,
        controls=tuple(controls),
    )


def _run_gui(renderer: Any | None = None) -> dict[str, np.ndarray] | None:
    result = inputgui(pop_icflag_dialog_spec(), renderer=renderer)
    if result is None:
        return None
    thresholds = []
    for index in range(len(ICLABEL_CLASSES)):
        thresholds.append(
            [_parse_threshold(result.get(f"min_{index}", "")), _parse_threshold(result.get(f"max_{index}", ""))]
        )
    return {"thresholds": _normalize_thresholds(thresholds)}


def _parse_threshold(value: Any) -> float:
    text = str(value).strip()
    if not text:
        return float("nan")
    return float(text)


def _normalize_thresholds(thresholds: Any) -> np.ndarray:
    normalized = np.asarray(thresholds, dtype=float)
    if normalized.shape != (7, 2):
        raise ValueError("thresholds must be a 7x2 array")
    finite = normalized[np.isfinite(normalized)]
    if finite.size and ((finite < 0).any() or (finite > 1).any()):
        raise ValueError("ICLabel thresholds must be between 0 and 1")
    for row in normalized:
        if np.isfinite(row).all() and row[0] > row[1]:
            raise ValueError("ICLabel threshold minimum cannot exceed maximum")
    return normalized


def _require_iclabel(EEG: dict) -> None:
    try:
        classifications = EEG["etc"]["ic_classification"]["ICLabel"]["classifications"]
    except KeyError as exc:
        raise ValueError("No ICLabel classifications found. Run pop_iclabel first.") from exc
    if np.asarray(classifications).size == 0:
        raise ValueError("No ICLabel classifications found. Run pop_iclabel first.")


def _threshold_text(value: float) -> str:
    return "" if np.isnan(value) else f"{value:g}"


def _history_command(thresholds: np.ndarray) -> str:
    return f"EEG = pop_icflag(EEG, thresholds={_history_thresholds(thresholds)});"


def _history_thresholds(thresholds: np.ndarray) -> str:
    values = [
        [None if np.isnan(value) else float(value) for value in row]
        for row in np.asarray(thresholds, dtype=float).tolist()
    ]
    return format_history_value(values, cell_for_sequence=None)
