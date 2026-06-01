"""EEGLAB-style multi-epoch eegplot helper."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.sigprocfunc.eegplot import eegplot, trial2eegplot


NEW_REJECTION_COLOR = (0.8, 0.8, 1.0)
OLD_REJECTION_COLOR = (0.8, 1.0, 0.8)


def eeg_multieegplot(
    data: Any,
    rej: Any = None,
    rejE: Any = None,
    oldrej: Any = None,
    oldrejE: Any = None,
    *args: Any,
    oldcolor: Any = OLD_REJECTION_COLOR,
    newcolor: Any = NEW_REJECTION_COLOR,
    show: bool = True,
    **kwargs: Any,
) -> Any:
    """Open ``eegplot`` with current and previous rejection marks.

    ``rej``/``rejE`` and ``oldrej``/``oldrejE`` follow EEGLAB's convention:
    trial vectors are sweeps-length arrays of 0/1 marks and electrode marks
    are ``channels x sweeps`` boolean arrays. For continuous data, rejection
    rows may be ``[start end]`` or EEGLAB event-like rows whose start/end are
    in columns 3 and 4.
    """
    array = np.asarray(data, dtype=float)
    if array.ndim not in {2, 3}:
        raise ValueError("eeg_multieegplot data must be 2-D or 3-D channel-major data")
    n_channels = int(array.shape[0])
    pnts = int(array.shape[1])
    if array.ndim == 3:
        winrej = _epoched_winrej(
            rej,
            rejE,
            oldrej,
            oldrejE,
            n_channels=n_channels,
            pnts=pnts,
            trials=int(array.shape[2]),
            oldcolor=_rgb(oldcolor, "oldcolor"),
            newcolor=_rgb(newcolor, "newcolor"),
        )
    else:
        winrej = _continuous_winrej(
            rej,
            oldrej,
            n_channels=n_channels,
            oldcolor=_rgb(oldcolor, "oldcolor"),
            newcolor=_rgb(newcolor, "newcolor"),
        )
    if not _option_supplied(args, kwargs, "winlength"):
        kwargs["winlength"] = 5
    if not _option_supplied(args, kwargs, "winrej"):
        kwargs["winrej"] = winrej
    if not _option_supplied(args, kwargs, "xgrid"):
        kwargs["xgrid"] = "off"
    return eegplot(array, *args, show=show, **kwargs)


def _epoched_winrej(
    rej: Any,
    rejE: Any,
    oldrej: Any,
    oldrejE: Any,
    *,
    n_channels: int,
    pnts: int,
    trials: int,
    oldcolor: tuple[float, float, float],
    newcolor: tuple[float, float, float],
) -> np.ndarray:
    rows = []
    old_rows = trial2eegplot(
        _trial_marks(oldrej, trials),
        _electrode_marks(oldrejE, n_channels, trials),
        pnts,
        oldcolor,
    )
    if old_rows.size:
        rows.append(old_rows)
    new_rows = trial2eegplot(
        _trial_marks(rej, trials),
        _electrode_marks(rejE, n_channels, trials),
        pnts,
        newcolor,
    )
    if new_rows.size:
        rows.append(new_rows)
    return np.vstack(rows) if rows else np.zeros((0, 5 + n_channels), dtype=float)


def _continuous_winrej(
    rej: Any,
    oldrej: Any,
    *,
    n_channels: int,
    oldcolor: tuple[float, float, float],
    newcolor: tuple[float, float, float],
) -> np.ndarray:
    rows = []
    new_rows = _continuous_rows(rej, n_channels, newcolor)
    if new_rows.size:
        rows.append(new_rows)
    old_rows = _continuous_rows(oldrej, n_channels, oldcolor)
    if old_rows.size:
        rows.append(old_rows)
    return np.vstack(rows) if rows else np.zeros((0, 5 + n_channels), dtype=float)


def _continuous_rows(
    value: Any,
    n_channels: int,
    color: tuple[float, float, float],
) -> np.ndarray:
    rows = np.asarray(value if value is not None else [], dtype=float)
    if rows.size == 0:
        return np.zeros((0, 5 + n_channels), dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] < 2:
        raise ValueError("continuous rejection rows must contain at least start and end columns")
    start_end = rows[:, 2:4] if rows.shape[1] >= 4 else rows[:, 0:2]
    out = np.zeros((rows.shape[0], 5 + n_channels), dtype=float)
    out[:, 0:2] = start_end
    out[:, 2:5] = np.asarray(color, dtype=float)
    return out


def _trial_marks(value: Any, trials: int) -> np.ndarray:
    marks = np.zeros(int(trials), dtype=bool)
    arr = np.asarray(value if value is not None else [], dtype=float).ravel()
    if arr.size == 0:
        return marks
    if arr.size == trials:
        return arr > 0
    for trial in arr.astype(int):
        if 1 <= trial <= trials:
            marks[trial - 1] = True
    return marks


def _electrode_marks(value: Any, n_channels: int, trials: int) -> np.ndarray:
    out = np.zeros((int(n_channels), int(trials)), dtype=bool)
    arr = np.asarray(value if value is not None else [], dtype=bool)
    if arr.size == 0:
        return out
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("electrode rejection marks must be a 2-D channels x trials array")
    rows = min(out.shape[0], arr.shape[0])
    cols = min(out.shape[1], arr.shape[1])
    out[:rows, :cols] = arr[:rows, :cols]
    return out


def _option_supplied(args: tuple[Any, ...], kwargs: dict[str, Any], name: str) -> bool:
    if name in kwargs:
        return True
    return any(isinstance(args[index], str) and args[index] == name for index in range(0, len(args), 2))


def _rgb(value: Any, name: str) -> tuple[float, float, float]:
    values = tuple(float(item) for item in value)
    if len(values) != 3 or any(item < 0.0 or item > 1.0 for item in values):
        raise ValueError(f"{name} must contain three RGB values between 0 and 1")
    return values


__all__ = ["eeg_multieegplot"]
