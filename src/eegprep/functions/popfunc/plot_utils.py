"""Shared helpers for EEGLAB-style plotting pop functions."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.miscfunc.misc import finite_matmul
from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc._pop_utils import is_empty_value, parse_numeric_sequence

# matplotlib's file-output backends cannot open a figure window.
_NONINTERACTIVE_BACKENDS = frozenset({"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"})


def backend_can_display() -> bool:
    """Return True when the active matplotlib backend can show a figure window."""
    return plt.get_backend().lower() not in _NONINTERACTIVE_BACKENDS


def show_figures(figures: Any, *, plot: str = "on") -> None:
    """Pop up figure windows on an interactive backend, like EEGLAB plots.

    Accepts a single figure, a list/tuple of figures, or ``None``. It is a no-op
    on file-output backends (Agg, PDF, ...) so headless rendering and tests stay
    silent. Displaying here, rather than in the noninteractive sigproc cores,
    keeps those cores test-safe while every plotting ``pop_*`` wrapper shows its
    figure from both the GUI and the console.

    Pass ``plot="off"`` to suppress the window entirely; callers still receive
    the built figure so they can save or embed it without a pop-up.
    """
    if str(plot).lower() == "off":
        # Interactive sessions (e.g. IPython's ``%matplotlib qt``) auto-display
        # every pyplot-managed figure, so suppressing the window means removing
        # the figure from pyplot's registry, not just skipping ``show()``. The
        # returned Figure object stays usable for ``savefig``/embedding.
        for figure in _as_figure_list(figures):
            plt.close(figure)
        return
    if not backend_can_display():
        return
    for figure in _as_figure_list(figures):
        figure.show()


def as_eeg_list(value: Any) -> list[dict[str, Any]]:
    """Return one or more EEG dictionaries as a list."""
    if isinstance(value, list):
        if not all(isinstance(item, dict) for item in value):
            raise ValueError("Expected EEG dataset dictionaries")
        return value
    if not isinstance(value, dict):
        raise ValueError("Expected an EEG dataset")
    return [value]


def eeg_data_array(EEG: dict[str, Any]) -> np.ndarray:
    """Return channel-major EEG data as a float array."""
    data = np.asarray(EEG.get("data"), dtype=float)
    if data.ndim not in {2, 3}:
        raise ValueError("EEG.data must be a 2-D or 3-D channel-major array")
    return data


def eeg_epoch_data(EEG: dict[str, Any]) -> np.ndarray:
    """Return EEG data as ``channels x points x trials``."""
    data = eeg_data_array(EEG)
    if data.ndim == 2:
        return data[:, :, np.newaxis]
    return data


def eeg_times_ms(EEG: dict[str, Any]) -> np.ndarray:
    """Return one epoch worth of time values in milliseconds."""
    pnts = int(EEG.get("pnts", 0) or 0)
    if pnts <= 0:
        raise ValueError("EEG.pnts must be positive")
    times = np.asarray(EEG.get("times", []), dtype=float).ravel()
    if times.size == pnts:
        return times
    xmin = float(EEG.get("xmin", 0) or 0)
    xmax = float(EEG.get("xmax", xmin) or xmin)
    return np.linspace(xmin * 1000.0, xmax * 1000.0, pnts)


def data_time_slice(EEG: dict[str, Any], timerange: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """Return epoch data and time values restricted to ``timerange`` in ms."""
    data = eeg_epoch_data(EEG)
    times = eeg_times_ms(EEG)
    if timerange is None or _is_empty_sequence(timerange):
        return data, times
    bounds = numeric_vector(timerange)
    if bounds.size != 2:
        raise ValueError("timerange must contain [min max] in milliseconds")
    mask = (times >= bounds[0]) & (times <= bounds[1])
    if not np.any(mask):
        raise ValueError("timerange does not contain any samples")
    return data[:, mask, :], times[mask]


def channel_labels(EEG: dict[str, Any]) -> list[str]:
    """Return channel labels with EEGLAB-style numeric fallbacks."""
    labels = []
    for index, chanloc in enumerate(chanlocs_as_list(EEG.get("chanlocs", [])), start=1):
        label = str(chanloc.get("labels") or "").strip()
        labels.append(label or str(index))
    nbchan = int(EEG.get("nbchan", len(labels)) or len(labels))
    while len(labels) < nbchan:
        labels.append(str(len(labels) + 1))
    return labels


def selected_indices(values: Any, maximum: int, *, default_all: bool = True) -> np.ndarray:
    """Normalize EEGLAB-facing 1-based indices to Python 0-based indices."""
    if values is None or _is_empty_sequence(values):
        if default_all:
            return np.arange(maximum, dtype=int)
        raise ValueError("At least one index is required")
    numeric = numeric_vector(values, dtype=float)
    if numeric.size == 0 and default_all:
        return np.arange(maximum, dtype=int)
    indices = numeric.astype(int)
    if np.any(indices != numeric):
        raise ValueError("Indices must be integers")
    if np.any(indices < 1) or np.any(indices > maximum):
        raise ValueError(f"Indices must be 1-based and within 1..{maximum}")
    return indices - 1


def component_activations(EEG: dict[str, Any], *, use_stored: bool = True) -> np.ndarray:
    """Return ICA activations as ``components x points x trials``.

    When ``use_stored`` is true, a non-empty ``EEG['icaact']`` is returned as-is.
    Rejection scoring passes ``use_stored=False`` to always recompute from the
    weight and sphere matrices, matching EEGLAB's rejection path.
    """
    icaact = EEG.get("icaact")
    if use_stored and icaact is not None and np.asarray(icaact).size:
        data = np.asarray(icaact, dtype=float)
        if data.ndim == 2:
            return data[:, :, np.newaxis]
        if data.ndim == 3:
            return data
    weights = np.asarray(EEG.get("icaweights", []), dtype=float)
    sphere = np.asarray(EEG.get("icasphere", []), dtype=float)
    if weights.size == 0 or sphere.size == 0:
        raise ValueError("no ICA activations or weights for this dataset")
    data = eeg_epoch_data(EEG)
    icachansind = component_channel_indices(EEG, data.shape[0])
    flat = data[icachansind, :, :].reshape(icachansind.size, -1)
    unmixing = finite_matmul(weights, sphere)
    if unmixing.shape[1] != flat.shape[0]:
        raise ValueError("ICA weights do not match EEG.icachansind channel count")
    acts = finite_matmul(unmixing, flat)
    return acts.reshape(weights.shape[0], data.shape[1], data.shape[2])


def component_channel_indices(EEG: dict[str, Any], channel_count: int) -> np.ndarray:
    """Return 0-based channel indices used by ICA."""
    value = EEG.get("icachansind")
    if _is_empty_sequence(value):
        return np.arange(channel_count, dtype=int)
    indices = np.asarray(value, dtype=float).ravel()
    if np.any(indices != indices.astype(int)):
        raise ValueError("EEG.icachansind must contain integer channel indices")
    indices = indices.astype(int)
    if np.any(indices < 0) or np.any(indices >= channel_count):
        raise ValueError(f"EEG.icachansind values must be 0-based and within 0..{channel_count - 1}")
    return indices


def component_maps(EEG: dict[str, Any]) -> np.ndarray:
    """Return ICA inverse maps as ``channels x components``."""
    icawinv = np.asarray(EEG.get("icawinv", []), dtype=float)
    if icawinv.size == 0:
        raise ValueError("no ICA maps for this dataset")
    if icawinv.ndim != 2:
        raise ValueError("EEG.icawinv must be 2-D")
    return icawinv


def component_map_data(EEG: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return ICA maps and matching channel locations for plotting."""
    maps = component_maps(EEG)
    chanlocs = chanlocs_as_list(EEG.get("chanlocs", []))
    if maps.shape[0] == len(chanlocs):
        return maps, chanlocs
    indices = component_channel_indices(EEG, int(EEG.get("nbchan", len(chanlocs)) or len(chanlocs)))
    if maps.shape[0] == indices.size and chanlocs and np.max(indices) < len(chanlocs):
        return maps, [chanlocs[index] for index in indices]
    raise ValueError("EEG.icawinv must have one row per channel location or ICA channel")


def numeric_vector(value: Any, *, dtype: Any = float) -> np.ndarray:
    """Parse EEGLAB-style numeric vectors from strings, lists, or arrays."""
    if isinstance(value, np.ndarray):
        return value.astype(dtype).ravel()
    return np.asarray(parse_numeric_sequence(value, dtype=dtype), dtype=dtype)


def colon_sequence(token: str) -> list[float]:
    """Parse MATLAB ``start:stop`` or ``start:step:stop`` tokens."""
    return parse_numeric_sequence(token, dtype=float)


def python_literal(value: Any) -> str:
    """Return a pasteable Python literal for EEGPrep console history."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value):
            return "float('nan')"
        if np.isposinf(value):
            return "float('inf')"
        if np.isneginf(value):
            return "float('-inf')"
        if value.is_integer():
            return str(int(value))
    if isinstance(value, list):
        return "[" + ", ".join(python_literal(item) for item in value) + "]"
    if isinstance(value, tuple):
        return "(" + ", ".join(python_literal(item) for item in value) + ("," if len(value) == 1 else "") + ")"
    return repr(value)


def parse_plot_options_text(text: Any) -> dict[str, Any]:
    """Parse plot option text as either Python ``key=value`` or ``'key', value`` pairs.

    Both styles may appear in a single string, separated by top-level commas
    (e.g. ``electrodes='off', 'style', 'blank'``).
    """
    stripped = str(text or "").strip()
    if not stripped or stripped in {"[]", "{}"}:
        return {}
    parts = _split_top_level_commas(stripped)
    options: dict[str, Any] = {}
    index = 0
    while index < len(parts):
        part = parts[index]
        eq_pos = _find_top_level_equals(part)
        if eq_pos is not None:
            key_text = part[:eq_pos].strip()
            value_text = part[eq_pos + 1 :].strip()
            if not key_text.isidentifier():
                raise ValueError(f"Plot option key must be a Python identifier: {key_text!r}")
            options[key_text.lower()] = _parse_option_atom(value_text)
            index += 1
            continue
        if index + 1 >= len(parts):
            raise ValueError("Plot options must be key/value pairs")
        key = _parse_option_atom(part)
        if not isinstance(key, str):
            raise ValueError("Plot option keys must be strings")
        options[key.lower()] = _parse_option_atom(parts[index + 1])
        index += 2
    return options


def history_command(function_name: str, *args: Any, eeg_name: str = "EEG", **kwargs: Any) -> str:
    """Build a valid Python console command for a plotting pop function."""
    pieces = [eeg_name]
    pieces.extend(python_literal(arg) for arg in args)
    pieces.extend(
        f"{key}={python_literal(value)}" for key, value in kwargs.items() if not _is_empty_history_value(value)
    )
    return f"{function_name}({', '.join(pieces)})"


def _as_figure_list(figures: Any) -> list[Any]:
    if figures is None:
        return []
    if isinstance(figures, (list, tuple)):
        return [figure for figure in figures if figure is not None]
    return [figures]


def _is_empty_sequence(value: Any) -> bool:
    return is_empty_value(value)


def _is_empty_history_value(value: Any) -> bool:
    return value is None or _is_empty_sequence(value)


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    bracket_depth = 0
    for char in text:
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char in "[({":
            bracket_depth += 1
        elif char in "])}" and bracket_depth:
            bracket_depth -= 1
        if char == "," and bracket_depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if quote is not None:
        raise ValueError("Unterminated quoted plot option")
    if bracket_depth:
        raise ValueError("Unbalanced bracket in plot options")
    parts.append("".join(current).strip())
    return [part for part in parts if part]


def _find_top_level_equals(text: str) -> int | None:
    quote: str | None = None
    bracket_depth = 0
    for index, char in enumerate(text):
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in "[({":
            bracket_depth += 1
        elif char in "])}" and bracket_depth:
            bracket_depth -= 1
        if char == "=" and bracket_depth == 0:
            return index
    return None


def _parse_option_atom(text: str) -> Any:
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    if len(stripped) >= 2 and stripped[0] == "[" and stripped[-1] == "]":
        return numeric_vector(stripped).tolist()
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    try:
        value = float(stripped)
    except ValueError:
        return stripped
    return int(value) if value.is_integer() else value
