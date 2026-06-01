"""EEGLAB-style component removal pop function."""

from __future__ import annotations

import copy
import logging
import re
from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.miscfunc.misc import finite_matmul, finite_pinv
from eegprep.functions.popfunc._pop_utils import format_history_value, is_on as _is_on
from eegprep.functions.sigprocfunc.eegplot import eegplot

logger = logging.getLogger(__name__)


def pop_subcomp(
    EEG: dict | list[dict],
    components: Any = None,
    plotag: int | bool = 0,
    keepcomp: int | bool = 0,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
):
    """Remove ICA components from data.

    Component indices are 1-based to match EEGLAB. Passing ``components=None``
    or ``components=[]`` removes components already flagged in
    ``EEG["reject"]["gcompreject"]``. ``keepcomp=1`` keeps the supplied
    components and removes all others.
    """
    if EEG is None:
        return (None, "") if return_com else None
    if gui is None:
        gui = components is None
    if gui:
        result = _run_gui(EEG, renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        components = result["components"]
        keepcomp = result["keepcomp"]

    if isinstance(EEG, list):
        results = [pop_subcomp(item, components, plotag, keepcomp, gui=False, return_com=True) for item in EEG]
        outputs = [result[0] for result in results]
        command = _history_command(components, plotag, keepcomp) if any(result[1] for result in results) else ""
        return (outputs, command) if return_com else outputs

    output, removed = _remove_components(EEG, components, keepcomp)
    if removed and _is_on(plotag):
        _plot_subcomp_confirmation(EEG, output)
    command = _history_command(components if components is not None else [], plotag, keepcomp) if removed else ""
    return (output, command) if return_com else output


def pop_subcomp_dialog_spec(EEG: dict | list[dict]) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_subcomp``."""
    default_components = _default_component_text(EEG)
    note = "Note: for group level analysis, remove components in STUDY"
    remove_label = "List of component(s) to remove from data"
    if isinstance(EEG, list):
        note = "Leave blank to remove flagged components in each selected dataset"
    return DialogSpec(
        title="Remove components from data -- pop_subcomp()",
        function_name="pop_subcomp",
        eeglab_source="functions/popfunc/pop_subcomp.m",
        geometry=((1,), (3.2, 1.0), (3.2, 1.0)),
        size=(538, 231),
        help_text="pophelp('pop_subcomp')",
        content_margins=(27, 37, 27, 13),
        row_spacing=18,
        controls=(
            ControlSpec("text", note),
            ControlSpec("text", remove_label),
            ControlSpec("edit", tag="remove", value=default_components),
            ControlSpec("text", "Or list of component(s) to retain"),
            ControlSpec("edit", tag="retain", value=""),
        ),
    )


def _run_gui(EEG: dict | list[dict], renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_subcomp_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    retain = _parse_components_text(result.get("retain", ""))
    if retain is not None:
        return {"components": retain, "keepcomp": 1}
    return {"components": _parse_components_text(result.get("remove", "")), "keepcomp": 0}


def _remove_components(EEG: dict, components: Any, keepcomp: int | bool) -> tuple[dict, bool]:
    _require_ica(EEG)
    output = copy.deepcopy(EEG)
    n_comps = int(np.asarray(output["icaweights"]).shape[0])
    remove_components = _components_to_remove(output, components, keepcomp, n_comps)
    if not remove_components:
        logger.warning("No components specified, no rejection performed")
        return output, False

    zero_based = np.asarray([component - 1 for component in remove_components], dtype=int)
    keep_mask = np.ones(n_comps, dtype=bool)
    keep_mask[zero_based] = False
    icachansind = _icachansind(output)
    data = np.asarray(output["data"])
    data_shape = data.shape
    data_2d = data.reshape(data_shape[0], -1, order="F")

    weights = np.asarray(output["icaweights"], dtype=float)
    sphere = np.asarray(output["icasphere"], dtype=float)
    unmixing = finite_matmul(weights, sphere)
    activations = finite_matmul(unmixing, data_2d[icachansind])
    icawinv = output.get("icawinv")
    mixing = np.asarray(icawinv, dtype=float) if icawinv is not None else np.array([])
    if mixing.size == 0:
        mixing = finite_pinv(unmixing)
    projected = finite_matmul(mixing[:, keep_mask], activations[keep_mask])

    cleaned = data_2d.copy()
    cleaned[icachansind] = projected
    output["data"] = cleaned.reshape(data_shape, order="F")
    output["setname"] = f"{output.get('setname', '')} pruned with ICA".strip()
    output["icaact"] = np.array([])
    output["icaweights"] = weights[keep_mask, :]
    output["icawinv"] = mixing[:, keep_mask]
    output["specicaact"] = np.array([])
    output["specdata"] = np.array([])
    output["reject"] = {}
    _trim_iclabel_classifications(output, keep_mask)
    _trim_dipfit_model(output, keep_mask)
    return output, True


def _plot_subcomp_confirmation(input_eeg: dict, output_eeg: dict) -> None:
    icachansind = _icachansind(input_eeg)
    input_data = np.asarray(input_eeg["data"])
    output_data = np.asarray(output_eeg["data"])
    preview_data = np.array(input_data[icachansind], dtype=float, copy=True)
    preview = {
        "data": preview_data,
        "nbchan": int(icachansind.size),
        "pnts": int(input_eeg.get("pnts", preview_data.shape[1]) or preview_data.shape[1]),
        "trials": int(input_eeg.get("trials", preview_data.shape[2] if preview_data.ndim == 3 else 1) or 1),
        "srate": float(input_eeg.get("srate", 256) or 256),
        "xmin": float(input_eeg.get("xmin", 0.0) or 0.0),
        "xmax": float(input_eeg.get("xmax", 0.0) or 0.0),
        "chanlocs": _selected_chanlocs(input_eeg, icachansind),
        "event": input_eeg.get("event", []),
        "setname": input_eeg.get("setname", ""),
    }
    eegplot(
        preview,
        srate=preview["srate"],
        title="Black = channel before rejection; red = after rejection -- eegplot()",
        limits=[preview["xmin"] * 1000.0, preview["xmax"] * 1000.0],
        data2=np.array(output_data[icachansind], dtype=float, copy=True),
    )


def _selected_chanlocs(EEG: dict, indices: np.ndarray) -> list[Any]:
    chanlocs = EEG.get("chanlocs", [])
    if chanlocs is None:
        chanlocs = []
    if isinstance(chanlocs, np.ndarray):
        chanlocs = chanlocs.tolist()
    if not isinstance(chanlocs, (list, tuple)):
        return []
    return [copy.deepcopy(chanlocs[int(index)]) for index in indices if 0 <= int(index) < len(chanlocs)]


def _components_to_remove(EEG: dict, components: Any, keepcomp: int | bool, n_comps: int) -> list[int]:
    parsed = _normalize_components(components)
    if parsed is None:
        parsed = _flagged_components(EEG)
    elif _is_on(keepcomp):
        parsed = [component for component in range(1, n_comps + 1) if component not in set(parsed)]
    if not parsed:
        return []
    if min(parsed) < 1 or max(parsed) > n_comps:
        raise ValueError("Component index out of range")
    return _unique_preserve_order(parsed)


def _normalize_components(components: Any) -> list[int] | None:
    if components is None:
        return None
    if isinstance(components, str):
        return _parse_components_text(components)
    values = np.asarray(components).astype(int).ravel().tolist()
    return values or None


def _parse_components_text(text: Any) -> list[int] | None:
    values = [value for value in re.split(r"[\s,]+", str(text).strip().strip("[]")) if value]
    if not values:
        return None
    parsed: list[int] = []
    for value in values:
        parsed.extend(_parse_component_token(value))
    return parsed


def _parse_component_token(value: str) -> list[int]:
    if ":" not in value:
        return [int(value)]
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        start, stop = parts
        step = 1
    elif len(parts) == 3:
        start, step, stop = parts
    else:
        raise ValueError("Component ranges must use start:stop or start:step:stop")
    if step == 0:
        raise ValueError("Component range step cannot be zero")
    if (step > 0 and start > stop) or (step < 0 and start < stop):
        return []
    return list(range(start, stop + (1 if step > 0 else -1), step))


def _flagged_components(EEG: dict) -> list[int]:
    flags = (EEG.get("reject") or {}).get("gcompreject", [])
    flags = np.asarray(flags, dtype=bool).ravel()
    return [index + 1 for index, flag in enumerate(flags) if flag]


def _default_component_text(EEG: dict | list[dict]) -> str:
    if isinstance(EEG, list):
        return ""
    return " ".join(str(component) for component in _flagged_components(EEG))


def _icachansind(EEG: dict) -> np.ndarray:
    indices = EEG.get("icachansind")
    if indices is None or np.asarray(indices).size == 0:
        return np.arange(int(EEG["nbchan"]), dtype=int)
    return np.asarray(indices, dtype=int).ravel()


def _require_ica(EEG: dict) -> None:
    weights = EEG.get("icaweights")
    sphere = EEG.get("icasphere")
    if weights is None or sphere is None or np.asarray(weights).size == 0 or np.asarray(sphere).size == 0:
        raise ValueError("No ICA decomposition found. Run pop_runica first.")


def _trim_iclabel_classifications(EEG: dict, keep_mask: np.ndarray) -> None:
    try:
        iclabel = EEG["etc"]["ic_classification"]["ICLabel"]
        classifications = iclabel["classifications"]
    except KeyError:
        return
    if np.asarray(classifications).size:
        iclabel["classifications"] = np.asarray(classifications)[keep_mask, :]


def _trim_dipfit_model(EEG: dict, keep_mask: np.ndarray) -> None:
    dipfit = EEG.get("dipfit")
    if not isinstance(dipfit, dict) or "model" not in dipfit:
        return
    model_value = dipfit["model"]
    if not isinstance(model_value, (list, tuple, np.ndarray)):
        return
    model = np.asarray(model_value, dtype=object)
    if model.size and model.shape[0] == keep_mask.size:
        dipfit["model"] = model[keep_mask].tolist()


def _unique_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _history_command(components: Any, plotag: int | bool, keepcomp: int | bool) -> str:
    parts = [
        "EEG = pop_subcomp(EEG, ",
        format_history_value(_normalize_components(components) or [], cell_for_sequence=None),
        f", {int(bool(plotag))}",
    ]
    if _is_on(keepcomp):
        parts.append(", 1")
    parts.append(");")
    return "".join(parts)
