"""EEGLAB-style pop wrapper for ICA decomposition."""

from __future__ import annotations

import copy
import re
from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args
from eegprep.functions.popfunc.eeg_amica import eeg_amica
from eegprep.functions.popfunc.eeg_decodechan import eeg_decodechan
from eegprep.functions.popfunc.eeg_picard import eeg_picard
from eegprep.functions.popfunc.eeg_runica import eeg_runica


_ALGORITHMS = (
    ("runica", "Extended Infomax (runica.m; default)", "'extended', 1"),
    ("runica", "Robust Extended Infomax (runica.m; slow)", "'extended', 1, 'lrate', 1e-5, 'maxsteps', 2000"),
    ("runamica15", "AMICA (slowest; best)", "'maxiter', 2000"),
    ("picard", "Infomax picard.m", "'maxiter', 500, 'mode', 'standard'"),
    ("picard", "FastICA picard.m (fastest)", "'maxiter', 500"),
)


def pop_runica(
    EEG,
    *args,
    icatype: str = "runica",
    options: Any = None,
    reorder: str | bool = "on",
    chanind: Any = None,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs,
):
    """Run ICA decomposition with EEGLAB ``pop_runica`` calling semantics."""
    parsed = _parse_runica_args(args, kwargs)
    icatype = _normalise_icatype(parsed.pop("icatype", icatype))
    options = parsed.pop("options", options)
    reorder = parsed.pop("reorder", reorder)
    chanind = parsed.pop("chanind", chanind)
    dataset = parsed.pop("dataset", None)
    concatenate = parsed.pop("concatenate", "off")
    concatcond = parsed.pop("concatcond", "off")

    if gui is None:
        gui = options is None and not parsed and chanind is None and dataset is None
    if gui:
        gui_result = _run_gui(EEG, renderer=renderer)
        if gui_result is None:
            return (EEG, "") if return_com else EEG
        icatype = gui_result["icatype"]
        options = gui_result["options"]
        reorder = gui_result["reorder"]
        chanind = gui_result["chanind"]
        dataset = gui_result["dataset"]
        concatenate = gui_result["concatenate"]
        concatcond = gui_result["concatcond"]
        if icatype == "runica":
            options = dict(options)
            options.setdefault("interrupt", "on")

    ica_options = _normalise_ica_options(icatype, options, parsed)
    if isinstance(EEG, list):
        output = _runica_on_datasets(
            EEG,
            dataset=dataset,
            icatype=icatype,
            options=ica_options,
            reorder=reorder,
            chanind=chanind,
            concatenate=concatenate,
            concatcond=concatcond,
        )
        command = _history_command(
            icatype,
            ica_options,
            reorder,
            chanind,
            dataset=dataset,
            concatenate=concatenate,
            concatcond=concatcond,
        )
        return (output, command) if return_com else output

    output = _runica_on_dataset(EEG, icatype, ica_options, reorder=reorder, chanind=chanind)
    command = _history_command(icatype, ica_options, reorder, chanind)
    return (output, command) if return_com else output


def pop_runica_dialog_spec(EEG) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_runica``."""
    dataset_count = len(EEG) if isinstance(EEG, list) else 1
    first_eeg = EEG[0] if isinstance(EEG, list) else EEG
    chanlocs = _as_list(first_eeg.get("chanlocs", []))
    labels = tuple(str(chan.get("labels", "")) for chan in chanlocs if isinstance(chan, dict))
    types = tuple(
        value for value in dict.fromkeys(
            str(chan.get("type", "")) for chan in chanlocs if isinstance(chan, dict) and chan.get("type", "")
        )
    )
    algorithm_labels = "|".join(description for _name, description, _options in _ALGORITHMS)
    controls = [
        ControlSpec("text", "ICA algorithm to use (click to select)"),
        ControlSpec("listbox", algorithm_labels, tag="icatype", value=1),
        ControlSpec("text", "Commandline options (See help messages)"),
        ControlSpec("edit", tag="params", value=_ALGORITHMS[0][2]),
        ControlSpec("checkbox", "Reorder components by variance (if that's not already the case)", tag="reorder", value=True),
        ControlSpec("text", "Use only channel type(s) or indices"),
        ControlSpec("edit", tag="chantype", value=""),
        ControlSpec(
            "pushbutton",
            "... types",
            tag="type_button",
            enabled=bool(types),
            callback=CallbackSpec(
                "select_channels",
                params={"button": "type_button", "target": "chantype", "channels": types},
                matlab_callback="pop_chansel({tmpchanlocs.type}, 'withindex', 'off')",
            ),
        ),
        ControlSpec(
            "pushbutton",
            "... channels",
            tag="chan_button",
            enabled=bool(labels),
            callback=CallbackSpec(
                "select_channels",
                params={"button": "chan_button", "target": "chantype", "channels": labels},
                matlab_callback="pop_chansel({tmpchanlocs.labels}, 'withindex', 'on')",
            ),
        ),
    ]
    geometry = [(2, 1.5), (2, 1.5), (1,), (2, 1, 1, 1)]
    height = 334
    if dataset_count > 1:
        dataset_values = list(range(1, dataset_count + 1))
        controls.extend(
            [
                ControlSpec("text", "Datasets to use for ICA decomposition"),
                ControlSpec("listbox", "|".join(_dataset_labels(EEG)), tag="dataset", value=dataset_values),
            ]
        )
        controls.extend(
            [
                ControlSpec("text", "Concatenate all datasets (check=yes; uncheck=run ICA on each dataset)?"),
                ControlSpec("checkbox", tag="concatenate", value=False),
                ControlSpec("text", "Concatenate datasets for the same subject and session (check=yes)?"),
                ControlSpec("checkbox", tag="concatcond", value=True),
            ]
        )
        geometry.extend([(2, 1.5), (2, 0.2), (2, 0.2)])
        height = 454
    return DialogSpec(
        title="Run ICA decomposition -- pop_runica()",
        function_name="pop_runica",
        eeglab_source="functions/popfunc/pop_runica.m",
        geometry=tuple(geometry),
        size=(824, height),
        help_text="pophelp('pop_runica')",
        controls=tuple(controls),
    )


def _run_gui(EEG, renderer=None):
    spec = pop_runica_dialog_spec(EEG)
    result = inputgui(spec, renderer=renderer)
    if result is None:
        return None
    algorithm_index = int(result.get("icatype", 1)) - 1
    algorithm_index = max(0, min(algorithm_index, len(_ALGORITHMS) - 1))
    chan_text = str(result.get("chantype", "") or "").strip()
    return {
        "icatype": _ALGORITHMS[algorithm_index][0],
        "options": _parse_option_text(str(result.get("params", "") or "")),
        "reorder": "on" if result.get("reorder") else "off",
        "chanind": _parse_channel_text(chan_text) if chan_text else None,
        "dataset": result.get("dataset"),
        "concatenate": "on" if result.get("concatenate") else "off",
        "concatcond": "on" if result.get("concatcond") else "off",
    }


def _runica_on_dataset(EEG, icatype, options, *, reorder, chanind):
    prepared = _prepare_ica_dataset(EEG)
    chanind = _resolve_chanind(prepared, chanind)
    if chanind is None:
        return _finalize_ica_dataset(_run_ica_backend(prepared, icatype, options, reorder=reorder), prepared)

    subset = copy.deepcopy(prepared)
    subset["data"] = np.asarray(subset["data"])[chanind, ...]
    subset["nbchan"] = len(chanind)
    chanlocs = _as_list(subset.get("chanlocs", []))
    if chanlocs:
        subset["chanlocs"] = [chanlocs[index] for index in chanind]
    subset = _run_ica_backend(subset, icatype, options, reorder=reorder)
    output = copy.deepcopy(prepared)
    output["icasphere"] = subset["icasphere"]
    output["icaweights"] = subset["icaweights"]
    output["icawinv"] = subset["icawinv"]
    output["icaact"] = subset["icaact"]
    output["icachansind"] = np.asarray(chanind)
    return _finalize_ica_dataset(output, prepared)


def _runica_on_datasets(EEG, *, dataset, icatype, options, reorder, chanind, concatenate, concatcond):
    output = list(EEG)
    indices = _dataset_indices(output, dataset)
    selected = [output[index] for index in indices]
    if _is_on(concatcond):
        updated = _runica_by_subject_session(selected, icatype, options, reorder=reorder, chanind=chanind)
    elif _is_on(concatenate):
        updated = _runica_concatenated(selected, icatype, options, reorder=reorder, chanind=chanind)
    else:
        updated = [_runica_on_dataset(item, icatype, options, reorder=reorder, chanind=chanind) for item in selected]
    for index, item in zip(indices, updated):
        output[index] = item
    return output


def _runica_concatenated(datasets, icatype, options, *, reorder, chanind):
    merged = copy.deepcopy(datasets[0])
    arrays = [_flatten_dataset_data(item) for item in datasets]
    nbchan = arrays[0].shape[0]
    if any(array.shape[0] != nbchan for array in arrays):
        raise ValueError("Cannot concatenate datasets with different channel counts")
    merged["data"] = np.concatenate(arrays, axis=1)
    merged["nbchan"] = nbchan
    merged["pnts"] = merged["data"].shape[1]
    merged["trials"] = 1
    merged["event"] = []
    merged["urevent"] = []
    merged = _runica_on_dataset(merged, icatype, options, reorder=reorder, chanind=chanind)
    return [_copy_ica_fields(item, merged) for item in datasets]


def _runica_by_subject_session(datasets, icatype, options, *, reorder, chanind):
    groups: dict[tuple[str, Any], list[int]] = {}
    for index, eeg in enumerate(datasets):
        groups.setdefault(_subject_session_key(eeg), []).append(index)
    output = list(datasets)
    for indices in groups.values():
        updated = _runica_concatenated([datasets[index] for index in indices], icatype, options, reorder=reorder, chanind=chanind)
        for index, item in zip(indices, updated):
            output[index] = item
    return output


def _subject_session_key(EEG):
    subject = EEG.get("subject")
    session = EEG.get("session")
    return ("" if subject in (None, "") else str(subject), "" if session in (None, "") else session)


def _flatten_dataset_data(EEG):
    data = np.asarray(EEG["data"])
    return data.reshape(data.shape[0], -1)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return [value]
    return list(value)


def _copy_ica_fields(EEG, source):
    output = _prepare_ica_dataset(EEG)
    for key in ["icaweights", "icasphere", "icawinv"]:
        output[key] = copy.deepcopy(source[key])
    output["icachansind"] = copy.deepcopy(source.get("icachansind", np.arange(int(output.get("nbchan", 0)))))
    output["icaact"] = np.array([])
    return output


def _prepare_ica_dataset(EEG):
    output = copy.deepcopy(EEG)
    _save_old_ica(output)
    for key in ["icaweights", "icasphere", "icawinv", "icaact"]:
        output[key] = np.array([])
    if isinstance(output.get("etc"), dict):
        output["etc"].pop("ic_classification", None)
    return output


def _save_old_ica(EEG):
    weights = np.asarray(EEG.get("icaweights", []))
    sphere = np.asarray(EEG.get("icasphere", []))
    if not weights.size or not sphere.size:
        return
    etc = EEG.setdefault("etc", {})
    if not isinstance(etc, dict):
        return
    etc.setdefault("oldicaweights", []).insert(0, copy.deepcopy(EEG.get("icaweights")))
    etc.setdefault("oldicasphere", []).insert(0, copy.deepcopy(EEG.get("icasphere")))
    etc.setdefault("oldicachansind", []).insert(0, copy.deepcopy(EEG.get("icachansind", [])))


def _finalize_ica_dataset(output, prepared):
    prepared_etc = prepared.get("etc")
    if isinstance(prepared_etc, dict):
        output_etc = output.setdefault("etc", {})
        if isinstance(output_etc, dict):
            for key in ["oldicaweights", "oldicasphere", "oldicachansind"]:
                if key in prepared_etc and key not in output_etc:
                    output_etc[key] = copy.deepcopy(prepared_etc[key])
            output_etc.pop("ic_classification", None)
    return output


def _run_ica_backend(EEG, icatype, options, *, reorder):
    if icatype == "runica":
        return eeg_runica(EEG, sortcomps=reorder, **options)
    if icatype == "picard":
        return eeg_picard(EEG, sortcomps=reorder, **_picard_options(options))
    if icatype == "runamica15":
        return eeg_amica(EEG, sortcomps=reorder, **_amica_options(options))
    raise NotImplementedError(
        "EEGPrep pop_runica supports runica, picard, and runamica15/amica. "
        f"The '{icatype}' ICA algorithm is not ported yet."
    )


def _picard_options(options):
    mapped = {}
    for key, value in options.items():
        lower_key = str(key).lower()
        if lower_key == "maxiter":
            mapped["max_iter"] = value
        elif lower_key == "mode":
            if str(value).lower() == "standard":
                mapped["ortho"] = False
            elif str(value).lower() in {"ortho", "picardo"}:
                mapped["ortho"] = True
            else:
                raise ValueError("Picard mode must be 'standard' or 'ortho'")
        else:
            mapped[lower_key] = value
    return mapped


def _amica_options(options):
    mapped = {}
    for key, value in options.items():
        lower_key = str(key).lower()
        mapped["max_iter" if lower_key == "maxiter" else lower_key] = value
    return mapped


def _dataset_indices(EEG, dataset):
    if dataset is None:
        return list(range(len(EEG)))
    values = np.asarray(dataset).tolist() if isinstance(dataset, np.ndarray) else dataset
    if isinstance(values, (int, np.integer)):
        values = [int(values)]
    if not values:
        raise ValueError("dataset must contain at least one index")
    indices = [int(value) - 1 for value in values]
    if any(index < 0 or index >= len(EEG) for index in indices):
        raise ValueError("dataset indices must be 1-based and within the ALLEEG list")
    return indices


def _resolve_chanind(EEG, chanind):
    if chanind is None:
        return None
    if isinstance(chanind, str):
        if chanind == "":
            return None
        chanind = _parse_channel_text(chanind)
    if isinstance(chanind, np.ndarray):
        chanind = chanind.tolist()
    values = list(chanind)
    if not values:
        return None
    if all(isinstance(value, str) and not _is_int_text(value) for value in values):
        indices, _labels = eeg_decodechan(EEG, values, "labels", True)
        if len(indices) != len(values):
            indices, _labels = eeg_decodechan(EEG, values, "type", True)
        return list(indices)
    indices = [int(value) - 1 for value in values]
    if any(index < 0 or index >= int(EEG.get("nbchan", 0)) for index in indices):
        raise ValueError("chanind values must be 1-based and within EEG.nbchan")
    return indices


def _parse_runica_args(args, kwargs):
    if args and len(args) % 2:
        first = str(args[0]).lower()
        if first in {"selectamica", "selectamicaloc"}:
            parsed = {"icatype": "runamica15", "options": {"outdir": "amicaout"}}
            if first == "selectamicaloc":
                parsed["options"]["qsub"] = "off"
            parsed.update(parse_key_value_args(args[1:], kwargs, lowercase_kwargs=True))
            return parsed
        return parse_key_value_args(("icatype", *args), kwargs, lowercase_kwargs=True)
    return parse_key_value_args(args, kwargs, lowercase_kwargs=True)


def _normalise_icatype(icatype):
    value = str(icatype).lower()
    return "runamica15" if value == "amica" else value


def _normalise_ica_options(icatype, options, extra):
    parsed = dict(extra)
    if options is None:
        defaults = _default_options(icatype)
        defaults.update(parsed)
        return defaults
    if isinstance(options, dict):
        parsed.update(options)
        return parsed
    if isinstance(options, str):
        parsed.update(_parse_option_text(options))
        return parsed
    if isinstance(options, (list, tuple)):
        parsed.update(_pairs_to_dict(options))
        return parsed
    raise TypeError("options must be a string, dict, list, tuple, or None")


def _default_options(icatype):
    if icatype == "runica":
        return {"extended": 1}
    if icatype == "picard":
        return {"maxiter": 500, "mode": "standard"}
    if icatype == "runamica15":
        return {"maxiter": 2000}
    return {}


def _parse_option_text(text):
    text = text.strip()
    if not text:
        return {}
    tokens = re.findall(r"'([^']*)'|\"([^\"]*)\"|([^,\s]+)", text)
    return _pairs_to_dict([_parse_scalar(next(part for part in token if part)) for token in tokens])


def _pairs_to_dict(values):
    if len(values) % 2:
        raise ValueError("ICA options must be key/value pairs")
    return {str(values[index]).lower(): values[index + 1] for index in range(0, len(values), 2)}


def _parse_channel_text(text):
    return [_parse_scalar(value) for value in re.split(r"[\s,]+", text.strip().strip("[]")) if value]


def _parse_scalar(value):
    value = str(value).strip()
    if _is_int_text(value):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def _is_int_text(value):
    return bool(re.fullmatch(r"[+-]?\d+", str(value).strip()))


def _dataset_labels(ALLEEG):
    labels = []
    for index, eeg in enumerate(ALLEEG, start=1):
        setname = str(eeg.get("setname") or f"Dataset {index}")
        labels.append(f"{index}: {setname}")
    return tuple(labels)


def _history_command(icatype, options, reorder, chanind, *, dataset=None, concatenate="off", concatcond="off"):
    parts = ["'icatype'", _runica_history_value(str(icatype).lower())]
    if dataset is not None:
        parts.extend(["'dataset'", _runica_history_value(dataset)])
    for key, value in options.items():
        parts.extend([_runica_history_value(key), _runica_history_value(value)])
    if str(reorder).lower() != "on":
        parts.extend(["'reorder'", _runica_history_value(reorder)])
    if chanind is not None:
        parts.extend(["'chanind'", _runica_history_value(chanind)])
    if _is_on(concatenate):
        parts.extend(["'concatenate'", "'on'"])
    if _is_on(concatcond):
        parts.extend(["'concatcond'", "'on'"])
    return f"EEG = pop_runica(EEG, {', '.join(parts)});"


def _runica_history_value(value):
    return format_history_value(value, cell_for_sequence=None)


def _is_on(value):
    return str(value).lower() in {"on", "yes", "true", "1"}
