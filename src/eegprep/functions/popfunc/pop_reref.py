"""EEGLAB-style EEG re-referencing pop function."""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._chanutils import (
    chanlocs_as_list as _chanlocs_as_list,
    is_number_like as _is_number_like,
    normalise_reflocs as _normalise_reflocs,
)
from eegprep.functions.popfunc._pop_utils import (
    format_history_value,
    parse_key_value_args,
    parse_text_tokens,
)
from eegprep.functions.popfunc.pop_interp import pop_interp
from eegprep.functions.sigprocfunc.reref import reref

logger = logging.getLogger(__name__)

_UNSET = object()
_VALID_OPTIONS = {
    "exclude",
    "keepref",
    "refloc",
    "refica",
    "interpchan",
    "huber",
}


def pop_reref(
    EEG: dict | list[dict] | object = _UNSET,
    ref: Any = _UNSET,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Convert an EEG dataset to average or common-reference data.

    Numeric channel indices are 0-based to match existing EEGPrep channel
    selection APIs. String channel labels are resolved case-insensitively
    against ``EEG["chanlocs"]``. Calling ``pop_reref(EEG)`` launches the GUI;
    pass ``ref=[]`` or ``ref=None`` for command-line average reference.

    Args:
        EEG: EEG dictionary, or a list of EEG dictionaries.
        ref: ``None``/``[]`` for average reference, or reference channel label
            or 0-based index/index sequence for common reference.
        *args: EEGLAB-style key/value options.
        gui: Force or suppress the GUI entrypoint.
        renderer: Optional GUI renderer for tests.
        return_com: Return ``(EEG, command)`` when true.
        **kwargs: Options such as ``exclude``, ``keepref``, ``refloc``,
            ``refica``, and ``huber``.

    Returns:
        dict or tuple: Re-referenced EEG, and optionally the EEGLAB-style
        command string.
    """
    if EEG is _UNSET:
        return (None, "") if return_com else None

    options, ref_from_options = _parse_options(args, kwargs)
    if ref is _UNSET and ref_from_options is not _UNSET:
        ref = ref_from_options
    elif ref is not _UNSET and ref_from_options is not _UNSET:
        raise ValueError("Reference was provided more than once")

    has_processing_args = ref is not _UNSET
    if gui is None:
        gui = not has_processing_args

    if isinstance(EEG, list):
        if not EEG:
            return ([], "") if return_com else []
        if gui:
            gui_result = _run_gui(EEG[0], renderer=renderer)
            if gui_result is None:
                return (EEG, "") if return_com else EEG
            ref = gui_result.pop("ref")
            options.update(gui_result)
        elif ref is _UNSET:
            ref = []
        outputs = [
            pop_reref(item, ref, gui=False, renderer=renderer, return_com=False, **options)
            for item in EEG
        ]
        com = _history_command(ref, _history_options(options))
        return (outputs, com) if return_com else outputs

    if gui:
        gui_result = _run_gui(EEG, renderer=renderer)
        if gui_result is None:
            return (EEG, "") if return_com else EEG
        ref = gui_result.pop("ref")
        options.update(gui_result)
    elif ref is _UNSET:
        ref = []

    _validate_eeg(EEG)
    EEG_out = copy.deepcopy(EEG)
    resolved = _resolve_options(EEG_out, ref, options)

    original_nbchan = int(EEG_out["data"].shape[0])
    original_chanlocs = _chanlocs_as_list(EEG_out.get("chanlocs", []))
    interp_labels = [str(chan.get("labels", "")) for chan in resolved["interpchan"]]
    if resolved["interpchan"]:
        EEG_out = pop_interp(EEG_out, resolved["interpchan"], "spherical")

    EEG_out["data"], chanlocs, removed_ref_chans, _mean_data = reref(
        EEG_out["data"],
        resolved["ref_indices"],
        exclude=resolved["exclude_indices"],
        keepref=resolved["keepref"],
        elocs=_chanlocs_as_list(EEG_out.get("chanlocs", [])),
        refloc=resolved["refloc"],
        huber=resolved["huber"],
    )
    if chanlocs is not None:
        EEG_out["chanlocs"] = chanlocs
    EEG_out["nbchan"] = int(EEG_out["data"].shape[0])
    if interp_labels:
        EEG_out = _remove_channels_by_labels(EEG_out, interp_labels)
    _store_removed_ref_channels(EEG_out, removed_ref_chans)
    _remove_refloc_from_removed_channels(EEG_out, resolved["refloc"])
    _update_legacy_ref(EEG_out, resolved["ref_indices"])
    _update_ica(
        EEG_out,
        original_nbchan=original_nbchan,
        original_chanlocs=original_chanlocs,
        resolved=resolved,
    )

    _normalise_checkset_types(EEG_out)
    EEG_out = eeg_checkset(EEG_out)
    com = _history_command(ref, resolved["history_options"])
    return (EEG_out, com) if return_com else EEG_out


def pop_reref_dialog_spec(
    current_ref: str = "unknown",
    channel_labels: Any = (),
    refloc_labels: Any = (),
) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_reref``."""
    channel_labels = tuple(str(label) for label in channel_labels)
    refloc_labels = tuple(str(label) for label in refloc_labels)
    geometry = (
        (1,),
        (1,),
        (1.8, 1, 0.3),
        (1.8, 1, 0.3),
        (1,),
        (1,),
        (1,),
        (1.8, 1, 0.3),
        (1.8, 1, 0.3),
    )
    return DialogSpec(
        title="pop_reref - average reference or re-reference data",
        function_name="pop_reref",
        eeglab_source="functions/popfunc/pop_reref.m",
        geometry=geometry,
        size=(616, 281),
        help_text="pophelp('pop_reref')",
        controls=(
            ControlSpec("text", f"Current data reference state is: {current_ref}"),
            ControlSpec(
                "checkbox",
                "Compute average reference",
                tag="ave",
                value=True,
                callback=CallbackSpec(
                    "set_reref_mode",
                    params={"source": "ave", "mode": "average"},
                    matlab_callback="cb_averef",
                ),
            ),
            ControlSpec(
                "checkbox",
                "Huber average ref. with threshold",
                tag="huberef",
                value=False,
                callback=CallbackSpec(
                    "set_reref_mode",
                    params={"source": "huberef", "mode": "huber"},
                    matlab_callback="cb_huberref",
                ),
            ),
            ControlSpec("edit", tag="huberval", value="25"),
            ControlSpec("text", "uV", tag="scale"),
            ControlSpec(
                "checkbox",
                "Re-reference data to channel(s):",
                tag="rerefstr",
                value=False,
                callback=CallbackSpec(
                    "set_reref_mode",
                    params={"source": "rerefstr", "mode": "channels"},
                    matlab_callback="cb_ref",
                ),
            ),
            ControlSpec("edit", tag="reref", value="", enabled=False),
            ControlSpec(
                "pushbutton",
                "...",
                tag="refbr",
                enabled=False,
                callback=CallbackSpec(
                    "select_channels",
                    params={
                        "button": "refbr",
                        "target": "reref",
                        "channels": channel_labels,
                    },
                    matlab_callback="pop_chansel({tmpchanlocs.labels}, 'withindex', 'on')",
                ),
            ),
            ControlSpec("checkbox", "Interpolate removed channel(s)", tag="interp", value=False),
            ControlSpec("spacer"),
            ControlSpec(
                "checkbox",
                "Retain ref. channel(s) in data (will be flat for single-channel ref.)",
                tag="keepref",
                value=False,
                enabled=False,
            ),
            ControlSpec("text", "Exclude channel indices (EMG, EOG)"),
            ControlSpec("edit", tag="exclude", value=""),
            ControlSpec(
                "pushbutton",
                "...",
                tag="exclude_button",
                callback=CallbackSpec(
                    "select_channels",
                    params={
                        "button": "exclude_button",
                        "target": "exclude",
                        "channels": channel_labels,
                    },
                    matlab_callback="pop_chansel({tmpchanlocs.labels}, 'withindex', 'on')",
                ),
            ),
            ControlSpec("text", "Add old ref. channel back to the data", tag="reflocstr"),
            ControlSpec("edit", tag="refloc", value=""),
            ControlSpec(
                "pushbutton",
                "...",
                tag="refloc_button",
                callback=CallbackSpec(
                    "select_channels",
                    params={
                        "button": "refloc_button",
                        "target": "refloc",
                        "channels": refloc_labels,
                        "no_channels_message": (
                            "There are no Reference channel defined, add it using the channel location editor"
                        ),
                    },
                    matlab_callback="pop_chansel({tmpchaninfo.nodatchans.labels}, 'withindex', 'on')",
                ),
            ),
        ),
    )


def _parse_options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    options = parse_key_value_args((), kwargs, lowercase_kwargs=True)
    arg_options = parse_key_value_args(args)
    ref = _UNSET
    if "ref" in arg_options:
        ref = arg_options.pop("ref")
    options.update(arg_options)
    unknown = sorted(set(options) - _VALID_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown pop_reref option(s): {', '.join(unknown)}")
    return options, ref


def _validate_eeg(EEG: dict) -> None:
    if not isinstance(EEG, dict):
        raise TypeError("EEG must be a dictionary")
    data = EEG.get("data")
    if data is None or np.size(data) == 0:
        raise ValueError("Pop_reref: cannot process empty data")


def _resolve_options(EEG: dict, ref: Any, options: dict[str, Any]) -> dict[str, Any]:
    ref_indices = _resolve_channels(EEG, ref)
    exclude_indices = _resolve_channels(EEG, options.get("exclude", []))
    refica = str(options.get("refica", "on")).lower()
    if refica not in {"on", "off", "backwardcomp", "remove"}:
        raise ValueError("refica must be 'on', 'off', 'backwardcomp', or 'remove'")

    keepref = str(options.get("keepref", "off")).lower()
    huber = options.get("huber")
    huber = None if _is_empty(huber) else float(huber)
    refloc = _resolve_refloc(EEG, options.get("refloc", []))
    interpchan = _resolve_interpchan(EEG, options.get("interpchan", "off"))
    return {
        "ref_indices": ref_indices,
        "exclude_indices": exclude_indices,
        "keepref": keepref,
        "refloc": refloc,
        "refica": refica,
        "huber": huber,
        "interpchan": interpchan,
        "history_options": {
            key: value for key, value in _history_options(options).items()
        },
    }


def _history_options(options: dict[str, Any]) -> dict[str, Any]:
    history: dict[str, Any] = {}
    for key, value in options.items():
        if key not in {"exclude", "keepref", "refloc", "refica", "huber", "interpchan"}:
            continue
        normalised = _normalise_history_option_value(key, value)
        if _is_default_option(key, normalised):
            continue
        history[key] = normalised
    return history


def _normalise_history_option_value(key: str, value: Any) -> Any:
    if key in {"keepref", "refica"} and isinstance(value, str):
        return value.lower()
    if key == "huber" and not _is_empty(value):
        number = float(value)
        return int(number) if number.is_integer() else number
    return value


def _resolve_channels(EEG: dict, channels: Any) -> list[int]:
    if _is_empty(channels):
        return []
    chanlocs = _chanlocs_as_list(EEG.get("chanlocs", []))
    nchan = int(np.asarray(EEG["data"]).shape[0])
    values = _normalise_channel_values(channels)
    labels = [str(chan.get("labels", "")).strip().lower() for chan in chanlocs]

    indices: list[int] = []
    for value in values:
        if isinstance(value, str) and not _is_int_text(value):
            matches = [idx for idx, label in enumerate(labels) if label == value.strip().lower()]
            if not matches:
                raise ValueError(f"Channel '{value}' not found")
            indices.extend(matches)
            continue
        index = int(value)
        if index < 0 or index >= nchan:
            raise ValueError("Channel index out of range")
        indices.append(index)
    return sorted(set(indices))


def _normalise_channel_values(channels: Any) -> list[Any]:
    if isinstance(channels, np.ndarray):
        channels = channels.tolist()
    if isinstance(channels, (int, np.integer)):
        return [int(channels)]
    if isinstance(channels, float) and channels.is_integer():
        return [int(channels)]
    if isinstance(channels, (str, bytes)):
        text = channels.decode("utf-8") if isinstance(channels, bytes) else channels
        text = text.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if text.startswith("{") and text.endswith("}"):
            text = text[1:-1]
        return parse_text_tokens(text)
    return list(channels)


def _store_removed_ref_channels(EEG: dict, removed: list[dict[str, Any]]) -> None:
    if not removed:
        return
    chaninfo = EEG.setdefault("chaninfo", {})
    existing = chaninfo.get("removedchans", [])
    if isinstance(existing, np.ndarray):
        existing = existing.tolist()
    if isinstance(existing, dict):
        existing = [existing]
    chaninfo["removedchans"] = list(existing) + removed


def _remove_channels_by_labels(EEG: dict, labels: list[str]) -> dict:
    labels_lower = {label.lower() for label in labels}
    chanlocs = _chanlocs_as_list(EEG.get("chanlocs", []))
    remove_indices = [
        index
        for index, chan in enumerate(chanlocs)
        if str(chan.get("labels", "")).lower() in labels_lower
    ]
    if not remove_indices:
        return EEG

    keep = [index for index in range(int(EEG["data"].shape[0])) if index not in set(remove_indices)]
    EEG["data"] = np.asarray(EEG["data"])[keep, ...]
    EEG["chanlocs"] = [chan for index, chan in enumerate(chanlocs) if index in keep]
    EEG["nbchan"] = int(EEG["data"].shape[0])
    return EEG


def _remove_refloc_from_removed_channels(EEG: dict, refloc: Any) -> None:
    if _is_empty(refloc):
        return
    chaninfo = EEG.get("chaninfo", {})
    removed = chaninfo.get("removedchans", [])
    if not removed:
        raise ValueError("Missing reference channel information. Edit channels and add reference first.")
    labels = {str(loc.get("labels", "")).lower() for loc in _normalise_reflocs(refloc)}
    if not labels:
        return
    chaninfo["removedchans"] = [
        loc for loc in _chanlocs_as_list(removed) if str(loc.get("labels", "")).lower() not in labels
    ]


def _resolve_refloc(EEG: dict, refloc: Any) -> list[dict[str, Any]]:
    if _is_empty(refloc):
        return []
    refloc_values = refloc.tolist() if isinstance(refloc, np.ndarray) else refloc
    if (
        isinstance(refloc_values, (list, tuple))
        and len(refloc_values) == 3
        and isinstance(refloc_values[0], str)
        and _is_number_like(refloc_values[1])
        and _is_number_like(refloc_values[2])
    ):
        return _normalise_reflocs(refloc_values)
    if isinstance(refloc_values, dict):
        return _normalise_reflocs(refloc_values)
    if isinstance(refloc_values, (list, tuple)) and refloc_values and isinstance(refloc_values[0], dict):
        return _normalise_reflocs(refloc_values)

    nodatchans = _reference_nodatchans(EEG)
    if not nodatchans:
        raise ValueError("Reference channel locations require EEG['chaninfo']['nodatchans']")
    return _resolve_locs_from_identifiers(nodatchans, refloc, "reference location")


def _resolve_interpchan(EEG: dict, interpchan: Any) -> list[dict[str, Any]]:
    if interpchan is None or (isinstance(interpchan, str) and interpchan.lower() == "off"):
        return []
    if isinstance(interpchan, dict):
        return [copy.deepcopy(interpchan)]
    if isinstance(interpchan, np.ndarray):
        interpchan = interpchan.tolist()
    if isinstance(interpchan, (list, tuple)) and interpchan and isinstance(interpchan[0], dict):
        return [copy.deepcopy(loc) for loc in interpchan if _has_xyz(loc)]
    if _is_empty(interpchan):
        inferred = _infer_removed_channels(EEG)
        if inferred:
            return inferred
        logger.info("pop_reref: no removed channels found for interpolation")
        return []

    urchanlocs = _chanlocs_as_list(EEG.get("urchanlocs", []))
    if not urchanlocs:
        raise ValueError("interpchan indices require EEG['urchanlocs']")
    return _resolve_locs_from_identifiers(urchanlocs, interpchan, "interpolation channel")


def _infer_removed_channels(EEG: dict) -> list[dict[str, Any]]:
    removedchans = _chanlocs_as_list(EEG.get("chaninfo", {}).get("removedchans", []))
    valid_removed = [copy.deepcopy(chan) for chan in removedchans if _has_xyz(chan)]
    if valid_removed:
        return valid_removed

    chanlocs = _chanlocs_as_list(EEG.get("chanlocs", []))
    urchanlocs = _chanlocs_as_list(EEG.get("urchanlocs", []))
    if not chanlocs or not urchanlocs:
        return []

    chan_types = [str(chan.get("type", "")).strip().lower() for chan in chanlocs]
    urchan_types = [str(chan.get("type", "")).strip().lower() for chan in urchanlocs]
    require_eeg_type = all(chan_types) and all(urchan_types)
    current_labels = {
        str(chan.get("labels", "")).lower()
        for chan in chanlocs
        if _is_referenceable_eeg(chan, require_eeg_type)
    }
    inferred = []
    for chan in urchanlocs:
        label = str(chan.get("labels", "")).lower()
        if (
            label
            and label not in current_labels
            and _is_referenceable_eeg(chan, require_eeg_type)
            and _has_xyz(chan)
        ):
            inferred.append(copy.deepcopy(chan))
    return inferred


def _resolve_locs_from_identifiers(
    locs: list[dict[str, Any]],
    identifiers: Any,
    label: str,
) -> list[dict[str, Any]]:
    values = _normalise_channel_values(identifiers)
    labels = [str(loc.get("labels", "")).lower() for loc in locs]
    out = []
    for value in values:
        if isinstance(value, str) and not _is_int_text(value):
            matches = [idx for idx, loc_label in enumerate(labels) if loc_label == value.lower()]
            if not matches:
                raise ValueError(f"{label.capitalize()} '{value}' not found")
            out.extend(copy.deepcopy(locs[idx]) for idx in matches)
        else:
            index = int(value)
            if index < 0 or index >= len(locs):
                raise ValueError(f"{label.capitalize()} index out of range")
            out.append(copy.deepcopy(locs[index]))
    return [loc for loc in out if _has_xyz(loc) or label == "reference location"]


def _has_xyz(chan: dict[str, Any]) -> bool:
    for key in ("X", "Y", "Z"):
        value = chan.get(key)
        if value is None:
            return False
        try:
            if np.size(value) == 0 or np.isnan(float(np.asarray(value).squeeze())):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _is_referenceable_eeg(chan: dict[str, Any], require_eeg_type: bool) -> bool:
    chan_type = str(chan.get("type", "")).strip().lower()
    if require_eeg_type:
        return chan_type == "eeg"
    return chan_type != "fid"


def _update_legacy_ref(EEG: dict, ref_indices: list[int]) -> None:
    if "ref" not in EEG:
        return
    if not ref_indices:
        if str(EEG["ref"]).lower() == "common":
            EEG["ref"] = "average"
    elif str(EEG["ref"]).lower() == "average":
        EEG["ref"] = "common"


def _update_ica(
    EEG: dict,
    *,
    original_nbchan: int,
    original_chanlocs: list[dict[str, Any]],
    resolved: dict[str, Any],
) -> None:
    if resolved["refica"] == "remove":
        _clear_ica(EEG)
        return

    icaweights = EEG.get("icaweights")
    icawinv = EEG.get("icawinv")
    if _is_empty_array(icaweights) or _is_empty_array(icawinv):
        return

    if resolved["refica"] == "off":
        EEG["icaact"] = np.array([])
        return
    if resolved["refica"] == "backwardcomp":
        resolved = {**resolved, "refica": "on"}
    else:
        EEG["icaact"] = np.array([])

    if resolved["interpchan"]:
        logger.warning(
            "Removing ICA decomposition because interpolation changed the re-referenced channel set."
        )
        _clear_ica(EEG)
        return

    icachansind = _normalise_icachansind(EEG.get("icachansind", []))
    if any(index in resolved["exclude_indices"] for index in icachansind):
        logger.warning("Removing ICA decomposition because ICA channels were excluded from re-referencing.")
        _clear_ica(EEG)
        return
    if len(icachansind) != original_nbchan - len(resolved["exclude_indices"]):
        logger.warning("Removing ICA decomposition because ICA channels do not match re-referenced channels.")
        _clear_ica(EEG)
        return

    original_icawinv = np.asarray(icawinv)
    expanded_icawinv = np.zeros((original_nbchan, original_icawinv.shape[1]), dtype=original_icawinv.dtype)
    expanded_icawinv[np.asarray(icachansind, dtype=int), :] = original_icawinv

    new_icawinv, new_chanlocs, _removed, _mean = reref(
        expanded_icawinv,
        resolved["ref_indices"],
        exclude=resolved["exclude_indices"],
        keepref=resolved["keepref"],
        elocs=original_chanlocs,
        refloc=resolved["refloc"],
        huber=resolved["huber"],
    )
    new_chanlocs = new_chanlocs or []
    new_labels = [str(chan.get("labels", "")).lower() for chan in new_chanlocs]
    new_icachansind = []
    keep_rows = []
    for old_index in icachansind:
        old_label = str(original_chanlocs[old_index].get("labels", "")).lower()
        if old_label not in new_labels:
            continue
        new_index = new_labels.index(old_label)
        new_icachansind.append(new_index)
        keep_rows.append(new_index)

    if len(new_icachansind) != len(keep_rows):
        _clear_ica(EEG)
        return
    EEG["icawinv"] = new_icawinv[np.asarray(keep_rows, dtype=int), :]
    EEG["icachansind"] = new_icachansind
    if len(EEG["icachansind"]) != EEG["icawinv"].shape[0]:
        _clear_ica(EEG)
        return
    EEG["icaweights"] = np.linalg.pinv(EEG["icawinv"])
    EEG["icasphere"] = np.eye(len(EEG["icachansind"]))


def _clear_ica(EEG: dict) -> None:
    EEG["icawinv"] = np.array([])
    EEG["icaweights"] = np.array([])
    EEG["icasphere"] = np.array([])
    EEG["icaact"] = np.array([])


def _normalise_checkset_types(EEG: dict) -> None:
    if "icachansind" in EEG:
        EEG["icachansind"] = np.asarray(_normalise_icachansind(EEG["icachansind"]), dtype=int)


def _normalise_icachansind(value: Any) -> list[int]:
    if _is_empty_array(value):
        return []
    return np.asarray(value, dtype=int).reshape(-1).tolist()


def _run_gui(EEG: dict, renderer: Any | None = None) -> dict[str, Any] | None:
    channel_labels = [chan.get("labels", "") for chan in _chanlocs_as_list(EEG.get("chanlocs", []))]
    refloc_labels = [chan.get("labels", "") for chan in _reference_nodatchans(EEG)]
    spec = pop_reref_dialog_spec(_current_reference(EEG), channel_labels, refloc_labels)
    result = inputgui(spec, renderer=renderer)
    if result is None:
        return None
    options: dict[str, Any] = {}
    if result.get("huberef"):
        options.update({"ref": [], "huber": result.get("huberval") or 25})
    elif result.get("rerefstr"):
        ref_text = result.get("reref", "")
        if not str(ref_text).strip():
            logger.info("Aborting: you must enter one or more reference channels")
            return None
        options["ref"] = _parse_gui_channel_text(ref_text)
    else:
        options["ref"] = []

    if result.get("keepref"):
        options["keepref"] = "on"
    if str(result.get("exclude", "")).strip():
        options["exclude"] = _parse_gui_channel_text(result["exclude"])
    if result.get("interp"):
        options["interpchan"] = []
    if str(result.get("refloc", "")).strip():
        try:
            options["refloc"] = _gui_reflocs(EEG, result["refloc"])
        except ValueError:
            logger.info("Error with old reference: ignoring it")
    return options


def _parse_gui_channel_text(text: str) -> list[int | str]:
    values = _normalise_channel_values(text)
    parsed = []
    for value in values:
        if isinstance(value, str) and _is_int_text(value):
            parsed.append(int(value))
        else:
            parsed.append(value)
    return parsed


def _gui_reflocs(EEG: dict, text: str) -> list[dict[str, Any]]:
    requested = _parse_gui_channel_text(text)
    locs = _reference_nodatchans(EEG)
    if not locs:
        raise ValueError("There are no Reference channel defined, add it using the channel location editor")
    labels = [str(loc.get("labels", "")).lower() for loc in locs]
    out = []
    for value in requested:
        if isinstance(value, int):
            if value < 0 or value >= len(locs):
                raise ValueError("Reference location index out of range")
            out.append(locs[value])
        else:
            matches = [idx for idx, label in enumerate(labels) if label == str(value).lower()]
            if not matches:
                raise ValueError(f"Reference location '{value}' not found")
            out.extend(locs[idx] for idx in matches)
    return out


def _reference_nodatchans(EEG: dict) -> list[dict[str, Any]]:
    locs = _chanlocs_as_list(EEG.get("chaninfo", {}).get("nodatchans", []))
    return [
        loc
        for loc in locs
        if str(loc.get("type", "")).strip().lower() != "fid"
    ]


def _current_reference(EEG: dict) -> str:
    chanlocs = _chanlocs_as_list(EEG.get("chanlocs", []))
    refs = [str(chan.get("ref", "")) for chan in chanlocs if "ref" in chan]
    refs = [ref if ref else "unknown" for ref in refs]
    if not refs:
        return str(EEG.get("ref", "unknown") or "unknown")
    return max(set(refs), key=refs.count)


def _history_command(ref: Any, options: dict[str, Any]) -> str:
    parts = [_format_channel_history_value([] if ref is _UNSET or ref is None else ref)]
    for key, value in options.items():
        parts.append(_format_reref_history_value(key))
        if key in {"exclude", "interpchan"}:
            parts.append(_format_channel_history_value(value))
        else:
            parts.append(_format_reref_history_value(value))
    return f"EEG = pop_reref( EEG, {', '.join(parts)});"


def _format_channel_history_value(channels: Any) -> str:
    if isinstance(channels, np.ndarray):
        channels = channels.tolist()
    if isinstance(channels, str):
        if _is_empty(channels):
            return "[]"
        if _is_int_text(channels):
            return _format_history_number(int(channels) + 1)
        return _format_reref_history_value([channels])
    if isinstance(channels, (int, np.integer)):
        return _format_history_number(int(channels) + 1)
    if isinstance(channels, (float, np.floating)) and float(channels).is_integer():
        return _format_history_number(int(channels) + 1)
    if isinstance(channels, dict):
        return _format_history_struct([channels])
    if isinstance(channels, (list, tuple)):
        values = list(channels)
        if not values:
            return "[]"
        if all(isinstance(item, dict) for item in values):
            return _format_history_struct(values)
        if all(_is_channel_number(item) for item in values):
            shifted = [int(item) + 1 for item in values]
            return "[" + " ".join(_format_history_number(item) for item in shifted) + "]"
        if any(isinstance(item, str) for item in values):
            return "{" + ",".join(_format_reref_history_value(item) for item in values) + "}"
    return _format_reref_history_value(channels)


def _format_reref_history_value(value: Any) -> str:
    return format_history_value(
        value,
        cell_for_sequence="any_strings",
        string_separator=",",
        none_as_empty=True,
        dict_formatter=_format_history_struct,
        number_formatter=_format_history_number,
    )


def _format_history_struct(values: list[dict[str, Any]]) -> str:
    if not values:
        return "struct([])"
    fields: list[str] = []
    for loc in values:
        for field in loc:
            if field not in fields:
                fields.append(field)

    parts = []
    for field in fields:
        contents = [loc.get(field, []) for loc in values]
        if len(contents) == 1 and _is_number_or_empty(contents[0]):
            parts.append(f"'{field}',{_format_reref_history_value(contents[0])}")
        elif len(contents) == 1 and isinstance(contents[0], np.ndarray) and contents[0].size == 0:
            parts.append(f"'{field}',[]")
        else:
            parts.append(f"'{field}'," + "{" + ",".join(_format_reref_history_value(item) for item in contents) + "}")
    return "struct(" + ",".join(parts) + ")"


def _format_history_number(value: Any) -> str:
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _is_channel_number(value: Any) -> bool:
    if isinstance(value, (int, np.integer)):
        return True
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return True
    return isinstance(value, str) and _is_int_text(value)


def _is_number_or_empty(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.size == 0 or np.issubdtype(value.dtype, np.number)
    return isinstance(value, (int, float, np.integer, np.floating)) or _is_empty(value)


def _is_default_option(key: str, value: Any) -> bool:
    if key == "keepref":
        return str(value).lower() == "off"
    if key == "refica":
        return str(value).lower() == "on"
    if key == "interpchan":
        return value in (None, "off")
    return _is_empty(value)


def _is_empty(value: Any) -> bool:
    if value is _UNSET or value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip() in {"[]", "{}"}
    if isinstance(value, np.ndarray):
        return value.size == 0
    try:
        return len(value) == 0
    except TypeError:
        return False


def _is_empty_array(value: Any) -> bool:
    return value is None or np.asarray(value, dtype=object).size == 0


def _is_int_text(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


__all__ = ["pop_reref", "pop_reref_dialog_spec"]
