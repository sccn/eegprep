"""EEGLAB-style channel-location editor."""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args


_CHANNEL_FIELDS = ("labels", "theta", "radius", "X", "Y", "Z", "sph_theta", "sph_phi", "sph_radius", "type", "ref")
_CHANNEL_FIELD_ALIASES = {field.lower(): field for field in _CHANNEL_FIELDS}


def pop_chanedit(
    chans: dict[str, Any] | list[dict[str, Any]],
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Edit EEG channel-location metadata.

    Programmatic channel indices are EEGLAB-facing 1-based indices. Passing an
    EEG dictionary returns an updated EEG dictionary; passing a chanloc list
    returns ``(chanlocs, chaninfo, urchanlocs)`` like EEGLAB's lower-level path.
    """
    options = parse_key_value_args(args, kwargs, lowercase_keys=True)
    is_eeg = _is_eeg_dataset(chans)
    is_eeg_list = _is_eeg_dataset_list(chans)
    if gui is None:
        gui = not bool(options)
    if gui:
        gui_eeg = chans[0] if is_eeg_list else chans
        result = _run_gui(gui_eeg if is_eeg or is_eeg_list else {"chanlocs": gui_eeg}, renderer=renderer)
        if result is None:
            return (chans, "") if return_com else chans
        options.update(result)
    if is_eeg_list:
        outputs = [_apply_chanedit(dataset, options, is_eeg=True)[0] for dataset in chans]
        command = _history_command(options)
        return (outputs, command) if return_com else outputs
    output, chaninfo, urchans = _apply_chanedit(chans, options, is_eeg=is_eeg)
    command = _history_command(options)
    return (output, command) if return_com else output


def pop_chanedit_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like ``pop_chanedit`` dialog spec."""
    chanlocs = chanlocs_as_list(EEG.get("chanlocs"))
    current = chanlocs[0] if chanlocs else {}
    field_rows = [
        ('Channel label ("label")', "labels", "Opt. head center"),
        ('Polar angle ("theta")', "theta", "Rotate axis"),
        ('Polar radius ("radius")', "radius", "Transform axes"),
        ('Cartesian X ("X")', "X", ""),
        ('Cartesian Y ("Y")', "Y", "xyz -> polar & sph."),
        ('Cartesian Z ("Z")', "Z", "sph. -> polar & xyz"),
        ('Spherical horiz. angle ("sph_theta")', "sph_theta", "polar -> sph. & xyz"),
        ('Spherical azimuth angle ("sph_phi")', "sph_phi", ""),
        ('Spherical radius ("sph_radius")', "sph_radius", "Set head radius"),
        ("Channel type", "type", "Set channel types"),
        ("Reference", "ref", "Set reference"),
        ('Index in backup "urchanlocs" structure', "urchan", ""),
        ("Channel in data array (set=yes)", "datachan", ""),
    ]
    controls: list[ControlSpec] = [
        ControlSpec('text', 'Channel information ("field_name"):', font_weight="bold"),
        ControlSpec("spacer"),
        ControlSpec("spacer"),
        ControlSpec("spacer"),
    ]
    geometry: list[tuple[float, ...]] = [(1.5, 1, 0.2, 1)]
    for label, field, button_label in field_rows:
        value = current.get(field, "")
        if field == "urchan":
            value = current.get("urchan", "")
        elif field == "datachan":
            value = bool(current.get("datachan", True))
        else:
            value = _display_channel_value(current, field)
        controls.extend(
            [
                ControlSpec("text", label),
                _channel_value_control(field, value),
                ControlSpec("spacer"),
                _channel_side_button(button_label),
            ]
        )
        geometry.append((1.5, 1, 0.2, 1))
    controls.extend(
        [
            ControlSpec("spacer"),
            ControlSpec("pushbutton", "Delete chan", tag="delete_button", enabled=False),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("text", f"Channel number (of {len(chanlocs)})", tag="chaneditscantitle", font_weight="bold"),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("pushbutton", "Insert chan", tag="insert_button", enabled=False),
            ControlSpec("pushbutton", "<<", tag="back10", enabled=False),
            ControlSpec("pushbutton", "<", tag="back1", enabled=False),
            ControlSpec("edit", tag="channel", value="1"),
            ControlSpec("pushbutton", ">", tag="next1", enabled=False),
            ControlSpec("pushbutton", ">>", tag="next10", enabled=False),
            ControlSpec("pushbutton", "Append chan", tag="append_button", enabled=False),
            ControlSpec("pushbutton", "Plot 2-D", tag="plot2d", enabled=False),
            ControlSpec("text", "Plot radius (0.2-1, []=auto)"),
            ControlSpec("edit", tag="plotrad", value=_display_value(EEG.get("chaninfo", {}).get("plotrad", ""))),
            ControlSpec(
                "popupmenu",
                "Nose along +X|Nose along -X|Nose along +Y|Nose along -Y",
                tag="nosedir",
                value=_nosedir_index(EEG.get("chaninfo", {}).get("nosedir", "+X")),
            ),
            ControlSpec("pushbutton", "Plot 3-D (xyz)", tag="plot3d", enabled=False),
            ControlSpec("spacer"),
            ControlSpec("pushbutton", "Read locations", tag="load", enabled=False),
            ControlSpec("pushbutton", "Read locs help", tag="readhelp", enabled=False),
            ControlSpec("pushbutton", "Look up locs", tag="lookup", enabled=False),
            ControlSpec("pushbutton", "Save (as .ced)", tag="save", enabled=False),
            ControlSpec("pushbutton", "Save (other types)", tag="saveothers", enabled=False),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("checkbox", "Overwrite Original Channels", tag="rplurchan", value=False),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
        ]
    )
    geometry.extend(
        [
            (1,),
            (1.15, 0.5, 0.6, 1.9, 0.4, 0.4, 1.15),
            (1.15, 0.7, 0.7, 1, 0.7, 0.7, 1.15),
            (0.9, 1.3, 0.6, 1.1, 0.9),
            (1,),
            (1, 1, 1, 1, 1),
            (0.5, 0.5, 1, 0.5, 0.5),
        ]
    )
    return DialogSpec(
        title="Edit channel info -- pop_chanedit()",
        function_name="pop_chanedit",
        eeglab_source="functions/popfunc/pop_chanedit.m",
        size=(900, 900),
        content_margins=(42, 26, 42, 30),
        row_spacing=5,
        help_text="pophelp('pop_chanedit')",
        geometry=tuple(geometry),
        controls=tuple(controls),
        geomvert=tuple(1 for _row in geometry),
        known_differences=(
            "EEGPrep supports command-line channel edits and one-channel GUI edits; navigation, plotting, and file buttons are visible but disabled.",
        ),
    )


def _run_gui(EEG: dict[str, Any], *, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_chanedit_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    channel = int(float(result.get("channel") or 1))
    chanlocs = chanlocs_as_list(EEG.get("chanlocs"))
    current = chanlocs[_single_index(channel, len(chanlocs))]
    options: dict[str, Any] = {}
    for field in _CHANNEL_FIELDS:
        text = str(result.get(f"field_{field}") or "").strip()
        if text != _display_channel_value(current, field).strip():
            options.setdefault("changefield", []).append([channel, field, _parse_field_value(text)])
    if options and result.get("rplurchan"):
        options["rplurchanloc"] = "on"
    return options


def _channel_value_control(field: str, value: Any) -> ControlSpec:
    if field == "datachan":
        return ControlSpec("checkbox", tag="field_datachan", value=bool(value), enabled=False)
    if field == "urchan":
        return ControlSpec("text", _display_value(value), tag="field_urchan")
    return ControlSpec("edit", tag=f"field_{field}", value=_display_value(value))


def _display_channel_value(chan: dict[str, Any], field: str) -> str:
    if field not in {"theta", "radius", "sph_theta", "sph_phi", "sph_radius"}:
        return _display_value(chan.get(field, ""))
    converted = dict(chan)
    if all(not _is_blank(converted.get(item)) for item in ("X", "Y", "Z")):
        _cart_to_all(converted)
    return _display_value(converted.get(field, chan.get(field, "")))


def _channel_side_button(label: str) -> ControlSpec:
    if not label:
        return ControlSpec("spacer")
    return ControlSpec("pushbutton", label, tag=f"button_{label.lower().replace(' ', '_')}", enabled=False)


def _nosedir_index(value: Any) -> int:
    choices = ["+X", "-X", "+Y", "-Y"]
    try:
        return choices.index(str(value)) + 1
    except ValueError:
        return 1


def _apply_chanedit(
    chans: dict[str, Any] | list[dict[str, Any]],
    options: dict[str, Any],
    *,
    is_eeg: bool,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    eeg = deepcopy(chans) if is_eeg and isinstance(chans, dict) else None
    chanlocs = chanlocs_as_list(eeg.get("chanlocs") if eeg is not None else chans)
    chaninfo = deepcopy(eeg.get("chaninfo", {})) if eeg is not None else {}
    urchans = chanlocs_as_list(eeg.get("urchanlocs")) if eeg is not None else []

    for key, value in options.items():
        if key == "changefield":
            for item in _as_command_list(value):
                _change_field(chanlocs, item)
        elif key in {"insert", "append"}:
            index, new_chan = _new_channel_args(value)
            chanlocs.insert(index - (0 if key == "append" else 1), new_chan)
        elif key == "delete":
            for index in sorted(_indices(value, len(chanlocs)), reverse=True):
                chanlocs.pop(index)
        elif key == "changechan":
            _change_channel(chanlocs, value)
        elif key == "convert":
            _convert_locations(chanlocs, value)
        elif key == "load":
            chanlocs = _read_chanloc_file(value)
        elif key == "save":
            _write_chanloc_file(value, chanlocs)
        elif key == "headrad":
            for chan in chanlocs:
                chan["sph_radius"] = float(value)
        elif key == "settype":
            indices, chan_type = _index_value_args(value, len(chanlocs))
            for index in indices:
                chanlocs[index]["type"] = chan_type
        elif key == "setref":
            indices, ref = _index_value_args(value, len(chanlocs))
            for index in indices:
                chanlocs[index]["ref"] = ref
        elif key in {"plotrad", "nosedir"}:
            chaninfo[key] = value
        elif key == "rplurchanloc":
            if bool(value):
                urchans = deepcopy(chanlocs)
        else:
            raise ValueError(f"Unsupported pop_chanedit option: {key}")

    if eeg is not None:
        eeg["chanlocs"] = chanlocs
        eeg["urchanlocs"] = urchans
        eeg["chaninfo"] = chaninfo
        eeg["nbchan"] = len(chanlocs)
        eeg["saved"] = "no"
        return eeg_checkset(eeg), chaninfo, urchans
    return chanlocs, chaninfo, urchans


def _change_field(chanlocs: list[dict[str, Any]], item: Any) -> None:
    if not isinstance(item, (list, tuple)) or len(item) != 3:
        raise ValueError("changefield requires [channel, field, value]")
    index = _single_index(item[0], len(chanlocs))
    chanlocs[index][str(item[1])] = item[2]


def _change_channel(chanlocs: list[dict[str, Any]], value: Any) -> None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError("changechan requires [channel, label, ...]")
    index = _single_index(value[0], len(chanlocs))
    for field, item in zip(_CHANNEL_FIELDS, value[1:]):
        chanlocs[index][field] = item


def _new_channel_args(value: Any) -> tuple[int, dict[str, Any]]:
    if isinstance(value, dict):
        index = int(value.get("index", 1))
        chan = {key: item for key, item in value.items() if key != "index"}
        return index, chan
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("insert/append requires an index")
    index = int(value[0])
    if len(value) == 2 and isinstance(value[1], dict):
        return index, dict(value[1])
    chan: dict[str, Any] = {}
    rest = list(value[1:])
    if rest and isinstance(rest[0], str) and len(rest) % 2 == 0:
        for field, item in zip(rest[0::2], rest[1::2]):
            chan[str(field)] = item
    else:
        for field, item in zip(_CHANNEL_FIELDS, rest):
            chan[field] = item
    return index, chan


def _convert_locations(chanlocs: list[dict[str, Any]], value: Any) -> None:
    mode = _convert_mode(value)
    for chan in chanlocs:
        if mode in {"cart2all", "cart2topo", "cart2sph"}:
            _cart_to_all(chan)
        elif mode in {"sph2all", "sph2topo", "sph2cart"}:
            _sph_to_all(chan)
        elif mode in {"topo2all", "topo2sph", "topo2cart"}:
            _topo_to_all(chan)
        else:
            raise ValueError(f"Unsupported channel conversion: {mode}")


def _cart_to_all(chan: dict[str, Any]) -> None:
    x = float(chan.get("X", 0) or 0)
    y = float(chan.get("Y", 0) or 0)
    z = float(chan.get("Z", 0) or 0)
    hypot_xy = float(np.hypot(x, y))
    sph_theta = np.degrees(np.arctan2(y, x))
    sph_phi = np.degrees(np.arctan2(z, hypot_xy))
    sph_radius = float(np.sqrt(x * x + y * y + z * z))
    chan.update(
        {
            "sph_theta": _wrap_degrees(sph_theta),
            "sph_phi": sph_phi,
            "sph_radius": sph_radius,
            "theta": _wrap_degrees(-sph_theta),
            "radius": 0.5 - sph_phi / 180.0,
        }
    )


def _sph_to_all(chan: dict[str, Any]) -> None:
    theta = np.radians(float(chan.get("sph_theta", 0) or 0))
    phi = np.radians(float(chan.get("sph_phi", 0) or 0))
    radius = float(chan.get("sph_radius", 1) or 1)
    chan.update(
        {
            "X": radius * np.cos(phi) * np.cos(theta),
            "Y": radius * np.cos(phi) * np.sin(theta),
            "Z": radius * np.sin(phi),
            "theta": _wrap_degrees(-np.degrees(theta)),
            "radius": 0.5 - np.degrees(phi) / 180.0,
        }
    )


def _topo_to_all(chan: dict[str, Any]) -> None:
    sph_theta = _wrap_degrees(-float(chan.get("theta", 0) or 0))
    sph_phi = (0.5 - float(chan.get("radius", 0.5) or 0.5)) * 180.0
    chan["sph_theta"] = sph_theta
    chan["sph_phi"] = sph_phi
    chan.setdefault("sph_radius", 1.0)
    _sph_to_all(chan)


def _read_chanloc_file(value: Any) -> list[dict[str, Any]]:
    path = Path(value[0] if isinstance(value, (list, tuple)) else value)
    rows = [_split_chanloc_row(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row]
    if not rows:
        return []
    header = [item.lower() for item in rows[0]]
    has_header = any(item in {"labels", "theta", "radius", "x", "y", "z"} for item in header)
    fields = [_canonical_channel_field(item) for item in rows.pop(0)] if has_header else ["labels", "theta", "radius"]
    chanlocs = []
    for row in rows:
        chanlocs.append({field: _parse_field_value(item) for field, item in zip(fields, row)})
    return chanlocs


def _split_chanloc_row(line: str) -> list[str]:
    text = line.strip()
    if not text or text.startswith(("%", "#")):
        return []
    if "," in text:
        return [item.strip() for item in next(csv.reader([text], skipinitialspace=True)) if item.strip()]
    return text.split()


def _canonical_channel_field(field: str) -> str:
    return _CHANNEL_FIELD_ALIASES.get(str(field).lower(), field)


def _write_chanloc_file(value: Any, chanlocs: list[dict[str, Any]]) -> None:
    path = Path(value)
    fields = [field for field in _CHANNEL_FIELDS if any(field in chan for chan in chanlocs)]
    lines = ["\t".join(fields)]
    for chan in chanlocs:
        lines.append("\t".join(str(chan.get(field, "")) for field in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _indices(value: Any, length: int) -> list[int]:
    if isinstance(value, np.ndarray):
        value = value.ravel().tolist()
    values = value if isinstance(value, (list, tuple)) else [value]
    return [_single_index(item, length) for item in values]


def _single_index(value: Any, length: int) -> int:
    index = int(value)
    if index < 1 or index > length:
        raise ValueError("channel indices must be 1-based and within EEG.nbchan")
    return index - 1


def _index_value_args(value: Any, length: int) -> tuple[list[int], Any]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("expected [indices, value]")
    return _indices(value[0], length), value[1]


def _as_command_list(value: Any) -> list[Any]:
    if isinstance(value, list) and value and all(isinstance(item, (list, tuple)) for item in value):
        return value
    return [value]


def _convert_mode(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    return str(value).lower().replace("->", "2").replace("xyz", "cart").replace("polar", "topo")


def _parse_field_value(value: str) -> Any:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    return int(numeric) if numeric.is_integer() else numeric


def _display_value(value: Any) -> str:
    if _is_blank(value):
        return ""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.5g}"
    return str(value)


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, str)) and len(value) == 0:
        return True
    return isinstance(value, np.ndarray) and value.size == 0


def _is_eeg_dataset(value: Any) -> bool:
    return isinstance(value, dict) and "chanlocs" in value


def _is_eeg_dataset_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_is_eeg_dataset(item) for item in value)


def _wrap_degrees(value: float) -> float:
    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    if np.isclose(wrapped, -180.0) and value > 0:
        return 180.0
    return wrapped


def _history_command(options: dict[str, Any]) -> str:
    if not options:
        return ""
    pieces = []
    for key, value in options.items():
        pieces.append(format_history_value(key))
        pieces.append(format_history_value(value, cell_for_sequence="any_strings"))
    return f"EEG = pop_chanedit(EEG, {', '.join(pieces)});"


__all__ = ["pop_chanedit", "pop_chanedit_dialog_spec"]
