"""EEGLAB-style channel-location editor."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from importlib.resources import files
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.adminfunc.eeg_checkchanlocs import _NO_DATA_TYPES, eeg_checkchanlocs
from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.pophelp import pophelp
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_numeric_sequence
from eegprep.functions.popfunc.pop_chancenter import pop_chancenter
from eegprep.functions.popfunc.plot_utils import show_figures
from eegprep.functions.sigprocfunc.forcelocs import forcelocs
from eegprep.functions.sigprocfunc.topoplot import topoplot
from eegprep.functions.sigprocfunc.convertlocs import convertlocs
from eegprep.functions.sigprocfunc.readlocs import readlocs
from eegprep.functions.sigprocfunc.writelocs import writelocs


_CHANNEL_FIELDS = ("labels", "theta", "radius", "X", "Y", "Z", "sph_theta", "sph_phi", "sph_radius", "type", "ref")
_LOCATION_FIELDS = ("theta", "radius", "X", "Y", "Z", "sph_theta", "sph_phi", "sph_radius")
_MAX_POWER_EXPONENT = 12
_MAX_EXACT_INTEGER = 2**53


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
    if len(args) % 2:
        raise ValueError("pop_chanedit: options must be name/value pairs. Supply a value after each option name.")
    options = [(str(key).lower(), value) for key, value in zip(args[::2], args[1::2])]
    options.extend((key.lower(), value) for key, value in kwargs.items())
    is_eeg = _is_eeg_dataset(chans)
    is_eeg_list = _is_eeg_dataset_list(chans)
    if gui is None:
        gui = not bool(options)
    if gui:
        gui_eeg = chans[0] if is_eeg_list else chans
        result = _run_gui(gui_eeg if is_eeg or is_eeg_list else {"chanlocs": gui_eeg}, renderer=renderer)
        if result is None:
            return (chans, "") if return_com else chans
        options.extend(result)
    if is_eeg_list:
        outputs = [_apply_chanedit(dataset, options, is_eeg=True)[0] for dataset in chans]
        command = _history_command(options)
        return (outputs, command) if return_com else outputs
    output, chaninfo, urchans = _apply_chanedit(chans, options, is_eeg=is_eeg)
    command = _history_command(options)
    return (output, command) if return_com else output


def pop_chanedit_dialog_spec(EEG: dict[str, Any], *, renderer: Any | None = None) -> DialogSpec:
    """Return the EEGLAB-like ``pop_chanedit`` dialog spec."""
    chanlocs = deepcopy(chanlocs_as_list(EEG.get("chanlocs")))
    chanlocs.extend(
        {**loc, "datachan": 0} for loc in deepcopy(chanlocs_as_list(EEG.get("chaninfo", {}).get("nodatchans", [])))
    )
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
    field_displays = [_channel_field_displays(chan, field_rows) for chan in chanlocs]
    state = {
        "chanlocs": chanlocs,
        "chaninfo": deepcopy(EEG.get("chaninfo", {})),
        "urchans": chanlocs_as_list(EEG.get("urchanlocs")),
        "commands": [],
        "displays": field_displays,
        "field_rows": field_rows,
        "navigation": [],
        "windows": [],
    }
    state["chaninfo"].pop("nodatchans", None)
    nav_base = {
        "channel_tag": "channel",
        "max_index": len(chanlocs),
        "field_displays": field_displays,
    }

    def _nav(tag: str, delta: int) -> CallbackSpec | None:
        params = {**nav_base, "delta": delta, "source": tag, "state": state}
        state["navigation"].append(params)
        return CallbackSpec("navigate_channel", params=params)

    channel_callback = _nav("channel", 0)

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
            value = _is_data_location(current)
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
            ControlSpec("pushbutton", "Delete chan", tag="delete_button"),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("text", f"Channel number (of {len(chanlocs)})", tag="chaneditscantitle", font_weight="bold"),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("pushbutton", "Insert chan", tag="insert_button"),
            ControlSpec("pushbutton", "<<", tag="back10", callback=_nav("back10", -10)),
            ControlSpec("pushbutton", "<", tag="back1", callback=_nav("back1", -1)),
            ControlSpec("edit", tag="channel", value="1", callback=channel_callback),
            ControlSpec("pushbutton", ">", tag="next1", callback=_nav("next1", 1)),
            ControlSpec("pushbutton", ">>", tag="next10", callback=_nav("next10", 10)),
            ControlSpec("pushbutton", "Append chan", tag="append_button"),
            ControlSpec("pushbutton", "Plot 2-D", tag="plot2d"),
            ControlSpec("text", "Plot radius (0.2-1, []=auto)"),
            ControlSpec("edit", tag="plotrad", value=_display_value(EEG.get("chaninfo", {}).get("plotrad", ""))),
            ControlSpec(
                "popupmenu",
                "Nose along +X|Nose along -X|Nose along +Y|Nose along -Y",
                tag="nosedir",
                value=_nosedir_index(EEG.get("chaninfo", {}).get("nosedir", "+X")),
            ),
            ControlSpec("pushbutton", "Plot 3-D (xyz)", tag="plot3d"),
            ControlSpec("spacer"),
            ControlSpec("pushbutton", "Read locations", tag="load"),
            ControlSpec("pushbutton", "Read locs help", tag="readhelp"),
            ControlSpec("pushbutton", "Look up locs", tag="lookup"),
            ControlSpec("pushbutton", "Save (as .ced)", tag="save"),
            ControlSpec("pushbutton", "Save (other types)", tag="saveothers"),
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
    controls = [
        replace(
            control,
            callback=CallbackSpec(
                "run_dialog_action",
                params={
                    "button": control.tag,
                    "function": "pop_chanedit",
                    "state": state,
                    "index_tag": "channel",
                    "action": partial(_run_chanedit_action, control.tag, state, renderer=renderer),
                },
            ),
        )
        if control.style == "pushbutton" and control.callback is None
        else control
        for control in controls
    ]
    return DialogSpec(
        title="Edit channel info -- pop_chanedit()",
        function_name="pop_chanedit",
        eeglab_source="functions/popfunc/pop_chanedit.m",
        size=(872, 900),
        content_margins=(42, 26, 42, 30),
        row_spacing=5,
        help_text="pophelp('pop_chanedit')",
        geometry=tuple(geometry),
        controls=tuple(controls),
        geomvert=tuple(1 for _row in geometry),
    )


def _run_gui(EEG: dict[str, Any], *, renderer: Any | None = None) -> list[tuple[str, Any]] | None:
    spec = pop_chanedit_dialog_spec(EEG, renderer=renderer)
    state = next(control.callback.params["state"] for control in spec.controls if control.callback is not None)
    locs = state["chanlocs"]
    if (
        locs
        and any(loc.get("labels") for loc in locs)
        and all(_is_blank(loc.get("theta")) and _is_blank(loc.get("X")) for loc in locs)
    ):
        commands = _prompt_chanedit_action("lookup", renderer)
        if commands:
            _apply_editor_commands(state, commands)
            spec = pop_chanedit_dialog_spec(
                {**EEG, "chanlocs": state["chanlocs"], "chaninfo": state["chaninfo"]}, renderer=renderer
            )
            state = next(control.callback.params["state"] for control in spec.controls if control.callback is not None)
            state["commands"].extend(commands)
    result = inputgui(spec, renderer=renderer)
    if result is None:
        return None
    _store_form_edits(state, result)
    if state["commands"] and result.get("rplurchan"):
        state["commands"].append(("rplurchanloc", 1))
    return state["commands"]


def _store_form_edits(state: dict, values: Mapping[str, Any]) -> int:
    locs = state["chanlocs"]
    try:
        number = float(values.get("channel") or 1)
        channel = int(number)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "pop_chanedit: channel number must be an integer. Enter a channel number and press Enter."
        ) from exc
    if number != channel or (locs and not 1 <= channel <= len(locs)):
        raise ValueError(
            f"pop_chanedit: channel number {channel} is outside 1 to {len(locs)}. Enter a number in that range."
        )
    commands = []
    for index, current in enumerate(locs, start=1):
        source = values if index == channel else state["displays"][index - 1]
        for field in _CHANNEL_FIELDS:
            text = str(source.get(f"field_{field}") or "").strip()
            if text != _display_channel_value(current, field).strip():
                value = _validated_field_value(field, _parse_field_value(text))
                commands.append(("changefield", [index, field, value]))
    plotrad = str(values.get("plotrad") or "").strip()
    if plotrad != _display_value(state["chaninfo"].get("plotrad", "")):
        radius = [] if plotrad in {"", "[]"} else _validated_field_value("radius", plotrad)
        if not _is_blank(radius) and not 0.2 <= radius <= 1:
            raise ValueError(
                "pop_chanedit: plot radius must be in 0.2..1. Enter a radius in that range or [] for automatic."
            )
        commands.append(("plotrad", radius))
    direction = ["+X", "-X", "+Y", "-Y"][int(values.get("nosedir", 1)) - 1]
    if direction != state["chaninfo"].get("nosedir", "+X"):
        commands.append(("nosedir", direction))
    _apply_editor_commands(state, commands)
    return channel


def _apply_editor_commands(state: dict, commands: list[tuple[str, Any]]) -> None:
    locs, info, urchans = _edit_locations(
        deepcopy(state["chanlocs"]),
        deepcopy(state["chaninfo"]),
        deepcopy(state["urchans"]),
        commands,
    )
    state.update(chanlocs=locs, chaninfo=info, urchans=urchans)
    state["commands"].extend(commands)


def _run_chanedit_action(tag, state, values, update, *, renderer=None):
    channel = _store_form_edits(state, values)
    commands = []
    conversions = {
        "button_xyz_->_polar_&_sph.": "cart2all",
        "button_sph._->_polar_&_xyz": "sph2all",
        "button_polar_->_sph._&_xyz": "topo2all",
    }
    if tag in conversions:
        commands = [("convert", conversions[tag])]
    elif tag in {"delete_button", "insert_button", "append_button"}:
        operation = tag.removesuffix("_button")
        commands = [(operation, channel if operation == "delete" else [channel])]
    elif tag in {"plot2d", "plot3d"}:
        _plot_locations(state["chanlocs"], tag, state["chaninfo"])
    elif tag == "readhelp":
        state["windows"].append(pophelp("pop_readlocs"))
    else:
        commands = _prompt_chanedit_action(tag, renderer)
    try:
        _apply_editor_commands(state, commands)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"pop_chanedit: {exc} Check the channel locations and the values entered for this action."
        ) from exc
    locs = state["chanlocs"]
    state["displays"][:] = [_channel_field_displays(loc, state["field_rows"]) for loc in locs]
    for params in state["navigation"]:
        params["max_index"] = len(locs)
    channel = min(channel, len(locs)) if locs else 1
    display = _channel_field_displays(locs[channel - 1] if locs else {}, state["field_rows"])
    update(
        {
            **display,
            "channel": str(channel),
            "chaneditscantitle": f"Channel number (of {len(locs)})",
            "plotrad": _display_value(state["chaninfo"].get("plotrad", "")),
        }
    )


def _prompt_chanedit_action(tag, renderer):
    file_actions = {"load", "lookup", "save", "saveothers"}
    if tag in file_actions:
        default = (
            str(files("eegprep").joinpath("resources/montages/standard-10-5-342ch.locs")) if tag == "lookup" else ""
        )
        rows = [("Location file", "filename", default)]
        if tag == "saveothers":
            rows.append(("File type (loc, sph, sfp, xyz, besa, chanedit)", "filetype", "loc"))
    else:
        rows = {
            "button_opt._head_center": [("Head center [X Y Z] ([] = optimize)", "center", "[]")],
            "button_rotate_axis": [
                ("X/Y value", "value", "0"),
                ("Axis (X or Y)", "axis", "X"),
                ("Channel labels (space separated)", "labels", ""),
            ],
            "button_transform_axes": [("Enter transform (e.g. TMP=X; X=-Y; Y=TMP)", "transform", "")],
            "button_set_head_radius": [("New head radius (location units)", "radius", "85")],
            "button_set_channel_types": [("Channel indices", "indices", ""), ("Type (e.g. EEG)", "text", "")],
            "button_set_reference": [("Channel indices", "indices", ""), ("Reference (e.g. Cz)", "text", "")],
        }[tag]
    controls = []
    geometry = []
    for label, field, default in rows:
        controls.extend([ControlSpec("text", label), ControlSpec("edit", tag=field, value=default)])
        geometry.append((1.6, 1))
    if tag in file_actions:
        controls.append(
            ControlSpec(
                "pushbutton",
                "Browse",
                tag="browse",
                callback=CallbackSpec(
                    "select_file",
                    params={
                        "button": "browse",
                        "target": "filename",
                        "mode": "save" if tag.startswith("save") else "open",
                        "filter": "Channel locations (*.loc *.locs *.ced *.elp *.elc *.sfp *.xyz *.sph);;All files (*)",
                    },
                ),
            )
        )
        geometry.append((1,))
    result = inputgui(
        DialogSpec(
            title=f"Channel locations: {tag.removeprefix('button_').replace('_', ' ')} -- pop_chanedit()",
            function_name=f"chanedit_{tag}",
            eeglab_source="functions/popfunc/pop_chanedit.m",
            controls=tuple(controls),
            geometry=tuple(geometry),
            help_text="pophelp('pop_chanedit')",
        ),
        renderer=renderer,
    )
    if result is None:
        return []
    if tag in file_actions:
        filename = str(result.get("filename", "")).strip()
        if not filename:
            return []
        if tag == "saveothers":
            filetype = str(result["filetype"]).strip().lower()
            if filetype not in {"loc", "sph", "sfp", "xyz", "besa", "chanedit"}:
                raise ValueError(
                    "pop_chanedit: unsupported output file type. Choose loc, sph, sfp, xyz, besa or chanedit."
                )
            return [(tag, [filename, "filetype", filetype])]
        return [(tag, filename)]
    if tag == "button_opt._head_center":
        center = parse_numeric_sequence(result["center"], dtype=float)
        if len(center) not in {0, 3}:
            raise ValueError("pop_chanedit: head center needs three numbers. Enter [X Y Z] or [] to optimize.")
        return [("convert", ["chancenter", center])]
    if tag == "button_rotate_axis":
        return [
            (
                "forcelocs",
                [
                    _validated_field_value("X", result["value"]),
                    str(result["axis"]).lower(),
                    *str(result["labels"]).split(),
                ],
            )
        ]
    if tag == "button_transform_axes":
        return [("transform", result["transform"])]
    if tag == "button_set_head_radius":
        return [("headrad", _validated_field_value("sph_radius", result["radius"]))]
    indices = parse_numeric_sequence(result["indices"], dtype=float)
    key = "settype" if tag == "button_set_channel_types" else "setref"
    return [(key, [indices, result["text"]])]


def _plot_locations(locs, kind, info):
    located = [loc for loc in locs if all(not _is_blank(loc.get(axis)) for axis in "XYZ")]
    if not located:
        raise ValueError(
            "pop_chanedit: no Cartesian channel locations are available. Read, look up or convert locations first."
        )
    if kind == "plot2d":
        figure, axes = plt.subplots()
        topoplot([], located, electrodes="labelpoint", axes=axes)
        radius = info.get("plotrad")
        if not _is_blank(radius):
            axes.set_xlim(-float(radius) * 1.2, float(radius) * 1.2)
            axes.set_ylim(-float(radius) * 1.2, float(radius) * 1.3)
    else:
        figure = plt.figure()
        axes = figure.add_subplot(projection="3d")
        xyz = np.array([[loc[axis] for axis in "XYZ"] for loc in located])
        axes.scatter(*xyz.T)
        for loc, (x, y, z) in zip(located, xyz):
            axes.text(x, y, z, str(loc.get("labels", "")))
        axes.set(xlabel="X", ylabel="Y", zlabel="Z", title="Channel locations")
        axes.set_box_aspect((1, 1, 1))
    show_figures(figure)


def _transform_locations(locs, expression):
    try:
        tree = ast.parse(expression.strip(), mode="exec")
        allowed = (
            ast.Module,
            ast.Assign,
            ast.Name,
            ast.Load,
            ast.Store,
            ast.Constant,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.USub,
            ast.UAdd,
        )
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                raise ValueError("unsupported expression")
            if isinstance(node, ast.Constant):
                if type(node.value) not in (int, float):
                    raise ValueError("constants must be numeric")
                if isinstance(node.value, int) and abs(node.value) > _MAX_EXACT_INTEGER:
                    raise ValueError("integer constant exceeds the exact float range")
                node.value = float(node.value)
                if not np.isfinite(node.value):
                    raise ValueError("constants must be finite")
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                exponent = ast.literal_eval(node.right)
                if type(exponent) not in (int, float) or abs(exponent) > _MAX_POWER_EXPONENT:
                    raise ValueError(f"power exponents must be numeric constants within +/-{_MAX_POWER_EXPONENT}")
        coordinates = {
            field: np.array([loc.get(field, np.nan) for loc in locs], dtype=float) for field in _LOCATION_FIELDS
        }
        exec(compile(tree, "<channel transform>", "exec"), {"__builtins__": {}}, coordinates)
        assigned = [target.id for statement in tree.body for target in statement.targets]
        changed = [field for field in assigned if field in _LOCATION_FIELDS]
        if not changed:
            raise ValueError("no coordinate field was assigned")
        for field in changed:
            values = np.broadcast_to(coordinates[field], (len(locs),))
            for loc, value in zip(locs, values):
                loc[field] = _validated_field_value(field, value)
        field = changed[-1]
        mode = "cart2all" if field in "XYZ" else "sph2all" if field.startswith("sph_") else "topo2all"
        return convertlocs(locs, mode)
    except (SyntaxError, ValueError, TypeError, NameError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(
            "pop_chanedit: invalid coordinate transform. Use numeric assignments such as X=-Y or TMP=X; X=-Y; Y=TMP."
        ) from exc


def _channel_value_control(field: str, value: Any) -> ControlSpec:
    if field == "datachan":
        return ControlSpec("checkbox", tag="field_datachan", value=bool(value), enabled=False)
    if field == "urchan":
        return ControlSpec("text", _display_value(value), tag="field_urchan")
    return ControlSpec("edit", tag=f"field_{field}", value=_display_value(value))


def _channel_field_displays(chan: dict[str, Any], field_rows: list[tuple[str, str, str]]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for _label, field, _button_label in field_rows:
        if field == "urchan":
            values["field_urchan"] = _display_value(chan.get("urchan", ""))
        elif field == "datachan":
            values["field_datachan"] = _is_data_location(chan)
        else:
            values[f"field_{field}"] = _display_channel_value(chan, field)
    return values


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
    return ControlSpec("pushbutton", label, tag=f"button_{label.lower().replace(' ', '_')}")


def _nosedir_index(value: Any) -> int:
    choices = ["+X", "-X", "+Y", "-Y"]
    try:
        return choices.index(str(value)) + 1
    except ValueError:
        return 1


def _is_data_location(loc: dict[str, Any]) -> bool:
    if "datachan" in loc:
        return bool(loc["datachan"])
    return str(loc.get("type", "")).strip().lower() not in _NO_DATA_TYPES


def _apply_chanedit(
    chans: dict[str, Any] | list[dict[str, Any]],
    options: list[tuple[str, Any]],
    *,
    is_eeg: bool,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    eeg = deepcopy(chans) if is_eeg and isinstance(chans, dict) else None
    chanlocs = deepcopy(chanlocs_as_list(eeg.get("chanlocs") if eeg is not None else chans))
    chaninfo = deepcopy(eeg.get("chaninfo", {})) if eeg is not None else {}
    urchans = chanlocs_as_list(eeg.get("urchanlocs")) if eeg is not None else []

    chanlocs.extend({**loc, "datachan": 0} for loc in chanlocs_as_list(chaninfo.pop("nodatchans", [])))
    chanlocs, chaninfo, urchans = _edit_locations(chanlocs, chaninfo, urchans, options)
    data_locs = [loc for loc in chanlocs if _is_data_location(loc)]
    chaninfo["nodatchans"] = [loc for loc in chanlocs if not _is_data_location(loc)]
    if eeg is not None and "data" in eeg and len(data_locs) != np.asarray(eeg["data"]).shape[0]:
        raise ValueError(
            "pop_chanedit: location count does not match data channels. Use pop_select to remove data channels, or mark added reference locations as non-data channels."
        )
    if eeg is not None:
        chanlocs = data_locs
        eeg["chanlocs"] = chanlocs
        eeg["urchanlocs"] = urchans
        eeg["chaninfo"] = chaninfo
        eeg["nbchan"] = len(chanlocs)
        eeg["saved"] = "no"
        return eeg_checkset(eeg), chaninfo, urchans
    return chanlocs, chaninfo, urchans


def _edit_locations(chanlocs, chaninfo, urchans, options):
    for key, value in options:
        if key == "changefield":
            for item in _as_command_list(value):
                _change_field(chanlocs, item)
        elif key in {"insert", "append"}:
            index, new_chan = _new_channel_args(value)
            new_chan.setdefault("datachan", False)
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
        elif key == "lookup":
            chanlocs = _lookup_chanlocs(chanlocs, value)
            chaninfo["filename"] = str(_channel_location_path(value))
            lookup_name = chaninfo["filename"].lower()
            chaninfo["nosedir"] = "+X"
            if "standard_10" in lookup_name and lookup_name.endswith(".elc"):
                chaninfo["originalnosedir"] = "+Y"
        elif key == "save":
            _write_chanloc_file(value, chanlocs)
        elif key == "saveothers":
            writelocs(chanlocs, value[0], *value[1:])
        elif key == "headrad":
            radius = _validated_field_value("sph_radius", value)
            if not radius or radius < 0:
                raise ValueError(
                    "pop_chanedit: head radius must be positive. Enter the radius in the location file's units."
                )
            chanlocs = convertlocs(chanlocs, "cart2all")
            radii = [float(chan["sph_radius"]) for chan in chanlocs if not _is_blank(chan.get("sph_radius"))]
            if not radii or np.mean(radii) <= 0:
                raise ValueError(
                    "pop_chanedit: no nonzero channel radius exists. Read or look up channel locations first."
                )
            for chan in chanlocs:
                if not _is_blank(chan.get("sph_radius")):
                    chan["sph_radius"] *= radius / np.mean(radii)
            chanlocs = convertlocs(chanlocs, "sph2all")
        elif key == "forcelocs":
            chanlocs = forcelocs(chanlocs, value)
        elif key == "transform":
            chanlocs = _transform_locations(chanlocs, str(value))
        elif key == "shrink":
            if chanlocs:
                chanlocs[0]["shrink"] = float(value)
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
            if value is True or value == 1 or value == "on":
                urchans = deepcopy(chanlocs)
        else:
            raise ValueError(f"Unsupported pop_chanedit option: {key}")

    return chanlocs, chaninfo, urchans


def _change_field(chanlocs: list[dict[str, Any]], item: Any) -> None:
    if not isinstance(item, (list, tuple)) or len(item) != 3:
        raise ValueError("changefield requires [channel, field, value]")
    index = _single_index(item[0], len(chanlocs))
    field = str(item[1])
    chanlocs[index][field] = _validated_field_value(field, item[2])


def _change_channel(chanlocs: list[dict[str, Any]], value: Any) -> None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError("changechan requires [channel, label, ...]")
    index = _single_index(value[0], len(chanlocs))
    for field, item in zip(_CHANNEL_FIELDS, value[1:]):
        chanlocs[index][field] = _validated_field_value(field, item)


def _validated_field_value(field: str, value: Any) -> Any:
    if field not in _LOCATION_FIELDS:
        return value
    if _is_blank(value) or (isinstance(value, str) and value.strip() == "[]"):
        return ""
    try:
        number = float(value)
    except (ValueError, TypeError):
        number = np.nan
    if not np.isfinite(number):
        raise ValueError(f"pop_chanedit: {field} must be numeric and finite. Enter a number or leave the field blank.")
    return number


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
    if mode == "chancenter":
        center = value[1] if isinstance(value, (list, tuple)) and len(value) > 1 else None
        chanlocs[:] = pop_chancenter(chanlocs, center)
    else:
        chanlocs[:] = convertlocs(chanlocs, mode)


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


def _read_chanloc_file(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, (list, tuple)):
        path = Path(value[0])
        return readlocs(path, *value[1:])
    return readlocs(Path(value))


def _lookup_chanlocs(chanlocs: list[dict[str, Any]], value: Any) -> list[dict[str, Any]]:
    template = _read_chanloc_file(value)
    lookup_name = str(_channel_location_path(value)).lower()
    if "standard_10" in lookup_name and lookup_name.endswith(".elc"):
        _, _, template = eeg_checkchanlocs(template, {"nosedir": "+Y"})
    by_label: dict[str, dict[str, Any]] = {}
    for channel in template:
        label = str(channel.get("labels") or "")
        if label.strip():
            by_label.setdefault(label.casefold(), channel)
    if not any(str(channel.get("labels") or "").casefold() in by_label for channel in chanlocs):
        return deepcopy(chanlocs)
    looked_up: list[dict[str, Any]] = []
    for index, channel in enumerate(chanlocs):
        label = str(channel.get("labels") or "")
        output = {
            "labels": label,
            "datachan": _is_data_location(channel),
            "type": channel.get("type", ""),
            "urchan": index + 1,
            "ref": "",
        }
        match = by_label.get(label.casefold())
        if match is not None:
            output.update({field: match[field] for field in _LOCATION_FIELDS if field in match})
            if _is_blank(output["type"]) and not _is_blank(match.get("type")):
                output["type"] = match["type"]
        looked_up.append(output)
    return looked_up


def _channel_location_path(value: Any) -> Path:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    return Path(value)


def _write_chanloc_file(value: Any, chanlocs: list[dict[str, Any]]) -> None:
    writelocs(chanlocs, value, "filetype", "chanedit", "header", "on")


def _indices(value: Any, length: int) -> list[int]:
    if isinstance(value, np.ndarray):
        value = value.ravel().tolist()
    values = value if isinstance(value, (list, tuple)) else [value]
    return [_single_index(item, length) for item in values]


def _single_index(value: Any, length: int) -> int:
    number = float(value)
    if not np.isfinite(number) or not number.is_integer() or not 1 <= number <= length:
        raise ValueError(
            f"pop_chanedit: channel index {value} is outside 1..{length}. Enter a whole channel number in that range."
        )
    index = int(number)
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


def _history_command(options: list[tuple[str, Any]]) -> str:
    if not options:
        return ""
    pieces = []
    for key, value in options:
        if key == "changefield":
            for item in _as_command_list(value):
                pieces.extend(["'changefield'", "{" + " ".join(format_history_value(part) for part in item) + "}"])
            continue
        pieces.append(format_history_value(key))
        pieces.append(format_history_value(value, cell_for_sequence="any_strings"))
    return f"EEG = pop_chanedit(EEG, {', '.join(pieces)});"


__all__ = ["pop_chanedit", "pop_chanedit_dialog_spec"]
