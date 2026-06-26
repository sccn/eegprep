"""Store EEGLAB-style DIPFIT head-model settings on an EEG dataset."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.popfunc._plot_utils import numeric_vector
from eegprep.plugins.dipfit._utils import (
    STANDARD_TEMPLATES,
    apply_template,
    coordinate_channel_indices,
    copy_eeg,
    dipfit_history,
    one_based_indices,
    template_by_name,
)


def pop_dipfit_settings(
    EEG: dict[str, Any] | None = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Configure DIPFIT settings without requiring an EEGLAB runtime.

    EEGPrep stores the same key DIPFIT fields that EEGLAB records on
    ``EEG.dipfit``. The standard spherical fitting path is standalone; BEM
    head-model assets and MRI co-registration remain explicit backend limits.
    """
    if EEG is None:
        return (None, "") if return_com else None
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if gui is None:
        gui = not options
    if gui:
        dialog_eeg = EEG[0] if isinstance(EEG, list) and EEG else EEG
        gui_options = _run_gui(dialog_eeg, renderer=renderer)
        if gui_options is None:
            return (EEG, "") if return_com else EEG
        options.update(gui_options)
    if isinstance(EEG, list):
        out = [_apply_settings(dataset, options) for dataset in EEG]
    else:
        out = _apply_settings(EEG, options)
    command = _history_command(options)
    return (out, command) if return_com else out


def pop_dipfit_settings_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like settings dialog spec."""
    raw_dipfit = EEG.get("dipfit")
    dipfit: dict[str, Any] = raw_dipfit if isinstance(raw_dipfit, dict) else {}
    selected_model = _template_index(dipfit)
    template = STANDARD_TEMPLATES[selected_model - 1]
    hdmfile = str(dipfit.get("hdmfile") or template.hdmfile)
    mrifile = str(dipfit.get("mrifile") or template.mrifile)
    chanfile = str(dipfit.get("chanfile") or template.chanfile)
    coordformat = str(dipfit.get("coordformat") or template.coordformat or "MNI")
    coord_index = {"spherical": 1, "mni": 2}.get(coordformat.lower(), 4)
    coord_transform = dipfit.get("coord_transform", template.coord_transform)
    coreg_text = " ".join(f"{float(value):g}" for value in coord_transform) if coord_transform else ""
    return DialogSpec(
        title="Dipole fit settings - pop_dipfit_settings()",
        function_name="pop_dipfit_settings",
        eeglab_source="plugins/dipfit/pop_dipfit_settings.m",
        help_text="pophelp('pop_dipfit_settings')",
        size=(1170, 404),
        geometry=(
            (1, 2),
            (0.25, 1, 1.3, 0.8, 0.6),
            (0.25, 1, 1.3, 0.8, 0.6),
            (0.25, 1, 1.3, 0.8, 0.6),
            (0.25, 1, 1.3, 0.8, 0.6),
            (1,),
            (1, 1, 0.8, 0.7),
            (1, 1, 0.5, 0.8),
        ),
        controls=(
            ControlSpec("text", "Select a head model", font_weight="bold"),
            ControlSpec(
                "popupmenu",
                "|".join(template.name for template in STANDARD_TEMPLATES),
                tag="model",
                value=selected_model,
            ),
            ControlSpec("spacer"),
            ControlSpec("text", "Output coordinates"),
            ControlSpec(
                "popupmenu",
                "spherical (head radius 85 mm)|MNI|Neuromag (MEG only)|Custom",
                tag="coordformat",
                value=coord_index,
            ),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("text", "Head model file"),
            ControlSpec("edit", tag="hdmfile", value=hdmfile),
            ControlSpec("pushbutton", "Browse", tag="hdmfile_browse", enabled=False),
            ControlSpec("pushbutton", "Help", tag="hdmfile_help"),
            ControlSpec("spacer"),
            ControlSpec("text", "Associated MRI file for plotting"),
            ControlSpec("edit", tag="mrifile", value=mrifile),
            ControlSpec("pushbutton", "Browse", tag="mrifile_browse", enabled=False),
            ControlSpec("pushbutton", "Help", tag="mrifile_help"),
            ControlSpec("spacer"),
            ControlSpec("text", "Associated coord. landmarks if any"),
            ControlSpec("edit", tag="chanfile", value=chanfile),
            ControlSpec("pushbutton", "Browse", tag="chanfile_browse", enabled=False),
            ControlSpec("pushbutton", "Help", tag="chanfile_help"),
            ControlSpec("spacer"),
            ControlSpec("text", "Matrix to align chan. locs. with head model"),
            ControlSpec("edit", tag="coord_transform", value=coreg_text),
            ControlSpec(
                "pushbutton",
                "Co-register",
                tag="coregister",
                enabled=True,
                font_weight="bold",
                callback=CallbackSpec(
                    "headplot_manual_coreg",
                    params={
                        "transform_target": "coord_transform",
                        "reference_source": "chanfile",
                        "mesh_source": "hdmfile",
                        "chanlocs": EEG.get("chanlocs", ()),
                        "chaninfo": EEG.get("chaninfo", {}),
                        "title": "Coregister channels for DIPFIT",
                    },
                ),
            ),
            ControlSpec("checkbox", "No Co-Reg.", tag="no_coreg", value=not bool(coreg_text)),
            ControlSpec("text", "Channels to omit from dipole fitting"),
            ControlSpec("edit", tag="chanomit", value=""),
            ControlSpec("pushbutton", "...", tag="chanomit_button", enabled=False),
            ControlSpec("spacer"),
        ),
        known_differences=(
            "EEGPrep records DIPFIT settings and uses them for native spherical fitting.",
            "Browse buttons are disabled until standalone MRI/BEM asset loading is implemented.",
        ),
        row_spacing=10,
    )


def _run_gui(EEG: dict[str, Any], renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_dipfit_settings_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    template = STANDARD_TEMPLATES[int(result.get("model", 2)) - 1]
    coord_formats = ("Spherical", "MNI", "Neuromag", "")
    coord_index = int(result.get("coordformat", 2)) - 1
    options: dict[str, Any] = {
        "model": template.shortname,
        "hdmfile": result.get("hdmfile", ""),
        "mrifile": result.get("mrifile", ""),
        "chanfile": result.get("chanfile", ""),
        "coordformat": coord_formats[coord_index] if coord_index < len(coord_formats) else "",
    }
    if not result.get("no_coreg", False):
        options["coord_transform"] = result.get("coord_transform", "")
    if str(result.get("chanomit", "")).strip():
        options["chanomit"] = result["chanomit"]
    return options


def _apply_settings(EEG: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    out = copy_eeg(EEG)
    existing = out.get("dipfit")
    dipfit = dict(existing) if isinstance(existing, dict) else {}
    if "model" in options and str(options["model"]).strip():
        apply_template(dipfit, template_by_name(options["model"]))
    for key in ("hdmfile", "mrifile", "chanfile", "coordformat"):
        value = options.get(key)
        if value not in (None, ""):
            dipfit[key] = value
    if "coord_transform" in options:
        transform = _numeric_list(options["coord_transform"])
        if transform:
            dipfit["coord_transform"] = transform
    usable = coordinate_channel_indices(out)
    nbchan = int(out.get("nbchan", len(usable)) or len(usable))
    if "chanomit" in options and str(options.get("chanomit", "")).strip():
        omit = set(one_based_indices(options["chanomit"], limit=nbchan))
        chansel = [index for index in range(1, nbchan + 1) if index not in omit]
    elif "chansel" in options:
        chansel = one_based_indices(options["chansel"], limit=nbchan, default_all=True)
    else:
        chansel = list(range(1, nbchan + 1))
    if "electrodes" in options:
        chansel = one_based_indices(options["electrodes"], limit=nbchan, default_all=True)
    dipfit["chansel"] = [index for index in chansel if index in usable]
    if not dipfit["chansel"]:
        raise ValueError("No channel locations with coordinates remain for DIPFIT")
    out["dipfit"] = dipfit
    return out


def _numeric_list(value: Any) -> list[float]:
    if value is None or str(value).strip() == "":
        return []
    return numeric_vector(value, dtype=float).tolist()


def _index_list(value: Any) -> list[int]:
    if value is None or str(value).strip() == "":
        return []
    return numeric_vector(value, dtype=int).tolist()


def _template_index(dipfit: dict[str, Any]) -> int:
    hdmfile = str(dipfit.get("hdmfile") or "").lower()
    for index, template in enumerate(STANDARD_TEMPLATES, start=1):
        if template.hdmfile.lower() == hdmfile or template.shortname.lower() == str(dipfit.get("model", "")).lower():
            return index
    return 2


def _history_command(options: dict[str, Any]) -> str:
    ordered: dict[str, Any] = {}
    for key in (
        "model",
        "hdmfile",
        "mrifile",
        "chanfile",
        "coordformat",
        "coord_transform",
        "chanomit",
        "chansel",
        "electrodes",
    ):
        if key in options:
            ordered[key] = (
                _numeric_list(options[key])
                if key == "coord_transform"
                else _index_list(options[key])
                if key in {"chanomit", "chansel", "electrodes"}
                else options[key]
            )
    return dipfit_history("pop_dipfit_settings", **ordered)


__all__ = ["pop_dipfit_settings", "pop_dipfit_settings_dialog_spec"]
