"""EEGLAB-style wrapper for 3-D scalp maps."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc._plot_utils import (
    component_maps,
    data_time_slice,
    eeg_times_ms,
    history_command,
    numeric_vector,
    parse_plot_options_text,
)
from eegprep.functions.sigprocfunc.headplot import (
    DEFAULT_MESH,
    MAPLIMIT_PADDING,
    default_headplot_mesh_transform,
    headplot,
    headplot_setup,
)

_MESH_CHOICES = ("mheadnew.mat", "colin27headmesh.mat")
_MESH_CHANNEL_CHOICES = ("mheadnew.xyz", "colin27headmesh.xyz")
_DEFAULT_COMPONENT_ITEMS = "1"


def pop_headplot(
    EEG: dict[str, Any] | None = None,
    typeplot: int = 1,
    items: Any = None,
    topotitle: str | None = None,
    rowcols: Any = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Plot ERP or component maps on a spline-interpolated 3-D head mesh.

    Like EEGLAB, ``pop_headplot`` requires a ``.spl`` spline setup file. Use
    ``setup={...}`` to create one, ``load=...`` to reuse one, or launch the GUI
    to select/create it interactively.
    """
    if EEG is None:
        return ([], "") if return_com else []

    typeplot = int(typeplot)
    if typeplot not in {0, 1}:
        raise ValueError("typeplot must be 1 for ERP maps or 0 for component maps")
    _validate_chanlocs(EEG)
    if gui is None:
        gui = items is None
    if gui:
        result = _run_gui(EEG, typeplot=typeplot, renderer=renderer)
        if result is None:
            return ([], "") if return_com else []
        items = result["items"]
        topotitle = result["topotitle"]
        rowcols = result["rowcols"]
        kwargs = {**kwargs, **result["options"]}
    elif items is None:
        raise ValueError("items must be provided when gui=False")

    options = _parse_options(args, kwargs)
    setup = options.pop("setup", None)
    load = options.pop("load", None)
    colorbar = str(options.pop("colorbar", "on")).lower() != "off"
    splinefile = _prepare_spline_file(EEG, typeplot, setup=setup, load=load)
    items_array = numeric_vector(items)
    if items_array.size == 0:
        raise ValueError("please choose at least one latency or component to plot")
    rows, columns = _normalise_rowcols(rowcols, items_array.size)
    requested_maplimits = options.pop("maplimits", None)
    maplimits = requested_maplimits
    if maplimits is None and typeplot:
        maplimits = _erp_maplimits(EEG, items_array)
    elif maplimits is None:
        maplimits = [-1, 1]
    topotitle = _default_title(EEG, typeplot) if topotitle is None else str(topotitle)

    maps, labels = _headplot_maps(EEG, typeplot, items_array)
    maplimits = _shared_maplimits(maps, maplimits)
    figures = _plot_pages(
        maps,
        labels,
        splinefile,
        topotitle=topotitle,
        rowcols=(rows, columns),
        colorbar=colorbar,
        maplimits=maplimits,
        options=options,
    )
    command_options = dict(options)
    if requested_maplimits is not None:
        command_options["maplimits"] = requested_maplimits
    command = _history_command(
        typeplot, items_array, topotitle, [rows, columns], colorbar, setup, load, command_options
    )
    return (figures, command) if return_com else figures


def pop_headplot_dialog_spec(EEG: dict[str, Any], *, typeplot: int = 1) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_headplot``."""
    _validate_chanlocs(EEG)
    compute_file = not _spline_file_is_ready(EEG, typeplot)
    spline_value = str(_current_spline_file(EEG, typeplot) or "")
    default_mesh = _MESH_CHOICES[0]
    transform = _default_transform_text(EEG, default_mesh)
    mesh_transforms = tuple(_default_transform_text(EEG, mesh) for mesh in _MESH_CHOICES)
    setup_file = _default_spline_path(EEG)
    plot_label = (
        f"Making headplots for these latencies (from {round(float(EEG.get('xmin', 0)) * 1000)} "
        f"to {round(float(EEG.get('xmax', 0)) * 1000)} ms):"
        if int(typeplot)
        else "Component numbers to plot (negative numbers invert comp. polarities):"
    )
    controls = (
        ControlSpec(
            "text",
            "Co-register channel locations with head mesh and compute a mesh spline file "
            "(each scalp montage needs a headplot() spline file)",
            font_weight="bold",
        ),
        ControlSpec(
            "checkbox",
            "Use the following spline file or structure",
            tag="loadcb",
            value=not compute_file,
            callback=CallbackSpec(
                "headplot_setup_mode",
                params={
                    "source": "loadcb",
                    "mode": "load",
                    "peer": "compcb",
                    "load_targets": ("load", "load_browse"),
                    "setup_targets": _setup_tags(),
                },
                matlab_callback="cb_load",
            ),
        ),
        ControlSpec("edit", tag="load", value=spline_value, enabled=not compute_file),
        ControlSpec(
            "pushbutton",
            "Browse",
            tag="load_browse",
            enabled=not compute_file,
            callback=CallbackSpec(
                "select_file",
                params={
                    "button": "load_browse",
                    "target": "load",
                    "mode": "open",
                    "caption": "Select a spline file",
                    "filter": "Spline files (*.spl)",
                },
                matlab_callback="uigetfile('*.spl')",
            ),
        ),
        ControlSpec(
            "pushbutton",
            "Help",
            tag="load_help",
            callback=CallbackSpec(
                "show_message",
                params={
                    "button": "load_help",
                    "title": "Load file for headplot()",
                    "message": (
                        "If you already generated a spline file for this channel-location structure, enter it here. "
                        "Enable 'Use the following spline file or structure' first."
                    ),
                },
                matlab_callback="cb_helpload",
            ),
        ),
        ControlSpec(
            "checkbox",
            "Or (re)compute a new spline file named:",
            tag="compcb",
            value=compute_file,
            callback=CallbackSpec(
                "headplot_setup_mode",
                params={
                    "source": "compcb",
                    "mode": "compute",
                    "peer": "loadcb",
                    "load_targets": ("load", "load_browse"),
                    "setup_targets": _setup_tags(),
                },
                matlab_callback="cb_comp",
            ),
        ),
        ControlSpec("edit", tag="setup_file", value=setup_file, enabled=compute_file),
        ControlSpec(
            "pushbutton",
            "Browse",
            tag="setup_browse",
            enabled=compute_file,
            callback=CallbackSpec(
                "select_file",
                params={
                    "button": "setup_browse",
                    "target": "setup_file",
                    "mode": "save",
                    "caption": "Select a spline file",
                    "filter": "Spline files (*.spl)",
                },
                matlab_callback="uiputfile('*.spl')",
            ),
        ),
        ControlSpec(
            "pushbutton",
            "Help",
            tag="setup_help",
            callback=CallbackSpec(
                "show_message",
                params={
                    "button": "setup_help",
                    "title": "Coregister channels for headplot()",
                    "message": (
                        "Channel locations must be co-registered with a 3-D head mesh before plotting. "
                        "Template montages fill this transform automatically; otherwise enter a Talairach transform."
                    ),
                },
                matlab_callback="cb_helpcoreg",
            ),
        ),
        ControlSpec("text", "            3-D head mesh file", tag="mesh_label", enabled=compute_file),
        ControlSpec(
            "popupmenu",
            "|".join(_MESH_CHOICES),
            tag="meshfile",
            value=1,
            enabled=compute_file,
            callback=CallbackSpec(
                "headplot_mesh_choice",
                params={
                    "source": "meshfile",
                    "reference_target": "meshchanfile",
                    "transform_target": "transform",
                    "transform_choices": mesh_transforms,
                },
                matlab_callback="cb_selectmesh",
            ),
        ),
        ControlSpec(
            "pushbutton",
            "Browse other",
            tag="mesh_browse",
            enabled=compute_file,
            callback=CallbackSpec(
                "select_file",
                params={
                    "button": "mesh_browse",
                    "target": "meshfile",
                    "mode": "open",
                    "caption": "Select a head mesh file",
                    "filter": "MAT files (*.mat)",
                    "transform_target": "transform",
                    "custom_transform": _default_transform_text(EEG, default_mesh),
                },
                matlab_callback="uigetfile('*.mat')",
            ),
        ),
        ControlSpec("spacer"),
        ControlSpec("text", "            Mesh associated channel file", tag="meshchan_label", enabled=compute_file),
        ControlSpec("popupmenu", "|".join(_MESH_CHANNEL_CHOICES), tag="meshchanfile", value=1, enabled=compute_file),
        ControlSpec(
            "pushbutton",
            "Browse other",
            tag="meshchan_browse",
            enabled=compute_file,
            callback=CallbackSpec(
                "select_file",
                params={
                    "button": "meshchan_browse",
                    "target": "meshchanfile",
                    "mode": "open",
                    "caption": "Select a mesh channel file",
                    "filter": "Channel files (*.xyz *.sfp *.elp *.locs)",
                },
                matlab_callback="uigetfile('*.xyz')",
            ),
        ),
        ControlSpec("spacer"),
        ControlSpec(
            "text",
            "            Talairach-model transformation matrix",
            tag="transform_label",
            enabled=compute_file,
        ),
        ControlSpec("edit", tag="transform", value=transform, enabled=compute_file),
        ControlSpec(
            "pushbutton",
            "Manual coreg.",
            tag="manual_coreg",
            enabled=compute_file,
            callback=CallbackSpec(
                "headplot_manual_coreg",
                params={
                    "button": "manual_coreg",
                    "chanlocs": chanlocs_as_list(EEG.get("chanlocs", [])),
                    "chaninfo": dict(EEG.get("chaninfo") or {}),
                    "mesh_source": "meshfile",
                    "mesh_choices": _MESH_CHOICES,
                    "reference_source": "meshchanfile",
                    "reference_choices": _MESH_CHANNEL_CHOICES,
                    "transform_target": "transform",
                    "title": "Co-registration plot for headplot mesh",
                },
                matlab_callback="coregister(...)",
            ),
        ),
        ControlSpec("spacer"),
        ControlSpec("spacer"),
        ControlSpec("text", "Plot interpolated activity onto 3-D head", font_weight="bold"),
        ControlSpec("text", plot_label),
        ControlSpec("edit", tag="items", value="0" if int(typeplot) else _component_items_value(EEG)),
        ControlSpec("spacer"),
        ControlSpec("text", "Plot title:"),
        ControlSpec("edit", tag="topotitle", value=_default_title(EEG, typeplot)),
        ControlSpec("spacer"),
        ControlSpec("text", "Plot geometry (rows,columns): (Default [] = near square)"),
        ControlSpec("edit", tag="rowcols", value=""),
        ControlSpec("spacer"),
        ControlSpec("text", "Other headplot options (See >> help headplot):"),
        ControlSpec("edit", tag="options", value=""),
        ControlSpec("spacer"),
    )
    geometry: tuple[Any, ...] = (
        1,
        (1.3, 1.6, 0.5, 0.5),
        (1.3, 1.6, 0.5, 0.5),
        (1.3, 1.6, 0.6, 0.4),
        (1.3, 1.6, 0.6, 0.4),
        (1.3, 1.6, 0.6, 0.4),
        1,
        1,
        (1.5, 1, 0.5),
        (1.5, 1, 0.5),
        (1.5, 1, 0.5),
        (1.5, 1, 0.5),
    )
    return DialogSpec(
        title=("ERP" if int(typeplot) else "Component") + " head plot(s) -- pop_headplot()",
        controls=controls,
        geometry=geometry,
        function_name="pop_headplot",
        eeglab_source="functions/popfunc/pop_headplot.m",
        help_text="pophelp('pop_headplot')",
        show_help_button=False,
    )


def _run_gui(EEG: dict[str, Any], *, typeplot: int, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_headplot_dialog_spec(EEG, typeplot=typeplot), renderer=renderer)
    if result is None:
        return None
    options = parse_plot_options_text(result.get("options", ""))
    if result.get("loadcb"):
        options["load"] = str(result.get("load", "") or "").strip()
    else:
        setup_file = str(result.get("setup_file", "") or "").strip()
        meshfile = _popup_value(result.get("meshfile"), _MESH_CHOICES)
        transform = numeric_vector(result.get("transform", [])).tolist()
        options["setup"] = {"splinefile": setup_file, "meshfile": meshfile, "transform": transform}
        options["meshfile"] = meshfile
    return {
        "items": numeric_vector(result.get("items", [])).tolist(),
        "topotitle": str(result.get("topotitle", "") or ""),
        "rowcols": numeric_vector(result.get("rowcols", [])).tolist(),
        "options": options,
    }


def _parse_options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if len(args) % 2:
        raise ValueError("headplot options must be key/value pairs")
    options = {str(key).lower(): value for key, value in kwargs.items()}
    for index in range(0, len(args), 2):
        key = args[index]
        if not isinstance(key, str):
            raise ValueError("headplot option keys must be strings")
        options[key.lower()] = args[index + 1]
    return options


def _prepare_spline_file(EEG: dict[str, Any], typeplot: int, *, setup: Any, load: Any) -> str:
    fieldname = _spline_field(typeplot)
    if load is not None and not _is_empty(load):
        path = str(load)
        EEG[fieldname] = path
        return path
    if setup is not None and not _is_empty(setup):
        setup_options = _normalise_setup_options(setup)
        splinefile = str(setup_options.pop("splinefile"))
        meshfile = setup_options.get("meshfile", DEFAULT_MESH)
        chaninfo = dict(EEG.get("chaninfo") or {})
        if "icachansind" not in chaninfo and not _is_empty(EEG.get("icachansind")):
            chaninfo["icachansind"] = EEG["icachansind"]
        recompute = _is_on(setup_options.pop("recompute", False))
        if recompute or not Path(splinefile).expanduser().exists():
            headplot_setup(
                EEG.get("chanlocs", []),
                splinefile,
                chaninfo=chaninfo,
                ica="on" if int(typeplot) == 0 else "off",
                **setup_options,
            )
        EEG[fieldname] = splinefile
        EEG["headplotmeshfile"] = "" if Path(str(meshfile)).name == DEFAULT_MESH else str(meshfile)
        return splinefile
    reusable = _current_spline_file(EEG, typeplot)
    if reusable:
        return reusable
    other = _current_spline_file(EEG, 0 if int(typeplot) else 1)
    if other and _ica_uses_all_channels(EEG):
        EEG[fieldname] = other
        return other
    raise ValueError(
        "Pop_headplot: cannot find spline file. Use the GUI to create one, "
        "or call pop_headplot(..., setup={'splinefile': 'name.spl', 'transform': [...]})"
    )


def _normalise_setup_options(setup: Any) -> dict[str, Any]:
    if isinstance(setup, dict):
        options = {str(key).lower(): value for key, value in setup.items()}
    elif isinstance(setup, (list, tuple)) and setup:
        options = {"splinefile": setup[0]}
        tail = list(setup[1:])
        if len(tail) % 2:
            raise ValueError("setup option lists must contain a spline file followed by key/value pairs")
        for index in range(0, len(tail), 2):
            key = tail[index]
            if not isinstance(key, str):
                raise ValueError("setup option keys must be strings")
            options[key.lower()] = tail[index + 1]
    else:
        options = {"splinefile": setup}
    splinefile = str(options.get("splinefile", "") or "").strip()
    if not splinefile:
        raise ValueError("headplot setup requires an output spline file")
    transform = numeric_vector(options.get("transform", []))
    if transform.size == 0:
        raise ValueError(
            "headplot setup requires a Talairach transform. Enter it in the GUI, "
            "or pass setup={'splinefile': 'name.spl', 'transform': [...]}"
        )
    options["splinefile"] = splinefile
    options["meshfile"] = str(options.get("meshfile", DEFAULT_MESH) or DEFAULT_MESH)
    options["transform"] = transform.tolist()
    return options


def _headplot_maps(EEG: dict[str, Any], typeplot: int, items: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
    if typeplot:
        data, _times = data_time_slice(EEG, None)
        erp = np.nanmean(data, axis=2)
        all_times = eeg_times_ms(EEG)
        maps = []
        labels = []
        for latency in items:
            if latency < np.nanmin(all_times) or latency > np.nanmax(all_times):
                raise ValueError("requested latency is outside the epoch time range")
            frame = int(np.argmin(np.abs(all_times - latency)))
            maps.append(erp[:, frame])
            labels.append(f"{latency:g} ms")
        return maps, labels
    icawinv = component_maps(EEG)
    maps = []
    labels = []
    for component_value in items:
        if np.isnan(component_value):
            maps.append(np.full(icawinv.shape[0], np.nan))
            labels.append("")
            continue
        component = int(component_value)
        if component == 0 or abs(component) > icawinv.shape[1]:
            raise ValueError("component index is outside available ICA components")
        sign = -1 if component < 0 else 1
        maps.append(sign * icawinv[:, abs(component) - 1])
        labels.append(f"IC {abs(component)}" + (" (inv.)" if component < 0 else ""))
    return maps, labels


def _plot_pages(
    maps: list[np.ndarray],
    labels: list[str],
    splinefile: str,
    *,
    topotitle: str,
    rowcols: tuple[int, int],
    colorbar: bool,
    maplimits: Any,
    options: dict[str, Any],
) -> list[Any]:
    rows, columns = rowcols
    per_page = rows * columns
    figures = []
    for page_start in range(0, len(maps), per_page):
        page_maps = maps[page_start : page_start + per_page]
        page_labels = labels[page_start : page_start + per_page]
        fig = plt.figure(figsize=(5.0 * columns, 4.6 * rows))
        fig.suptitle(topotitle, fontweight="bold")
        for offset, values in enumerate(page_maps):
            ax = fig.add_subplot(rows, columns, offset + 1, projection="3d")
            if np.all(np.isnan(values)):
                ax.set_axis_off()
                continue
            headplot(
                values,
                splinefile,
                title=page_labels[offset] if len(page_maps) > 1 else topotitle,
                maplimits=maplimits,
                cbar=0 if colorbar and len(page_maps) == 1 else None,
                ax=ax,
                tight_layout=False,
                **options,
            )
        if colorbar and len(page_maps) > 1:
            mappable = plt.cm.ScalarMappable(norm=plt.Normalize(*np.asarray(maplimits, dtype=float).ravel()[:2]))
            fig.colorbar(mappable, ax=fig.axes, shrink=0.75)
            fig.subplots_adjust(right=0.9)
        else:
            fig.tight_layout()
        figures.append(fig)
    return figures


def _normalise_rowcols(rowcols: Any, count: int) -> tuple[int, int]:
    values = numeric_vector(rowcols, dtype=int)
    if values.size == 0 or int(values[0]) == 0:
        columns = int(math.ceil(math.sqrt(count)))
        rows = int(math.ceil(count / columns))
        return rows, columns
    if values.size != 2 or np.any(values < 1):
        raise ValueError("rowcols must be empty or contain [rows columns]")
    return int(values[0]), int(values[1])


def _erp_maplimits(EEG: dict[str, Any], items: np.ndarray) -> list[float]:
    maps, _labels = _headplot_maps(EEG, 1, items)
    data = np.asarray(maps, dtype=float)
    limit = float(np.nanmax(np.abs(data)) * MAPLIMIT_PADDING)
    if not np.isfinite(limit) or limit == 0:
        limit = 1.0
    return [-limit, limit]


def _shared_maplimits(maps: list[np.ndarray], setting: Any) -> list[float]:
    if isinstance(setting, str):
        lower = setting.lower()
        finite_values = np.concatenate([np.asarray(values, dtype=float).ravel() for values in maps])
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            finite_values = np.asarray([0.0])
        if lower in {"maxmin", "minmax"}:
            return [
                float(np.nanmin(finite_values) * MAPLIMIT_PADDING),
                float(np.nanmax(finite_values) * MAPLIMIT_PADDING),
            ]
        if lower == "absmax":
            limit = float(np.nanmax(np.abs(finite_values)) * MAPLIMIT_PADDING)
            if not np.isfinite(limit) or limit == 0:
                limit = 1.0
            return [-limit, limit]
    numeric = np.asarray(setting, dtype=float).ravel()
    if numeric.size == 2:
        return [float(numeric[0]), float(numeric[1])]
    raise ValueError("headplot maplimits must be 'absmax', 'maxmin', or [min max]")


def _history_command(
    typeplot: int,
    items: np.ndarray,
    topotitle: str,
    rowcols: list[int],
    colorbar: bool,
    setup: Any,
    load: Any,
    options: dict[str, Any],
) -> str:
    command_kwargs = dict(options)
    if setup is not None and not _is_empty(setup):
        command_kwargs["setup"] = _normalise_setup_options(setup)
    if load is not None and not _is_empty(load):
        command_kwargs["load"] = str(load)
    if not colorbar:
        command_kwargs["colorbar"] = "off"
    return history_command(
        "pop_headplot",
        int(typeplot),
        numeric_vector(items).tolist(),
        str(topotitle),
        rowcols,
        **command_kwargs,
    )


def _spline_field(typeplot: int) -> str:
    return "splinefile" if int(typeplot) else "icasplinefile"


def _current_spline_file(EEG: dict[str, Any], typeplot: int) -> str:
    value = EEG.get(_spline_field(typeplot), "")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        value = value.item() if value.size == 1 else ""
    return str(value or "").strip()


def _spline_file_is_ready(EEG: dict[str, Any], typeplot: int) -> bool:
    path = _current_spline_file(EEG, typeplot)
    return bool(path and Path(path).expanduser().exists())


def _default_spline_path(EEG: dict[str, Any]) -> str:
    base = Path(str(EEG.get("filepath") or "")).expanduser() if EEG.get("filepath") else Path.cwd()
    filename = str(EEG.get("filename") or "").strip()
    stem = Path(filename).stem if filename else "tmp"
    return str(base / f"{stem}.spl")


def _default_title(EEG: dict[str, Any], typeplot: int) -> str:
    prefix = "ERP scalp maps of dataset:" if int(typeplot) else "Components of dataset:"
    setname = str(EEG.get("setname") or "").strip()
    return f"{prefix} {setname}".rstrip()


def _default_transform_text(EEG: dict[str, Any], meshfile: str | Path) -> str:
    transform = default_headplot_mesh_transform(meshfile, dict(EEG.get("chaninfo") or {}))
    return " ".join(f"{value:g}" for value in transform)


def _component_items_value(EEG: dict[str, Any]) -> str:
    count = np.asarray(EEG.get("icaweights", [])).shape[0]
    return f"1:{count}" if count else _DEFAULT_COMPONENT_ITEMS


def _popup_value(value: Any, choices: tuple[str, ...]) -> str:
    try:
        index = int(value) - 1
    except (TypeError, ValueError):
        return str(value or choices[0])
    return choices[index] if 0 <= index < len(choices) else choices[0]


def _setup_tags() -> tuple[str, ...]:
    return (
        "setup_file",
        "setup_browse",
        "mesh_label",
        "meshfile",
        "mesh_browse",
        "meshchan_label",
        "meshchanfile",
        "meshchan_browse",
        "transform_label",
        "transform",
        "manual_coreg",
    )


def _ica_uses_all_channels(EEG: dict[str, Any]) -> bool:
    icachansind = np.asarray(EEG.get("icachansind", []), dtype=int).ravel()
    nbchan = int(EEG.get("nbchan", 0) or 0)
    return icachansind.size == nbchan


def _validate_chanlocs(EEG: dict[str, Any]) -> None:
    if not chanlocs_as_list(EEG.get("chanlocs", [])):
        raise ValueError("Pop_headplot: this dataset does not contain channel locations. Use Edit > Channel locations.")


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, np.ndarray):
        return value.size == 0
    return isinstance(value, (list, tuple, dict)) and len(value) == 0


def _is_on(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


__all__ = ["pop_headplot", "pop_headplot_dialog_spec"]
