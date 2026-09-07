"""GUI/API wrapper for STUDY component preclustering."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc.plot_utils import numeric_vector
from eegprep.functions.popfunc._pop_utils import is_on
from eegprep.functions.studyfunc._cluster_utils import checked_study_and_datasets, cluster_command
from eegprep.functions.studyfunc.std_preclust import DEFAULT_PRECLUST, normalize_preclust_specs, std_preclust

_MEASURE_TARGETS = {
    "spectra": ("spectra_PCA", "spectra_weight", "spectra_freq_txt", "spectra_freq_edit"),
    "erp": ("erp_PCA", "erp_weight", "erp_time_txt", "erp_time_edit", "erp_filter_txt", "erp_filter_edit"),
    "ersp": ("ersp_PCA", "ersp_weight", "ersp_time_txt", "ersp_time_edit", "ersp_freq_txt", "ersp_freq_edit"),
    "itc": ("itc_PCA", "itc_weight", "itc_time_txt", "itc_time_edit", "itc_freq_txt", "itc_freq_edit"),
}
_SCALP_TARGETS = ("scalp_PCA", "scalp_weight", "scalp_choice", "scalp_absolute")


def pop_preclust(
    STUDY: dict[str, Any] | None,
    ALLEEG: Any,
    cluster_ind: int | None = 1,
    *args: Any,
    preproc: Any = None,
    measures: Any = None,
    gui: bool = False,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Prepare a STUDY component clustering matrix."""
    study, datasets = checked_study_and_datasets(STUDY, ALLEEG)
    if args:
        preproc = args
    if kwargs:
        if "gui" in kwargs:
            gui = is_on(kwargs.pop("gui"))
        if kwargs:
            raise ValueError(f"Unknown pop_preclust option(s): {', '.join(sorted(kwargs))}")
    if gui:
        result = inputgui(pop_preclust_dialog_spec(study), renderer=renderer)
        if result is None:
            return (study, datasets, "") if return_com else (study, datasets)
        cluster_ind = _int_field(result, "cluster_ind", int(cluster_ind or 1))
        preproc = _specs_from_gui(result)
    elif preproc is None:
        preproc = _specs_from_measures(measures)
    specs = normalize_preclust_specs(tuple(preproc or DEFAULT_PRECLUST))
    study, datasets, _std_command = std_preclust(study, datasets, cluster_ind, specs, return_com=True)
    command = cluster_command(
        "pop_preclust",
        ("STUDY", "ALLEEG"),
        "STUDY",
        "ALLEEG",
        cluster_ind=cluster_ind,
        preproc=specs,
    )
    return (study, datasets, command) if return_com else (study, datasets)


def pop_preclust_dialog_spec(STUDY: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like preclustering measure dialog spec."""
    geomline = (0.5, 2, 1, 0.5, 1, 2, 1, 2, 1)
    controls = [
        ControlSpec(
            "text", f"Build pre-clustering matrix for STUDY set: {STUDY.get('name') or ''}", font_weight="bold"
        ),
        ControlSpec("text", "Only measures that have been precomputed may be used for clustering"),
        ControlSpec("text", "Mixing time-based and location-based measures might result in statistical double-dipping"),
        ControlSpec(
            "pushbutton",
            "Help",
            tag="double_dip_help",
            callback=CallbackSpec(
                "show_message",
                {
                    "button": "double_dip_help",
                    "title": "Double dipping",
                    "message": (
                        "When clustering on both time- and location-based measures, take care when performing "
                        "statistical tests on cluster measures to avoid double-dipping interpretation problems."
                    ),
                },
            ),
        ),
        ControlSpec("text", "Time-based info             PCA              Weight", font_weight="bold"),
        *_measure_row(
            "spectra",
            "spectra",
            "spectra_PCA",
            "spectra_weight",
            "Freq.range [Hz]",
            "spectra_freq_edit",
            "3 25",
            "",
            "",
        ),
        *_measure_row(
            "erp",
            "ERPs",
            "erp_PCA",
            "erp_weight",
            "Time range [ms]",
            "erp_time_edit",
            "",
            "Lowpass [Hz]",
            "erp_filter_edit",
            "20",
        ),
        *_measure_row(
            "ersp",
            "ERSPs",
            "ersp_PCA",
            "ersp_weight",
            "Time range [ms]",
            "ersp_time_edit",
            "",
            "Freq. range [Hz]",
            "ersp_freq_edit",
            "",
        ),
        *_measure_row(
            "itc",
            "ITCs",
            "itc_PCA",
            "itc_weight",
            "Time range [ms]",
            "itc_time_edit",
            "",
            "Freq. range [Hz]",
            "itc_freq_edit",
            "",
        ),
        ControlSpec("text", "Location-based info      PCA              Weight", font_weight="bold"),
        *_location_measure_row("dipole", "dipole locations", "3", "locations_weight"),
        *_location_measure_row(
            "moment", "dipole orient.", "3", "moment_weight", note="Amplitude & polarity is ignored"
        ),
        ControlSpec("checkbox", "", tag="scalp_on", value=0, callback=_toggle("scalp_on", _SCALP_TARGETS)),
        ControlSpec("text", "scalp maps"),
        ControlSpec("edit", tag="scalp_PCA", value="3", enabled=False),
        ControlSpec("spacer"),
        ControlSpec("edit", tag="scalp_weight", value="1", enabled=False),
        ControlSpec(
            "popupmenu",
            "Use channel values|Use Laplacian values|Use Gradient values",
            tag="scalp_choice",
            value=1,
            enabled=False,
        ),
        ControlSpec("spacer"),
        ControlSpec("checkbox", "Absolute values", tag="scalp_absolute", value=1, enabled=False),
        ControlSpec("spacer"),
        ControlSpec("spacer"),
    ]
    return DialogSpec(
        title="Select and compute component measures for later clustering -- pop_preclust()",
        controls=tuple(controls),
        geometry=(
            (1,),
            (1,),
            (1, 0.2),
            (3,),
            geomline,
            geomline,
            geomline,
            geomline,
            (1,),
            geomline,
            (0.5, 2, 1, 0.5, 1, 5.4, 0.2, 0.2, 0.2),
            (0.5, 2, 1, 0.5, 1, 2.9, 0.1, 2.9, 0.1),
            (1,),
        ),
        function_name="pop_preclust",
        eeglab_source="functions/studyfunc/pop_preclust.m",
        help_text="pop_preclust",
        size=(880, 565),
        content_margins=(24, 20, 24, 14),
        row_spacing=5,
        geomvert=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5),
        button_size=(58, 20),
        extra_stylesheet="""
            QDialog#pop_preclust QPushButton#double_dip_help {
                min-width: 96px;
                max-width: 96px;
            }
        """,
    )


def _toggle(source: str, targets: tuple[str, ...]) -> CallbackSpec:
    return CallbackSpec("toggle_enabled", {"source": source, "targets": targets})


def _measure_row(
    prefix: str,
    label: str,
    pca_tag: str,
    weight_tag: str,
    first_label: str,
    first_edit_tag: str,
    first_value: str,
    second_label: str,
    second_edit_tag: str,
    second_value: str = "",
) -> tuple[ControlSpec, ...]:
    on_tag = f"{prefix}_on"
    return (
        ControlSpec("checkbox", "", tag=on_tag, value=0, callback=_toggle(on_tag, _MEASURE_TARGETS[prefix])),
        ControlSpec("text", label),
        ControlSpec("edit", tag=pca_tag, value="3", enabled=False),
        ControlSpec("spacer"),
        ControlSpec("edit", tag=weight_tag, value="1", enabled=False),
        ControlSpec("text", first_label, tag=f"{prefix}_{_label_kind(first_label)}_txt", enabled=False),
        ControlSpec("edit", tag=first_edit_tag, value=first_value, enabled=False),
        ControlSpec("text", second_label, tag=f"{prefix}_{_label_kind(second_label)}_txt", enabled=False)
        if second_label
        else ControlSpec("spacer"),
        ControlSpec("edit", tag=second_edit_tag, value=second_value, enabled=False)
        if second_edit_tag
        else ControlSpec("spacer"),
    )


def _label_kind(label: str) -> str:
    if "freq" in label.lower():
        return "freq"
    if "lowpass" in label.lower():
        return "filter"
    return "time"


def _location_measure_row(
    prefix: str, label: str, pca_text: str, weight_tag: str, note: str = ""
) -> tuple[ControlSpec, ...]:
    on_tag = f"{prefix}_on"
    targets = (weight_tag,)
    return (
        ControlSpec("checkbox", "", tag=on_tag, value=0, callback=_toggle(on_tag, targets)),
        ControlSpec("text", label),
        ControlSpec("text", pca_text, enabled=False),
        ControlSpec("spacer"),
        ControlSpec("edit", tag=weight_tag, value="1", enabled=False),
        ControlSpec("text", note) if note else ControlSpec("spacer"),
        ControlSpec("spacer"),
        ControlSpec("spacer"),
        ControlSpec("spacer"),
    )


def _specs_from_measures(measures: Any) -> list[dict[str, Any]]:
    if measures is None:
        return list(DEFAULT_PRECLUST)
    if isinstance(measures, str):
        measures = [measures]
    return [{"measure": str(measure).lower(), "npca": 5, "norm": 1, "weight": 1} for measure in measures]


def _specs_from_gui(result: dict[str, Any]) -> list[dict[str, Any]]:
    specs = []
    gui_measures = (
        ("spectra", "spec", "spectra_PCA", "spectra_weight"),
        ("erp", "erp", "erp_PCA", "erp_weight"),
        ("ersp", "ersp", "ersp_PCA", "ersp_weight"),
        ("itc", "itc", "itc_PCA", "itc_weight"),
        ("dipole", "dipoles", "", "locations_weight"),
        ("moment", "moments", "", "moment_weight"),
        ("scalp", "scalp", "scalp_PCA", "scalp_weight"),
    )
    for prefix, measure, pca_tag, weight_tag in gui_measures:
        if not is_on(_gui_value(result, f"{prefix}_on", f"{measure}_on")):
            continue
        spec: dict[str, Any] = {"measure": measure, "norm": 1}
        if measure != "dipoles":
            spec["npca"] = _int_field(result, pca_tag, 3)
        spec["weight"] = _float_field(result, weight_tag, 1.0)
        if measure == "scalp":
            spec["abso"] = int(is_on(_gui_value(result, "scalp_absolute", "scalp_abso")))
        if measure == "erp":
            lowpass = _gui_value(result, "erp_filter_edit")
            if lowpass:
                spec["filter"] = _float_field(result, "erp_filter_edit", 20.0)
        if measure == "spec":
            freqrange = _gui_value(result, "spectra_freq_edit", "spec_freqrange")
            if freqrange:
                spec["freqrange"] = numeric_vector(freqrange).tolist()
        if measure in {"ersp", "itc"}:
            freqrange = _gui_value(result, f"{measure}_freq_edit", f"{measure}_freqrange")
            if freqrange:
                spec["freqrange"] = numeric_vector(freqrange).tolist()
        if measure in {"erp", "ersp", "itc"}:
            timewindow = _gui_value(result, f"{measure}_time_edit", f"{measure}_timewindow")
            if timewindow:
                spec["timewindow"] = numeric_vector(timewindow).tolist()
        specs.append(spec)
    if not specs:
        raise ValueError("Select at least one preclustering measure")
    return specs


def _gui_value(result: dict[str, Any], *tags: str) -> Any:
    for tag in tags:
        if tag and tag in result:
            return result[tag]
    return None


def _int_field(result: dict[str, Any], tag: str, default: int) -> int:
    values = numeric_vector(result.get(tag, default), dtype=int)
    return int(values[0]) if values.size else default


def _float_field(result: dict[str, Any], tag: str, default: float) -> float:
    values = numeric_vector(result.get(tag, default), dtype=float)
    return float(values[0]) if values.size else default


__all__ = ["pop_preclust", "pop_preclust_dialog_spec"]
