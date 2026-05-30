"""GUI/API wrapper for STUDY component preclustering."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._plot_utils import numeric_vector
from eegprep.functions.popfunc._pop_utils import is_on
from eegprep.functions.studyfunc._cluster_utils import checked_study_and_datasets, cluster_command
from eegprep.functions.studyfunc.std_preclust import DEFAULT_PRECLUST, normalize_preclust_specs, std_preclust


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
    return DialogSpec(
        title="Build pre-clustering matrix -- pop_preclust()",
        controls=(
            ControlSpec(
                "text", f"Build pre-clustering matrix for STUDY set: {STUDY.get('name') or ''}", font_weight="bold"
            ),
            ControlSpec("text", "Only measures that have been precomputed may be used for clustering"),
            ControlSpec(
                "text", "Mixing time-based and location-based measures might result in statistical double-dipping"
            ),
            ControlSpec("text", "Time-based info             PCA              Weight", font_weight="bold"),
            ControlSpec("checkbox", "", tag="spec_on", value=0),
            ControlSpec("text", "spectra"),
            ControlSpec("edit", tag="spec_npca", value="3", enabled=False),
            ControlSpec("edit", tag="spec_weight", value="1", enabled=False),
            ControlSpec("text", "Freq.range [Hz]", enabled=False),
            ControlSpec("edit", tag="spec_freqrange", value="3 25", enabled=False),
            ControlSpec("checkbox", "", tag="erp_on", value=0),
            ControlSpec("text", "ERPs"),
            ControlSpec("edit", tag="erp_npca", value="3", enabled=False),
            ControlSpec("edit", tag="erp_weight", value="1", enabled=False),
            ControlSpec("text", "Time range [ms]", enabled=False),
            ControlSpec("edit", tag="erp_timewindow", value="", enabled=False),
            ControlSpec("checkbox", "", tag="ersp_on", value=0),
            ControlSpec("text", "ERSPs"),
            ControlSpec("edit", tag="ersp_npca", value="3", enabled=False),
            ControlSpec("edit", tag="ersp_weight", value="1", enabled=False),
            ControlSpec("text", "Time range [ms]", enabled=False),
            ControlSpec("edit", tag="ersp_timewindow", value="", enabled=False),
            ControlSpec("checkbox", "", tag="itc_on", value=0),
            ControlSpec("text", "ITCs"),
            ControlSpec("edit", tag="itc_npca", value="3", enabled=False),
            ControlSpec("edit", tag="itc_weight", value="1", enabled=False),
            ControlSpec("text", "Time range [ms]", enabled=False),
            ControlSpec("edit", tag="itc_timewindow", value="", enabled=False),
            ControlSpec("text", "Location-based info      PCA              Weight", font_weight="bold"),
            ControlSpec("checkbox", "", tag="dipoles_on", value=0),
            ControlSpec("text", "dipole locations"),
            ControlSpec("text", "3", enabled=False),
            ControlSpec("edit", tag="dipoles_weight", value="1", enabled=False),
            ControlSpec("checkbox", "", tag="moments_on", value=0),
            ControlSpec("text", "dipole orient."),
            ControlSpec("text", "3", enabled=False),
            ControlSpec("edit", tag="moments_weight", value="1", enabled=False),
            ControlSpec("text", "Amplitude & polarity is ignored"),
            ControlSpec("checkbox", "", tag="scalp_on", value=0),
            ControlSpec("text", "scalp maps"),
            ControlSpec("edit", tag="scalp_npca", value="3", enabled=False),
            ControlSpec("edit", tag="scalp_weight", value="1", enabled=False),
            ControlSpec("checkbox", "Absolute values", tag="scalp_abso", value=1, enabled=False),
        ),
        geometry=(
            (1,),
            (1,),
            (1,),
            (1,),
            (0.2, 0.9, 0.45, 0.45, 0.8, 0.8),
            (0.2, 0.9, 0.45, 0.45, 0.8, 0.8),
            (0.2, 0.9, 0.45, 0.45, 0.8, 0.8),
            (0.2, 0.9, 0.45, 0.45, 0.8, 0.8),
            (1,),
            (0.2, 0.9, 0.45, 0.45),
            (0.2, 0.9, 0.45, 0.45, 1.4),
            (0.2, 0.9, 0.45, 0.45, 1.4),
        ),
        function_name="pop_preclust",
        eeglab_source="functions/studyfunc/pop_preclust.m",
        help_text="pop_preclust",
        size=(760, 520),
    )


def _specs_from_measures(measures: Any) -> list[dict[str, Any]]:
    if measures is None:
        return list(DEFAULT_PRECLUST)
    if isinstance(measures, str):
        measures = [measures]
    return [{"measure": str(measure).lower(), "npca": 5, "norm": 1, "weight": 1} for measure in measures]


def _specs_from_gui(result: dict[str, Any]) -> list[dict[str, Any]]:
    specs = []
    for measure in ("spec", "erp", "ersp", "itc", "dipoles", "moments", "scalp"):
        if not is_on(result.get(f"{measure}_on")):
            continue
        spec: dict[str, Any] = {"measure": measure, "norm": 1}
        if measure != "dipoles":
            spec["npca"] = _int_field(result, f"{measure}_npca", 3)
        spec["weight"] = _float_field(result, f"{measure}_weight", 1.0)
        if measure == "scalp":
            spec["abso"] = int(is_on(result.get("scalp_abso")))
        if measure in {"spec", "ersp", "itc"}:
            freqrange = result.get(f"{measure}_freqrange")
            if freqrange:
                spec["freqrange"] = numeric_vector(freqrange).tolist()
        if measure in {"erp", "ersp", "itc"}:
            timewindow = result.get(f"{measure}_timewindow")
            if timewindow:
                spec["timewindow"] = numeric_vector(timewindow).tolist()
        specs.append(spec)
    if not specs:
        raise ValueError("Select at least one preclustering measure")
    return specs


def _int_field(result: dict[str, Any], tag: str, default: int) -> int:
    values = numeric_vector(result.get(tag, default), dtype=int)
    return int(values[0]) if values.size else default


def _float_field(result: dict[str, Any], tag: str, default: float) -> float:
    values = numeric_vector(result.get(tag, default), dtype=float)
    return float(values[0]) if values.size else default


__all__ = ["pop_preclust", "pop_preclust_dialog_spec"]
