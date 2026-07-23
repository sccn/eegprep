"""GUI wrapper for STUDY measure precompute."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import is_on, parse_key_value_args
from eegprep.functions.popfunc.plot_utils import parse_plot_options_text, python_literal
from eegprep.functions.studyfunc._study_utils import as_alleeg_list, build_python_call, ensure_study
from eegprep.functions.studyfunc.std_checkset import std_checkset
from eegprep.functions.studyfunc.std_precomp import std_precomp


def pop_precomp(
    STUDY: dict[str, Any],
    ALLEEG: list[dict[str, Any]] | None,
    chanorcomp: Any = "channels",
    *args: Any,
    gui: bool | str | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Precompute STUDY channel or component measures."""
    datasets = as_alleeg_list(ALLEEG)
    study, datasets = std_checkset(ensure_study(STUDY), datasets)
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    gui = options.pop("gui", gui)
    use_gui = is_on(gui) if gui is not None else False
    if use_gui:
        result = inputgui(pop_precomp_dialog_spec(study, datasets, chanorcomp), renderer=renderer)
        if result is None:
            return (study, datasets, "") if return_com else (study, datasets)
        options.update(_options_from_gui(result))
        chanorcomp = _kind_from_gui(result)
    study, datasets, _std_command = std_precomp(study, datasets, chanorcomp, return_com=True, **options)
    command = _history_command(chanorcomp, options)
    return (study, datasets, command) if return_com else (study, datasets)


def pop_precomp_dialog_spec(
    STUDY: dict[str, Any], ALLEEG: list[dict[str, Any]] | None, chanorcomp: Any = "channels"
) -> DialogSpec:
    """Return the EEGLAB-like STUDY measure precompute dialog spec."""
    study = ensure_study(STUDY)
    mode_value = 2 if isinstance(chanorcomp, str) and chanorcomp.lower() == "components" else 1
    title_name = str(study.get("name") or "")
    is_components = mode_value == 2
    extra_measure_controls = (
        (
            ControlSpec("checkbox", "", tag="component_rv_on", value=True),
            ControlSpec(
                "text",
                "Compute ERP/spectrum/ERSP only for components selected by RV (set) or for all components (unset)",
            ),
        )
        if is_components
        else (
            ControlSpec("checkbox", "", tag="interpolate_on", value=True),
            ControlSpec(
                "text",
                "Spherical interpolation of missing channels (performed after optional ICA removal below)",
            ),
            ControlSpec("checkbox", "", tag="rmica1_on", value=False),
            ControlSpec("text", "Remove ICA artifactual components pre-tagged in each dataset"),
            ControlSpec("checkbox", "", tag="rmica2_on", value=False),
            ControlSpec("text", "Remove artifactual ICA cluster or clusters (hold shift key)"),
            ControlSpec("listbox", "ParentCluster", tag="rmica2_val", value=1),
        )
    )
    controls = (
        ControlSpec("text", f"Pre-compute measures for STUDY '{title_name}'", font_weight="bold"),
        ControlSpec("text", "Measure target"),
        ControlSpec("popupmenu", "Channels|Components", tag="mode", value=mode_value),
        *extra_measure_controls,
        ControlSpec("text", "List of measures to precompute", font_weight="bold"),
        ControlSpec("text", "Measure parameters", font_weight="bold"),
        ControlSpec("checkbox", "", tag="erp_on", value=True),
        ControlSpec("text", "ERPs"),
        ControlSpec("edit", tag="erp_params", value=""),
        ControlSpec("pushbutton", "...", tag="erp_button", enabled=False),
        ControlSpec("spacer"),
        ControlSpec("checkbox", "", tag="spectra_on", value=True),
        ControlSpec("text", "Power spectrum"),
        ControlSpec("edit", tag="spec_params", value="'specmode', 'fft', 'logtrials', 'off'"),
        ControlSpec("pushbutton", "...", tag="spec_button", enabled=False),
        ControlSpec("pushbutton", "Test", tag="spec_test", enabled=False),
        ControlSpec("checkbox", "", tag="erpimage_on", value=False, enabled=False),
        ControlSpec("text", "ERP-image", enabled=False),
        ControlSpec("edit", tag="erpimage_params", value="'nlines', 10, 'smoothing', 10", enabled=False),
        ControlSpec("pushbutton", "...", tag="erpimage_button", enabled=False),
        ControlSpec("spacer"),
        ControlSpec("checkbox", "", tag="ersp_on", value=False),
        ControlSpec("text", "ERSPs"),
        ControlSpec("edit", tag="ersp_params", value="'cycles', [3, 0.8], 'nfreqs', 100, 'timesout', 200"),
        ControlSpec("pushbutton", "...", tag="ersp_button", enabled=False),
        ControlSpec("pushbutton", "Test", tag="ersp_test", enabled=False),
        ControlSpec("checkbox", "", tag="itc_on", value=False),
        ControlSpec("text", "ITCs"),
        ControlSpec("spacer"),
        ControlSpec("spacer"),
        ControlSpec("spacer"),
        ControlSpec("checkbox", "", tag="scalp_on", value=False, enabled=is_components),
        ControlSpec("text", "Scalp maps", enabled=is_components),
        ControlSpec("checkbox", "All components", tag="allcomps_on", value=True),
        ControlSpec("checkbox", "Overwrite cached measures", tag="recomp_on", value=True),
    )
    intro_geometry = ((0.35, 6),) if is_components else ((0.35, 6), (0.35, 6), (0.35, 4, 2))
    intro_geomvert = (1,) if is_components else (1, 1, 2)
    return DialogSpec(
        title="Select and compute component measures for later clustering -- pop_precomp()",
        controls=controls,
        geometry=(
            (1,),
            (1, 1),
            *intro_geometry,
            (1, 1),
            (0.35, 1.25, 4.1, 0.42, 0.58),
            (0.35, 1.25, 4.1, 0.42, 0.58),
            (0.35, 1.25, 4.1, 0.42, 0.58),
            (0.35, 1.25, 4.1, 0.42, 0.58),
            (0.35, 1.25, 4.1, 0.42, 0.58),
            (0.35, 6),
            (1,),
            (1,),
        ),
        function_name="pop_precomp",
        eeglab_source="functions/studyfunc/pop_precomp.m",
        help_text="pophelp('pop_precomp')",
        size=(780, 520 if is_components else 590),
        content_margins=(28, 22, 28, 16),
        row_spacing=5,
        geomvert=(1, 0.5, *intro_geomvert, 0.5, 1, 1, 1, 1, 1, 1, 1, 1),
        button_size=(62, 20),
        extra_stylesheet="""
            QDialog#pop_precomp QLabel,
            QDialog#pop_precomp QCheckBox,
            QDialog#pop_precomp QPushButton,
            QDialog#pop_precomp QLineEdit,
            QDialog#pop_precomp QComboBox,
            QDialog#pop_precomp QListWidget {
                font-size: 13px;
            }
            QDialog#pop_precomp QLineEdit,
            QDialog#pop_precomp QPushButton,
            QDialog#pop_precomp QComboBox {
                min-height: 20px;
                max-height: 20px;
            }
            QDialog#pop_precomp QPushButton#erp_button,
            QDialog#pop_precomp QPushButton#spec_button,
            QDialog#pop_precomp QPushButton#erpimage_button,
            QDialog#pop_precomp QPushButton#ersp_button {
                min-width: 36px;
                max-width: 36px;
                padding: 0;
            }
            QDialog#pop_precomp QPushButton#spec_test,
            QDialog#pop_precomp QPushButton#ersp_test {
                min-width: 52px;
                max-width: 52px;
                padding: 0 3px;
            }
            QDialog#pop_precomp QListWidget#rmica2_val {
                min-height: 42px;
                max-height: 58px;
            }
        """,
        known_differences=(
            "EEGPrep stores STUDY measures in the STUDY dictionary instead of EEGLAB .dat/.ica sidecar files.",
            "The selected design is recorded in measure metadata; Phase 5b cached arrays are dataset-level averages and "
            "are not split into EEGLAB design cells.",
        ),
    )


def _options_from_gui(result: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {
        "erp": "on" if is_on(result.get("erp_on")) else "off",
        "spec": "on" if is_on(result.get("spectra_on")) else "off",
        "ersp": "on" if is_on(result.get("ersp_on")) else "off",
        "itc": "on" if is_on(result.get("itc_on")) else "off",
        "scalp": "on" if is_on(result.get("scalp_on")) else "off",
        "allcomps": "on" if is_on(result.get("allcomps_on")) else "off",
        "recompute": "on" if is_on(result.get("recomp_on")) else "off",
    }
    spec_params = _params_from_text(result.get("spec_params"))
    ersp_params = _params_from_text(result.get("ersp_params"))
    erp_params = _params_from_text(result.get("erp_params"))
    if spec_params:
        options["specparams"] = spec_params
    if ersp_params:
        options["erspparams"] = ersp_params
    if erp_params:
        options["erpparams"] = erp_params
    return options


def _kind_from_gui(result: dict[str, Any]) -> str:
    return "components" if int(result.get("mode") or 1) == 2 else "channels"


def _params_from_text(text: Any) -> dict[str, Any]:
    return parse_plot_options_text(text)


def _history_command(chanorcomp: Any, options: dict[str, Any]) -> str:
    kwargs = {key: value for key, value in options.items() if value not in (None, [], "")}
    return build_python_call(
        ("STUDY", "ALLEEG"),
        "pop_precomp",
        "STUDY",
        "ALLEEG",
        python_literal(chanorcomp),
        **kwargs,
    )


__all__ = ["pop_precomp", "pop_precomp_dialog_spec"]
