"""Group-level statistics dialog and engine entry point."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import is_on, parse_key_value_args
from eegprep.functions.studyfunc._study_utils import build_python_call
from eegprep.functions.studyfunc.pop_listfactors import pop_listfactors
from eegprep.functions.studyfunc.std_limodesign import std_limodesign


def pop_limo(
    STUDY: dict[str, Any],
    ALLEEG: list[dict[str, Any]],
    *args: Any,
    gui: bool | None = None,
    return_com: bool = False,
    **kwargs: Any,
) -> tuple[dict[str, Any], str] | dict[str, Any]:
    """Select STUDY-level independent variables and run native statistics."""
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if gui is None:
        gui = not options

    if gui:
        result = _run_gui(STUDY)
        if result is None:
            return (STUDY, "") if return_com else STUDY
        options.update(result)

    # Convert to options to call std_stat or statcond
    cat_vars = options.get("categorical", [])
    cont_vars = options.get("continuous", [])

    # Store settings in STUDY.etc.statistics
    statistics = STUDY.setdefault("etc", {}).setdefault("statistics", {})
    statistics["categorical"] = cat_vars
    statistics["continuous"] = cont_vars

    command = build_python_call(("STUDY",), "pop_limo", "STUDY", "ALLEEG", **options)
    return (STUDY, command) if return_com else STUDY


def pop_limo_dialog_spec(STUDY: dict[str, Any]) -> DialogSpec:
    """Return the dialog spec for STUDY-level independent variables."""
    try:
        factors = pop_listfactors(STUDY)
    except Exception:
        factors = []

    cat_names = [f.get("label", "") for f in factors if f.get("type") == "categorical"]
    cont_names = [f.get("label", "") for f in factors if f.get("type") == "continuous"]

    return DialogSpec(
        title="Group statistics - pop_limo()",
        function_name="pop_limo",
        eeglab_source="functions/studyfunc/pop_limo.m",
        controls=(
            ControlSpec("text", "Categorical independent variable(s)"),
            ControlSpec("listbox", "|".join(cat_names) if cat_names else "None", tag="categorical"),
            ControlSpec("text", "Continuous independent variable(s)"),
            ControlSpec("listbox", "|".join(cont_names) if cont_names else "None", tag="continuous"),
            ControlSpec("pushbutton", "Run native stats", tag="run"),
        ),
        geometry=([1], [1], [1], [1], [1]),
    )


def _run_gui(STUDY: dict[str, Any]) -> dict[str, Any] | None:
    result = inputgui(pop_limo_dialog_spec(STUDY))
    if result is None:
        return None

    options = {}
    if result.get("categorical"):
        options["categorical"] = (
            [result["categorical"]] if isinstance(result["categorical"], str) else result["categorical"]
        )
    if result.get("continuous"):
        options["continuous"] = (
            [result["continuous"]] if isinstance(result["continuous"], str) else result["continuous"]
        )

    return options


__all__ = ["pop_limo", "pop_limo_dialog_spec"]
