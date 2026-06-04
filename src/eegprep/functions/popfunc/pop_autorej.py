"""Automatic EEGLAB-style artifact rejection for epoched data."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._eegplot_rejection import open_epoched_rejection_browser
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args
from eegprep.functions.popfunc._rejection import (
    copy_eeg,
    one_based_indices,
    parse_numeric_sequence,
    rejection_data,
    update_reject_fields,
)
from eegprep.functions.popfunc.pop_eegthresh import pop_eegthresh
from eegprep.functions.popfunc.pop_jointprob import pop_jointprob
from eegprep.functions.popfunc.pop_rejepoch import pop_rejepoch
from eegprep.functions.popfunc.pop_rejkurt import pop_rejkurt


def pop_autorej(
    EEG: dict[str, Any],
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    command_callback: Any | None = None,
    show: bool = True,
    **kwargs: Any,
):
    """Run EEGLAB's automatic epoch rejection protocol."""
    if EEG is None:
        return (None, "") if return_com else (None, [])
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if gui is None:
        gui = str(options.get("nogui", "off")).lower() != "on" and not options
    if gui:
        gui_options = _run_gui(EEG, renderer=renderer)
        if gui_options is None:
            return (EEG, "") if return_com else (EEG, [])
        options.update(gui_options)
    command = _history_command(options)
    out, rejected = _apply_one(
        EEG,
        options,
        display=str(options.get("eegplot", "off")).lower() == "on",
        command=command,
        command_callback=command_callback,
        show=show,
    )
    return (out, command) if return_com else (out, rejected)


def pop_autorej_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_autorej``."""
    return DialogSpec(
        title="Automatic artifact rejection -- pop_autorej()",
        function_name="pop_autorej",
        eeglab_source="functions/popfunc/pop_autorej.m",
        help_text="pophelp('pop_autorej')",
        size=(610, 404),
        geometry=((1,), (2, 1), (1,), (1,), (2, 1), (2, 1), (2, 1), (2, 1), (1,), (2, 1)),
        controls=(
            ControlSpec("text", "Detection of extremely large fluctuations (channels only)", font_weight="bold"),
            ControlSpec("text", "Threshold limit (microV)"),
            ControlSpec("edit", tag="threshold", value="1000"),
            ControlSpec("spacer"),
            ControlSpec("text", "Detection of improbable activity (channels or ICA)", font_weight="bold"),
            ControlSpec("text", "Do not use these channel indices (default=all)"),
            ControlSpec("edit", tag="exclude", value=""),
            ControlSpec("text", "Use these ICA components instead of data channels"),
            ControlSpec("edit", tag="icacomps", value=""),
            ControlSpec("text", "Probability threshold (std. dev.)"),
            ControlSpec("edit", tag="startprob", value="5"),
            ControlSpec("text", "Maximum % of total trials to reject per iteration"),
            ControlSpec("edit", tag="maxrej", value="5"),
            ControlSpec("spacer"),
            ControlSpec("text", "Check box for visual inspection of results"),
            ControlSpec("checkbox", tag="eegplot", value=True, enabled=True),
        ),
    )


def _run_gui(EEG: dict[str, Any], renderer: Any | None) -> dict[str, Any] | None:
    result = inputgui(pop_autorej_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    options: dict[str, Any] = {"nogui": "on"}
    if result.get("threshold", "1000") != "1000":
        options["threshold"] = result.get("threshold")
    if str(result.get("exclude", "")).strip():
        exclude = set(one_based_indices(result.get("exclude"), limit=int(EEG.get("nbchan", 0)), default_all=False))
        options["electrodes"] = [index for index in range(1, int(EEG.get("nbchan", 0)) + 1) if index not in exclude]
    if str(result.get("icacomps", "")).strip():
        options["icacomps"] = result.get("icacomps")
    if result.get("startprob", "5") != "5":
        options["startprob"] = result.get("startprob")
    if result.get("maxrej", "5") != "5":
        options["maxrej"] = result.get("maxrej")
    if result.get("eegplot"):
        options["eegplot"] = "on"
    return options


def _apply_one(
    EEG: dict[str, Any],
    options: dict[str, Any],
    *,
    display: bool = False,
    command: str = "",
    command_callback: Any | None = None,
    show: bool = True,
) -> tuple[dict[str, Any], list[int]]:
    if int(EEG.get("trials", 1) or 1) <= 1:
        raise ValueError("pop_autorej requires epoched data")
    threshold = float(parse_numeric_sequence(options.get("threshold", 1000), dtype=float)[0])
    startprob = float(parse_numeric_sequence(options.get("startprob", 5), dtype=float)[0])
    maxrej = float(parse_numeric_sequence(options.get("maxrej", 5), dtype=float)[0])
    electrodes = one_based_indices(options.get("electrodes"), limit=int(EEG.get("nbchan", 0)), default_all=True)
    icacomps = one_based_indices(
        options.get("icacomps"),
        limit=int(np.asarray(EEG.get("icaweights", [])).shape[0]) if np.asarray(EEG.get("icaweights", [])).size else 0,
        default_all=False,
    )
    out, _command = pop_eegthresh(
        EEG, 1, electrodes, -threshold, threshold, EEG.get("xmin", 0), EEG.get("xmax", 0), 0, 0, return_com=True
    )
    # EEGLAB computes high-amplitude threshold marks here but leaves the
    # removal block commented out, so rmep only reports epochs actually pruned
    # by the following probability/kurtosis passes.
    rejected: set[int] = set()
    process_data = not bool(icacomps)
    rows = electrodes if process_data else icacomps
    work = out
    remaining = list(range(1, int(work.get("trials", 1) or 1) + 1))
    limit = startprob
    for _iteration in range(12):
        work, _locthresh, _globthresh, _nrej = pop_jointprob(work, int(process_data), rows, limit, limit, 0, 0)
        field = "rejjp" if process_data else "icarejjp"
        marks = np.asarray((work.get("reject") or {}).get(field, []), dtype=bool)
        current = (np.flatnonzero(marks) + 1).tolist()
        if not current:
            break
        if len(current) / max(1, int(work.get("trials", 1))) <= maxrej / 100:
            rejected.update(remaining[index - 1] for index in current)
            work = pop_rejepoch(work, current, 0)
            drop = set(current)
            remaining = [value for index, value in enumerate(remaining, start=1) if index not in drop]
            if int(work.get("trials", 1) or 1) <= 1:
                break
        else:
            limit += 0.5
    if int(work.get("trials", 1) or 1) > 1:
        work, _locthresh, _globthresh, _nrej = pop_rejkurt(work, int(process_data), rows, 6, 6, 0, 0)
        field = "rejkurt" if process_data else "icarejkurt"
        marks = np.asarray((work.get("reject") or {}).get(field, []), dtype=bool)
        current = (np.flatnonzero(marks) + 1).tolist()
        rejected.update(remaining[index - 1] for index in current)
        if current:
            work = pop_rejepoch(work, current, 0)
    rejected_sorted = sorted(rejected)
    if display:
        marked = _autorej_marked_dataset(EEG, rejected_sorted, process_data=process_data, rows=rows)
        browser_data, _row_count = rejection_data(marked, int(process_data))
        open_epoched_rejection_browser(
            marked,
            data=browser_data,
            icacomp=int(process_data),
            elecrange=rows,
            kind="rejauto",
            superpose=0,
            reject=1,
            command=command,
            command_callback=command_callback,
            show=show,
        )
        return marked, rejected_sorted
    return work, rejected_sorted


def _autorej_marked_dataset(
    EEG: dict[str, Any],
    rejected: list[int],
    *,
    process_data: bool,
    rows: list[int],
) -> dict[str, Any]:
    out = copy_eeg(EEG)
    trials = int(out.get("trials", 1) or 1)
    row_count = int(out.get("nbchan", 0) or 0) if process_data else int(np.asarray(out.get("icaweights", [])).shape[0])
    marks = np.zeros(trials, dtype=bool)
    for index in rejected:
        if 1 <= int(index) <= trials:
            marks[int(index) - 1] = True
    row_marks = np.zeros((row_count, trials), dtype=bool)
    selected = np.asarray(rows, dtype=int) - 1
    selected = selected[(selected >= 0) & (selected < row_count)]
    if selected.size:
        row_marks[np.ix_(selected, np.flatnonzero(marks))] = True
    update_reject_fields(out, icacomp=int(process_data), kind="rejauto", reject=marks, reject_e=row_marks)
    return out


def _history_command(options: dict[str, Any]) -> str:
    values: list[Any] = []
    for key in ("startprob", "electrodes", "icacomps", "maxrej", "nogui", "threshold", "eegplot"):
        if key in options:
            values.extend([key, options[key]])
    return (
        "EEG = pop_autorej(EEG, "
        + ", ".join(format_history_value(item, cell_for_sequence=None) for item in values)
        + ");"
    )
