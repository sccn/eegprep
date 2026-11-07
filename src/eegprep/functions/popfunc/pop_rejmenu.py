"""Calculate, inspect, and combine epoch rejection marks."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._eegplot_rejection import DEFAULT_REJECTION_COLORS, open_epoched_rejection_browser
from eegprep.functions.popfunc._rejection import copy_eeg, parse_numeric_sequence, rejection_data
from eegprep.functions.popfunc.eeg_rejsuperpose import eeg_rejsuperpose
from eegprep.functions.popfunc.pop_eegthresh import pop_eegthresh
from eegprep.functions.popfunc.pop_jointprob import pop_jointprob
from eegprep.functions.popfunc.pop_rejepoch import pop_rejepoch
from eegprep.functions.popfunc.pop_rejkurt import pop_rejkurt
from eegprep.functions.popfunc.pop_rejspec import pop_rejspec
from eegprep.functions.popfunc.pop_rejtrend import pop_rejtrend


_METHODS = {
    "thresh": (pop_eegthresh, "rejthresh", "threshtrial"),
    "const": (pop_rejtrend, "rejconst", "consttrial"),
    "ent": (pop_jointprob, "rejjp", "enttrial"),
    "kurt": (pop_rejkurt, "rejkurt", "kurttrial"),
    "freq": (pop_rejspec, "rejfreq", "freqtrial"),
}


def pop_rejmenu(
    EEG: dict[str, Any],
    icacomp: int | bool = 1,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
):
    """Calculate and review data or ICA epoch marks before combining them."""
    if EEG is None:
        return (None, "") if return_com else None
    if gui is None:
        gui = True
    source = EEG
    commands: list[str] = []
    if gui:
        source = copy_eeg(EEG)
        result = inputgui(pop_rejmenu_dialog_spec(source, icacomp, commands=commands), renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        flags = [
            int(bool(result.get(tag, False))) for tag in ("manual", "thresh", "const", "jp", "kurt", "freq", "other")
        ]
        reject = int(bool(result.get("reject", False)))
    else:
        flags = [1, 1, 1, 1, 1, 1, 1]
        reject = 0
    out, command = eeg_rejsuperpose(source, icacomp, *flags, return_com=True)
    commands.append(command)
    if reject and np.asarray((out.get("reject") or {}).get("rejglobal", [])).any():
        out, reject_command = pop_rejepoch(out, out["reject"]["rejglobal"], 0, return_com=True)
        commands.append(reject_command)
    return (out, " ".join(commands)) if return_com else out


def pop_rejmenu_dialog_spec(
    EEG: dict[str, Any], icacomp: int | bool = 1, *, commands: list[str] | None = None
) -> DialogSpec:
    """Build a rejection menu operating on EEG and recording accepted actions."""
    if commands is None:
        commands = []
    is_data = int(bool(icacomp))
    prefix = "" if is_data else "ica"
    reject = EEG.get("reject") or {}
    row_label = "Electrode(s)" if is_data else "Component(s)"
    row_range = f"1:{int(EEG.get('nbchan', 0) or 0)}" if is_data else f"1:{_ica_count(EEG)}"
    unit = "uV" if is_data else "std. dev."
    activity = "Scroll Data" if is_data else "Scroll Acts."
    title = (
        "Reject trials using data statistics - pop_rejmenu()"
        if is_data
        else "Reject trials using component activity statistics - pop_rejmenu()"
    )
    geometry: list[tuple[float, ...]] = []
    controls: list[ControlSpec] = []

    def action_button(label: str, method: str, mode: str) -> ControlSpec:
        tag = f"{mode}_{method}"
        return ControlSpec(
            "pushbutton",
            label,
            tag=tag,
            callback=CallbackSpec(
                "run_dialog_action",
                params={
                    "button": tag,
                    "function": "pop_rejmenu",
                    "action": partial(_run_method, EEG, is_data, method, mode, commands),
                },
            ),
        )

    def help_button(method: str) -> ControlSpec:
        tag = f"help_{method}"
        return ControlSpec(
            "pushbutton",
            "HELP",
            tag=tag,
            callback=CallbackSpec(
                "show_help",
                params={"button": tag, "function": _METHODS[method][0].__name__},
            ),
        )

    def color_button(tag: str, family: str) -> ControlSpec:
        return ControlSpec(
            "pushbutton",
            tag=tag,
            callback=CallbackSpec(
                "select_color",
                params={
                    "button": tag,
                    "eeg": EEG,
                    "field": f"rej{family}col",
                    "default": DEFAULT_REJECTION_COLORS[family],
                },
            ),
        )

    def add_row(weights: tuple[float, ...], *row_controls: ControlSpec) -> None:
        geometry.append(weights)
        controls.extend(row_controls)

    def spacer_row() -> None:
        add_row((1,), ControlSpec("spacer"))

    def section(title_text: str, swatch_tag: str) -> None:
        family = {"buttrend": "const", "butspec": "freq"}.get(swatch_tag, swatch_tag[3:])
        add_row(
            (0.9, 0.18, 1.95),
            ControlSpec("text", title_text, font_weight="bold"),
            color_button(swatch_tag, family),
            ControlSpec("spacer"),
        )

    def scroll_callback(button: str) -> CallbackSpec:
        return CallbackSpec(
            "open_rejection_browser",
            params={
                "button": button,
                "eeg": EEG,
                "icacomp": 1 if is_data else 0,
                "count_tags": ("mantrial", *[item[2] for item in _METHODS.values()]),
                "count_fields": {
                    "mantrial": prefix + "rejmanual",
                    **{count_tag: prefix + field for _function, field, count_tag in _METHODS.values()},
                },
                "commands": commands,
            },
            matlab_callback="pop_eegplot(EEG, icacomp, rejstatus, 0)",
        )

    def threshold_rows(
        prefix_text: str,
        edit_prefix: str,
        count_tag: str,
        count_value: Any,
        swatch_tag: str | None = None,
        unit_label: str | None = None,
    ) -> None:
        display_unit = unit_label or unit
        section(prefix_text, swatch_tag or f"but{edit_prefix}")
        add_row(
            (1.2, 0.75, 1.2, 0.75),
            ControlSpec("text", f"Upper limit(s) ({display_unit})"),
            ControlSpec("edit", tag=f"{edit_prefix}pos", value="25"),
            ControlSpec("text", f"Lower limit(s) ({display_unit})"),
            ControlSpec("edit", tag=f"{edit_prefix}neg", value="-25"),
        )
        add_row(
            (1.2, 0.75, 1.2, 0.75),
            ControlSpec("text", "Start time(s) (ms)" if edit_prefix == "thresh" else "Low frequency(s) (Hz)"),
            ControlSpec(
                "edit",
                tag=f"{edit_prefix}start",
                value=f"{float(EEG.get('xmin', 0.0)) * 1000:g}" if edit_prefix == "thresh" else "0",
            ),
            ControlSpec("text", "Ending time(s) (ms)" if edit_prefix == "thresh" else "High frequency(s) (Hz)"),
            ControlSpec(
                "edit",
                tag=f"{edit_prefix}end",
                value=f"{float(EEG.get('xmax', 0.0)) * 1000:g}" if edit_prefix == "thresh" else "50",
            ),
        )
        add_row(
            (1.2, 0.75, 1.2, 0.75),
            ControlSpec("text", row_label),
            ControlSpec("edit", tag=f"{edit_prefix}elec", value=row_range),
            ControlSpec("text", "Currently marked trials"),
            ControlSpec("text", str(_count(count_value)), tag=count_tag),
        )
        add_row(
            (1.0, 0.2, 1.0, 1.0),
            action_button("Calc / Plot", edit_prefix, "scroll"),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            help_button(edit_prefix),
        )
        spacer_row()

    add_row(
        (0.9, 0.18, 0.15, 0.62, 0.62, 0.52),
        ControlSpec("text", "Mark trials by appearance", font_weight="bold"),
        color_button("butmanual", "manual"),
        ControlSpec("spacer"),
        ControlSpec("pushbutton", activity, tag="scrollmanual", enabled=True, callback=scroll_callback("scrollmanual")),
        ControlSpec("text", "Marked trials"),
        ControlSpec("text", str(_count(reject.get(prefix + "rejmanual"))), tag="mantrial"),
    )
    spacer_row()
    threshold_rows("Find abnormal values", "thresh", "threshtrial", reject.get(prefix + "rejthresh"))

    section("Find abnormal trends", "buttrend")
    add_row(
        (1.2, 0.75, 1.2, 0.75),
        ControlSpec("text", f"Max slope ({'uV/epoch' if is_data else 'std. dev./epoch'})"),
        ControlSpec("edit", tag="constpnts", value="50"),
        ControlSpec("text", "R-squared limit (0 to 1)"),
        ControlSpec("edit", tag="conststd", value="0.3"),
    )
    add_row(
        (1.2, 0.75, 1.2, 0.75),
        ControlSpec("text", row_label),
        ControlSpec("edit", tag="constelec", value=row_range),
        ControlSpec("text", "Currently marked trials"),
        ControlSpec("text", str(_count(reject.get(prefix + "rejconst"))), tag="consttrial"),
    )
    add_row(
        (1.0, 0.2, 1.0, 1.0),
        action_button("Calc / Plot", "const", "scroll"),
        ControlSpec("spacer"),
        ControlSpec("spacer"),
        help_button("const"),
    )
    spacer_row()

    for title_text, edit_prefix, loc_default, count_tag, count_value in (
        ("Find improbable data", "ent", "5" if is_data else "20", "enttrial", reject.get(prefix + "rejjp")),
        (
            "Find abnormal distributions",
            "kurt",
            "5" if is_data else "20",
            "kurttrial",
            reject.get(prefix + "rejkurt"),
        ),
    ):
        section(title_text, f"but{'jp' if edit_prefix == 'ent' else edit_prefix}")
        add_row(
            (1.2, 0.75, 1.2, 0.75),
            ControlSpec(
                "text",
                "Single-channel limit (std. dev.)" if is_data else "Single-comp. limit (std. dev.)",
            ),
            ControlSpec("edit", tag=f"{edit_prefix}loc", value=loc_default),
            ControlSpec("text", "All channels limit (std. dev.)" if is_data else "All comp. limit (std. dev.)"),
            ControlSpec("edit", tag=f"{edit_prefix}glob", value="5"),
        )
        add_row(
            (1.2, 0.75, 1.2, 0.75),
            ControlSpec("text", row_label),
            ControlSpec("edit", tag=f"{edit_prefix}elec", value=row_range),
            ControlSpec("text", "Currently marked trials"),
            ControlSpec("text", str(_count(count_value)), tag=count_tag),
        )
        add_row(
            (0.8, 0.8, 0.8, 0.8),
            action_button("Calculate", edit_prefix, "calculate"),
            action_button(activity, edit_prefix, "scroll"),
            action_button("PLOT", edit_prefix, "plot"),
            help_button(edit_prefix),
        )
        spacer_row()

    threshold_rows(
        "Find abnormal spectra (slow)",
        "freq",
        "freqtrial",
        reject.get(prefix + "rejfreq"),
        "butspec",
        "dB",
    )
    add_row((1,), ControlSpec("text", "Plotting options", font_weight="bold"))
    add_row(
        (1,),
        ControlSpec(
            "popupmenu",
            "Show only new trials marked for rejection|Show previous and new trials marked for rejection|"
            "Show all trials marked for rejection by the measure selected above or checked below",
            tag="rejstatus",
            value=3,
        ),
    )
    add_row(
        (1, 1, 1),
        ControlSpec("checkbox", "Abnormal appearance", tag="manual", value=True),
        ControlSpec("checkbox", "Abnormal values", tag="thresh", value=True),
        ControlSpec("checkbox", "Abnormal trends", tag="const", value=True),
    )
    add_row(
        (1, 1, 1),
        ControlSpec("checkbox", "Improbable epochs", tag="jp", value=True),
        ControlSpec("checkbox", "Abnormal distributions", tag="kurt", value=True),
        ControlSpec("checkbox", "Abnormal spectra", tag="freq", value=True),
    )
    add_row(
        (1, 1),
        ControlSpec("checkbox", "Include other rejection family", tag="other", value=False),
        ControlSpec("checkbox", "Reject marked epochs when closing", tag="reject", value=False),
    )
    return DialogSpec(
        title=title,
        function_name="pop_rejmenu",
        eeglab_source="functions/adminfunc/pop_rejmenu.m",
        help_text="pophelp('pop_rejmenu')",
        size=(1073, 994),
        geometry=tuple(geometry),
        controls=tuple(controls),
        ok_label="Close (keep marks)",
        cancel_label="Cancel",
        button_size=(168, 22),
        content_margins=(32, 30, 32, 24),
        row_spacing=3,
        extra_stylesheet="""
            QDialog#pop_rejmenu QPushButton {
                min-width: 145px;
                max-width: 245px;
            }
            QDialog#pop_rejmenu QPushButton#butmanual { background: #ff9999; min-width: 30px; max-width: 30px; padding: 0; }
            QDialog#pop_rejmenu QPushButton#butthresh { background: #ff6666; min-width: 30px; max-width: 30px; padding: 0; }
            QDialog#pop_rejmenu QPushButton#buttrend { background: #ffcc66; min-width: 30px; max-width: 30px; padding: 0; }
            QDialog#pop_rejmenu QPushButton#butjp { background: #99ccff; min-width: 30px; max-width: 30px; padding: 0; }
            QDialog#pop_rejmenu QPushButton#butkurt { background: #99ff99; min-width: 30px; max-width: 30px; padding: 0; }
            QDialog#pop_rejmenu QPushButton#butspec { background: #cc99ff; min-width: 30px; max-width: 30px; padding: 0; }
            QDialog#pop_rejmenu QPushButton:disabled {
                color: #777777;
            }
        """,
    )


def _run_method(EEG, icacomp, method, mode, commands, values, update):
    function, kind, _count_tag = _METHODS[method]
    prefix = "" if icacomp else "ica"

    def numbers(tag):
        try:
            result = parse_numeric_sequence(values[tag], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"pop_rejmenu: {tag} must contain numbers. Enter a number, space-separated values, "
                "or a range such as 1:3."
            ) from exc
        if not result:
            raise ValueError(f"pop_rejmenu: {tag} is empty. Enter a numeric value before calculating.")
        return result

    options: dict[str, Any] = {"elecrange": numbers(f"{method}elec"), "gui": False, "return_com": True}
    if method == "freq":
        options.update(
            threshold=np.column_stack(np.broadcast_arrays(numbers("freqneg"), numbers("freqpos"))),
            freqlimits=np.column_stack(np.broadcast_arrays(numbers("freqstart"), numbers("freqend"))),
            eegplotreject=0,
        )
    else:
        options["reject"] = 0
        if method == "thresh":
            options.update(
                negthresh=numbers("threshneg"),
                posthresh=numbers("threshpos"),
                starttime=np.asarray(numbers("threshstart")) / 1000,
                endtime=np.asarray(numbers("threshend")) / 1000,
            )
        elif method == "const":
            options.update(winsize=int(EEG["pnts"]), minslope=numbers("constpnts")[0], minstd=numbers("conststd")[0])
        else:
            options.update(locthresh=numbers(f"{method}loc"), globthresh=numbers(f"{method}glob"))

    def store(output, command):
        EEG.clear()
        EEG.update(output)
        if command:
            commands.append(command)
        update({tag: str(_count(EEG["reject"].get(prefix + field))) for _function, field, tag in _METHODS.values()})

    try:
        output, command = function(EEG, icacomp, **options)
    except ValueError as exc:
        raise ValueError(
            f"pop_rejmenu / {function.__name__}: {exc}. Check the {method} channel/component range "
            "and limits, then calculate again."
        ) from exc
    store(output, command)
    if mode == "plot":
        _plot_statistics(EEG, icacomp, method, options)
    if mode == "scroll":
        EEG["reject"]["disprej"] = [family for family in DEFAULT_REJECTION_COLORS if values.get(family)]
        data, _row_count = rejection_data(EEG, icacomp)
        open_epoched_rejection_browser(
            EEG,
            data=data,
            icacomp=icacomp,
            elecrange=[int(value) for value in options["elecrange"]],
            kind=kind,
            superpose=int(values["rejstatus"]) - 1,
            reject=0,
            command="",
            command_callback=store,
        )


def _plot_statistics(EEG, icacomp, method, options):
    field = ("" if icacomp else "ica") + ("jp" if method == "ent" else "kurt")
    title = "Joint probability" if method == "ent" else "Kurtosis"
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6), layout="constrained")
    figure.suptitle(f"{title} — {EEG.get('setname', '')}")
    trials = np.arange(1, int(EEG["trials"]) + 1)
    for axis, scores, limits, label in (
        (
            axes[0],
            np.asarray(EEG["stats"][field + "E"]).T,
            options["locthresh"],
            "Single-channel" if icacomp else "Single-component",
        ),
        (axes[1], EEG["stats"][field], options["globthresh"], "All channels" if icacomp else "All components"),
    ):
        axis.plot(trials, scores, ".-")
        for limit in [-abs(limits[0]), abs(limits[0])] if len(limits) == 1 else limits:
            axis.axhline(limit, color="red", linestyle="--")
        axis.set_ylabel(f"{label}\n(std. dev.)")
        axis.grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoch (1-based)")
    plt.show(block=False)


def _count(value: Any) -> int:
    return int(np.asarray(value if value is not None else [], dtype=bool).sum())


def _ica_count(EEG: dict[str, Any]) -> int:
    return int(np.asarray(EEG.get("icaweights", [])).shape[0])
