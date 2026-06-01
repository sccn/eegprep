"""Non-browser rejection mark manager."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._rejection import copy_eeg
from eegprep.functions.popfunc.eeg_rejsuperpose import eeg_rejsuperpose
from eegprep.functions.popfunc.pop_rejepoch import pop_rejepoch


def pop_rejmenu(
    EEG: dict[str, Any],
    icacomp: int | bool = 1,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
):
    """Open a compact rejection menu that combines stored non-browser marks."""
    if EEG is None:
        return (None, "") if return_com else None
    if gui is None:
        gui = True
    source = EEG
    if gui:
        source = copy_eeg(EEG)
        result = inputgui(pop_rejmenu_dialog_spec(source, icacomp), renderer=renderer)
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
    if reject and np.asarray((out.get("reject") or {}).get("rejglobal", [])).any():
        out, reject_command = pop_rejepoch(out, out["reject"]["rejglobal"], 0, return_com=True)
        command = f"{command} {reject_command}"
    return (out, command) if return_com else out


def pop_rejmenu_dialog_spec(EEG: dict[str, Any], icacomp: int | bool = 1) -> DialogSpec:
    """Return the EEGPrep non-browser rejection menu spec."""
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

    def add_row(weights: tuple[float, ...], *row_controls: ControlSpec) -> None:
        geometry.append(weights)
        controls.extend(row_controls)

    def spacer_row() -> None:
        add_row((1,), ControlSpec("spacer"))

    def section(title_text: str, swatch_tag: str) -> None:
        add_row(
            (0.9, 0.18, 1.95),
            ControlSpec("text", title_text, font_weight="bold"),
            ControlSpec("pushbutton", tag=swatch_tag, enabled=False),
            ControlSpec("spacer"),
        )

    def scroll_callback(button: str) -> CallbackSpec:
        return CallbackSpec(
            "open_rejection_browser",
            params={
                "button": button,
                "eeg": EEG,
                "icacomp": 1 if is_data else 0,
                "count_tags": ("mantrial",),
                "count_fields": {"mantrial": prefix + "rejmanual"},
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
            ControlSpec("pushbutton", "Calc / Plot", enabled=False),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("pushbutton", "HELP", enabled=False),
        )
        spacer_row()

    add_row(
        (0.9, 0.18, 0.15, 0.62, 0.62, 0.52),
        ControlSpec("text", "Mark trials by appearance", font_weight="bold"),
        ControlSpec("pushbutton", tag="butmanual", enabled=False),
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
        ControlSpec("pushbutton", "Calc / Plot", enabled=False),
        ControlSpec("spacer"),
        ControlSpec("spacer"),
        ControlSpec("pushbutton", "HELP", enabled=False),
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
            ControlSpec("pushbutton", "Calculate", enabled=False),
            ControlSpec(
                "pushbutton",
                activity,
                tag=f"scroll{edit_prefix}",
                enabled=True,
                callback=scroll_callback(f"scroll{edit_prefix}"),
            ),
            ControlSpec("pushbutton", "PLOT", enabled=False),
            ControlSpec("pushbutton", "HELP", enabled=False),
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
        size=(1000, 1160),
        geometry=tuple(geometry),
        controls=tuple(controls),
        ok_label="Close (keep marks)",
        cancel_label="Cancel",
        button_size=(168, 22),
        content_margins=(32, 30, 32, 24),
        row_spacing=12,
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


def _count(value: Any) -> int:
    return int(np.asarray(value if value is not None else [], dtype=bool).sum())


def _ica_count(EEG: dict[str, Any]) -> int:
    return int(np.asarray(EEG.get("icaweights", [])).shape[0])
