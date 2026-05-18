"""EEGLAB-style pop wrapper for clean_rawdata."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import (
    format_history_value,
    parse_key_value_args,
    parse_text_tokens,
)
from eegprep.plugins.clean_rawdata.clean_artifacts import clean_artifacts


logger = logging.getLogger(__name__)
_SHOW_VIS_ARTIFACTS_KEY = "_show_vis_artifacts"

_OPTION_ALIASES = {
    "channelcriterion": "ChannelCriterion",
    "linenoisecriterion": "LineNoiseCriterion",
    "burstcriterion": "BurstCriterion",
    "windowcriterion": "WindowCriterion",
    "highpass": "Highpass",
    "flatlinecriterion": "FlatlineCriterion",
    "burstrejection": "BurstRejection",
    "distance": "Distance",
    "channels": "Channels",
    "channels_ignore": "Channels_ignore",
}


def pop_clean_rawdata(
    EEG,
    *args,
    gui: bool | None = None,
    renderer=None,
    return_com: bool = False,
    **kwargs,
):
    """Clean continuous EEG data using the clean_rawdata workflow."""
    if EEG is None:
        return (None, "") if return_com else None
    options = _normalise_options(parse_key_value_args(args, kwargs, lowercase_keys=False))
    show_vis_artifacts = bool(options.pop(_SHOW_VIS_ARTIFACTS_KEY, False))
    if gui is None:
        gui = not bool(options)
    if gui:
        gui_options = _run_gui(EEG[0] if isinstance(EEG, list) else EEG, renderer=renderer)
        if gui_options is None:
            return (EEG, "") if return_com else EEG
        show_vis_artifacts = bool(gui_options.pop(_SHOW_VIS_ARTIFACTS_KEY, False))
        options.update(gui_options)
    if isinstance(EEG, list):
        output = [pop_clean_rawdata(item, gui=False, **options) for item in EEG]
        command = _history_command(options)
        if show_vis_artifacts:
            _notify_vis_artifacts_unavailable()
        return (output, command) if return_com else output
    if int(EEG.get("trials", 1) or 1) > 1 or np.asarray(EEG.get("data")).ndim == 3:
        raise ValueError("Input data must be continuous. This data seems epoched.")
    clean_eeg, _hp, _bur, _removed_channels = clean_artifacts(EEG, **options)
    command = _history_command(options)
    if show_vis_artifacts:
        _notify_vis_artifacts_unavailable()
    return (clean_eeg, command) if return_com else clean_eeg


def pop_clean_rawdata_dialog_spec(EEG) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_clean_rawdata``."""
    chanlocs = _chanloc_records(EEG.get("chanlocs", []))
    labels = tuple(str(chan.get("labels", "")) for chan in chanlocs if isinstance(chan, dict))
    winsize = max(0.5, 1.5 * float(EEG.get("nbchan", 1)) / float(EEG.get("srate", 1)))
    row4 = (0.1, 0.8, 0.2, 0.3)
    row = (0.1, 1, 0.3)
    row2 = (0.1, 1.2, 0.1)
    return DialogSpec(
        title="pop_clean_rawdata()",
        function_name="pop_clean_rawdata",
        eeglab_source="plugins/clean_rawdata/pop_clean_rawdata.m",
        geometry=(1, row, 1, 1, row4, row4, row, row, row, 1, 1, row, row2, row2, 1, 1, row, row, 1, 1),
        geomvert=(1, 1, 0.3, 1, 1, 1, 1, 1, 1, 0.3, 1, 1, 1, 1, 0.3, 1, 1, 1, 0.3, 1),
        size=(681, 733),
        help_text="pophelp('pop_clean_rawdata')",
        controls=(
            ControlSpec(
                "checkbox",
                "Remove channel drift (data not already high-pass filtered)",
                tag="filter",
                value=False,
                font_weight="bold",
                callback=CallbackSpec("toggle_enabled", params={"source": "filter", "targets": ("filterfreqs",)}),
            ),
            ControlSpec("spacer"),
            ControlSpec("text", "Linear filter (FIR) transition band [lo hi] in Hz", enabled=False),
            ControlSpec("edit", tag="filterfreqs", value="0.25 0.75", enabled=False),
            ControlSpec("spacer"),
            ControlSpec("checkbox", "Process/remove channels", tag="chanrm", value=True, font_weight="bold"),
            ControlSpec("spacer"),
            ControlSpec("checkbox", "Only consider these channels", tag="chanuseflag", value=False),
            ControlSpec(
                "pushbutton",
                "...",
                tag="chanuse_button",
                enabled=bool(labels),
                callback=CallbackSpec(
                    "select_channels",
                    params={"button": "chanuse_button", "target": "chanuse", "channels": labels},
                    matlab_callback="pop_chansel(get(gcbf, 'userdata'), 'field', 'labels')",
                ),
            ),
            ControlSpec("edit", tag="chanuse", value=""),
            ControlSpec("spacer"),
            ControlSpec("checkbox", "Ignore these channels (ECG, EMG, ...)", tag="chanignoreflag", value=False),
            ControlSpec(
                "pushbutton",
                "...",
                tag="chanignore_button",
                enabled=bool(labels),
                callback=CallbackSpec(
                    "select_channels",
                    params={"button": "chanignore_button", "target": "chanignore", "channels": labels},
                    matlab_callback="pop_chansel(get(gcbf, 'userdata'), 'field', 'labels')",
                ),
            ),
            ControlSpec("edit", tag="chanignore", value=""),
            ControlSpec("spacer"),
            ControlSpec("checkbox", "Remove channel if it is flat for more than (seconds)", tag="rmflat", value=True),
            ControlSpec("edit", tag="rmflatsec", value="5"),
            ControlSpec("spacer"),
            ControlSpec("checkbox", "Max acceptable high-frequency noise std dev", tag="rmnoise", value=True),
            ControlSpec("edit", tag="rmnoiseval", value="4"),
            ControlSpec("spacer"),
            ControlSpec("checkbox", "Min acceptable correlation with nearby chans [0-1]", tag="rmcorr", value=True),
            ControlSpec("edit", tag="rmcorrval", value="0.8"),
            ControlSpec("spacer"),
            ControlSpec(
                "checkbox",
                "Perform Artifact Subspace Reconstruction bad burst correction/rejection",
                tag="asr",
                value=True,
                font_weight="bold",
            ),
            ControlSpec("spacer"),
            ControlSpec("text", f"Max acceptable {winsize:1.1f} second window std dev"),
            ControlSpec("edit", tag="asrstdval", value="20"),
            ControlSpec("spacer"),
            ControlSpec("checkbox", "Use Riemanian distance metric (not Euclidean) - beta", tag="distance", value=False),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("checkbox", "Remove bad data periods (when uncheck, correct using ASR)", tag="asrrej", value=True),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec(
                "checkbox",
                "Additional removal of bad data periods",
                tag="rejwin",
                value=True,
                font_weight="bold",
            ),
            ControlSpec("spacer"),
            ControlSpec("text", "Acceptable [min max] channel RMS range (+/- std dev)"),
            ControlSpec("edit", tag="rejwinval1", value="-Inf 7"),
            ControlSpec("spacer"),
            ControlSpec("text", "Maximum out-of-bound channels (%)"),
            ControlSpec("edit", tag="rejwinval2", value="25"),
            ControlSpec("spacer"),
            ControlSpec("checkbox", "Pop up scrolling data window with rejected data highlighted", tag="vis", value=False),
        ),
    )


def _chanloc_records(chanlocs):
    if chanlocs is None:
        return []
    if isinstance(chanlocs, dict):
        return [chanlocs]
    if isinstance(chanlocs, np.ndarray):
        return list(chanlocs.ravel())
    return list(chanlocs)


def _run_gui(EEG, renderer=None):
    result = inputgui(pop_clean_rawdata_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    options: dict[str, Any] = {
        "FlatlineCriterion": "off",
        "ChannelCriterion": "off",
        "LineNoiseCriterion": "off",
        "Highpass": "off",
        "BurstCriterion": "off",
        "WindowCriterion": "off",
        "BurstRejection": False,
        "Distance": "Euclidian",
    }
    if result.get("filter"):
        options["Highpass"] = _parse_numeric_text(result.get("filterfreqs", ""))
    if result.get("chanrm"):
        if result.get("chanignoreflag"):
            options["Channels_ignore"] = parse_text_tokens(result.get("chanignore", ""))
        if result.get("chanuseflag"):
            options["Channels"] = parse_text_tokens(result.get("chanuse", ""))
        if result.get("rmflat"):
            options["FlatlineCriterion"] = float(result.get("rmflatsec", 5))
        if result.get("rmcorr"):
            options["ChannelCriterion"] = float(result.get("rmcorrval", 0.8))
        if result.get("rmnoise"):
            options["LineNoiseCriterion"] = float(result.get("rmnoiseval", 4))
    if result.get("asr"):
        options["BurstCriterion"] = float(result.get("asrstdval", 20))
        if result.get("distance"):
            options["Distance"] = "Riemannian"
    if result.get("rejwin"):
        options["WindowCriterionTolerances"] = _parse_numeric_text(result.get("rejwinval1", ""))
        options["WindowCriterion"] = float(result.get("rejwinval2", 25)) / 100.0
    if result.get("asrrej") and options["BurstCriterion"] != "off":
        options["BurstRejection"] = True
    options[_SHOW_VIS_ARTIFACTS_KEY] = bool(result.get("vis"))
    return options


def _normalise_options(options):
    normalised = {}
    for key, value in options.items():
        canonical = _OPTION_ALIASES.get(str(key).lower(), key)
        if canonical == "BurstRejection":
            value = _as_bool(value)
        normalised[canonical] = value
    return normalised


def _as_bool(value):
    if isinstance(value, str):
        return value.lower() == "on"
    return bool(value)


def _parse_numeric_text(text):
    values = []
    for value in re.split(r"[\s,]+", str(text).strip().strip("[]")):
        if not value:
            continue
        if value.lower() == "-inf":
            values.append(-np.inf)
        elif value.lower() == "inf":
            values.append(np.inf)
        else:
            values.append(float(value))
    return values


def _history_command(options):
    if not options:
        return "EEG = pop_clean_rawdata(EEG);"
    parts = []
    for key, value in options.items():
        parts.extend([_clean_rawdata_history_value(key), _clean_rawdata_history_value(value)])
    return f"EEG = pop_clean_rawdata(EEG, {', '.join(parts)});"


def _clean_rawdata_history_value(value):
    return format_history_value(value, bool_style="onoff", empty_sequence="{}")


def _notify_vis_artifacts_unavailable():
    message = (
        "The clean_rawdata rejected-data scrolling viewer is not yet available in EEGPrep. "
        "The dataset was still cleaned."
    )
    try:
        from PySide6 import QtWidgets
    except ImportError:
        logger.warning(message)
        return
    app = QtWidgets.QApplication.instance()
    if app is None:
        logger.warning(message)
        return
    QtWidgets.QMessageBox.information(app.activeWindow(), "EEGPrep", message)
