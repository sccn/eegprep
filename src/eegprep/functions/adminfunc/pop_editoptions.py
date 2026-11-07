"""Edit and persist EEGPrep workflow preferences."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.miscfunc.value_parsing import parse_key_value_args
from eegprep.functions.popfunc._pop_utils import format_history_value


_OPTION_LABELS = {
    "option_storedisk": "Keep at most one dataset in memory (save other datasets to disk)",
    "option_savetwofiles": "Save two files for each dataset (header and data)",
    "option_saveversion6": "Write files in MATLAB v6.5 format",
    "option_saveasstruct": "Save EEG fields as individual variables in the file",
    "option_parallel": "Use the parallel toolbox when processing multiple datasets",
    "option_single": "Use single precision (32-bit instead of 64-bit) in memory",
    "option_memmapdata": "Use memory-mapped arrays when loading two-file datasets",
    "option_eegobject": "Use the MATLAB EEG object instead of a structure",
    "option_computeica": "Precompute ICA activations (requires more RAM)",
    "option_scaleicarms": "Scale ICA component activities to RMS microvolt",
    "option_rememberfolder": "Remember the previous dataset folder when browsing",
    "option_donotusetoolboxes": "Do not use additional MATLAB toolboxes",
    "option_showadvanced": "Show advanced options (reopen this dialog)",
    "option_boundary99": "Use type -99 for boundary events with numerical event types",
    "option_showpendingplugins": "Show plugins pending approval",
    "option_allmenus": "Show all menu items from previous EEGLAB versions",
    "option_htmlingraphics": "Allow HTML in graphics",
    "option_checkversion": "Check for new versions at startup",
    "option_cachesize": "STUDY cache size in MB",
    "option_native_dialogs": "Use native OS file dialogs",
}
_DEFAULT_OPTIONS = {
    "option_storedisk",
    "option_savetwofiles",
    "option_parallel",
    "option_computeica",
    "option_rememberfolder",
    "option_showadvanced",
    "option_boundary99",
    "option_allmenus",
    "option_checkversion",
    "option_cachesize",
    "option_native_dialogs",
}
_MATLAB_ONLY_OPTIONS = {"option_parallel", "option_eegobject", "option_donotusetoolboxes"}


def eegprep_options_path() -> Path:
    """Return the standalone EEGPrep preference file."""
    return Path.home() / ".eegprep" / "options.json"


def pop_editoptions(
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    persist: bool = False,
    options_path: str | Path | None = None,
    **updates: Any,
) -> str:
    """Edit options and return a replayable command.

    With no arguments, open Preferences. Name/value pairs or keywords update
    ``EEG_OPTIONS``; an initial dictionary selects a separate target instead.
    GUI changes persist in ``~/.eegprep/options.json``. Use ``persist=True`` to
    save scripted changes too, or ``options_path`` to select a different file.
    """
    target = EEG_OPTIONS
    pair_args = args
    if args and isinstance(args[0], dict):
        target = args[0]
        pair_args = args[1:]
    options = parse_key_value_args(pair_args, updates, lowercase_keys=False, lowercase_kwargs=False)
    if gui is None:
        gui = not args and not updates
    path = Path(options_path) if options_path is not None else eegprep_options_path()
    if gui:
        result = inputgui(pop_editoptions_dialog_spec(target, path), renderer=renderer)
        if result is None:
            return ""
        options.update({key: value for key, value in result.items() if key in target})
        persist = True
    validated = _validate_options(target, options)
    if persist:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(dict(target, **validated), stream, indent=2)
                stream.write("\n")
                stream.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    target.update(validated)
    arguments = ", ".join(f"{format_history_value(key)}, {value}" for key, value in validated.items())
    return f"pop_editoptions({arguments});" if arguments else ""


def load_eegprep_options(options_path: str | Path | None = None) -> None:
    """Load saved preferences into the process's shared option dictionary."""
    path = Path(options_path) if options_path is not None else eegprep_options_path()
    if not path.exists():
        return
    try:
        options = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(options, dict):
            raise ValueError("expected a JSON object of option names and values")
        validated = _validate_options(EEG_OPTIONS, options)
    except (ValueError, KeyError) as exc:
        raise ValueError(f"pop_editoptions: invalid option file {path}: {exc}. Correct or remove this file.") from exc
    EEG_OPTIONS.update(validated)


def pop_editoptions_dialog_spec(options: dict[str, Any], path: Path | None = None) -> DialogSpec:
    """Return the memory-options dialog, including advanced preferences."""
    controls = [ControlSpec("text", f"Option file: {path or eegprep_options_path()}")]
    geometry: list[tuple[float, ...]] = [(1,)]
    for key, label in _OPTION_LABELS.items():
        if key not in _DEFAULT_OPTIONS and not options["option_showadvanced"]:
            continue
        enabled = key not in _MATLAB_ONLY_OPTIONS
        controls.extend(
            [
                ControlSpec("text", label),
                ControlSpec(
                    "edit" if key == "option_cachesize" else "checkbox",
                    tag=key,
                    value=str(options[key]) if key == "option_cachesize" else bool(options[key]),
                    enabled=enabled,
                    tooltip=None if enabled else "Requires MATLAB; unavailable in standalone EEGPrep.",
                ),
            ]
        )
        geometry.append((6, 1))
    return DialogSpec(
        title="Memory options - pop_editoptions()",
        function_name="pop_editoptions",
        eeglab_source="functions/adminfunc/pop_editoptions.m",
        help_text="pophelp('pop_editoptions')",
        controls=tuple(controls),
        geometry=tuple(geometry),
        size=(830, 560),
        scrollable=True,
    )


def _validate_options(target: dict[str, Any], options: dict[str, Any]) -> dict[str, int]:
    validated = {}
    for key, value in options.items():
        if key not in target:
            raise KeyError(f"pop_editoptions: unknown option {key!r}. Use a name from EEG_OPTIONS.")
        try:
            numeric = float(value)
            if not numeric.is_integer() or (numeric < 0 if key == "option_cachesize" else numeric not in (0, 1)):
                raise ValueError
        except (TypeError, ValueError) as exc:
            expected = "a nonnegative whole number of MB" if key == "option_cachesize" else "0 or 1"
            raise ValueError(f"pop_editoptions: {key} must be {expected}; enter a valid value.") from exc
        validated[key] = int(numeric)
    return validated


load_eegprep_options()
