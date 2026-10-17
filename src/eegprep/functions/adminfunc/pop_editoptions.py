"""Edit EEGPrep options used by EEGLAB-style workflows."""

from __future__ import annotations

from typing import Any

from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.functions.miscfunc.value_parsing import parse_key_value_args


def pop_editoptions(*args: Any, **updates: Any) -> str:
    """Update EEGPrep's EEGLAB-compatible options.

    A mapping in the first positional argument selects the dictionary to
    update. Remaining positional arguments are name/value pairs such as
    ``pop_editoptions("option_storedisk", 1)``. Keywords update the same
    names. Unknown names raise ``KeyError``. A bool stored into an ``int``
    option is stored as ``0`` or ``1``.
    """
    target = EEG_OPTIONS
    pair_args = args
    if args and isinstance(args[0], dict):
        target = args[0]
        pair_args = args[1:]
    options = parse_key_value_args(pair_args, updates, lowercase_keys=False, lowercase_kwargs=False)
    for key, value in options.items():
        if key not in target:
            raise KeyError(f"Unknown EEGPrep option: {key}")
        target[key] = _option_value(target[key], value)
    return "LASTCOM = pop_editoptions();"


def _option_value(current: Any, value: Any) -> Any:
    if isinstance(current, bool) or not isinstance(current, int):
        return value
    if isinstance(value, bool):
        return int(value)
    return value
