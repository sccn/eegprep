"""EEGLAB-style wrapper for CORRMAP component matching."""

from __future__ import annotations

from typing import Any

from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.popfunc.plot_utils import python_literal
from eegprep.functions.studyfunc.corrmap import corrmap


def pop_corrmap(
    STUDY: dict[str, Any] | None,
    ALLEEG: Any,
    n_tmp: int,
    index: int,
    *args: Any,
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Run noninteractive CORRMAP matching and optionally return history.

    Scripted calls suppress CORRMAP's summary figures, as in the MATLAB
    ``pop_corrmap`` command-line path. Use the returned ``CORRMAP`` dictionary
    for correlations, selected components, polarities, and average maps.
    """
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    options["plot"] = "off"
    result, study, datasets = corrmap(STUDY, ALLEEG, n_tmp, index, **options)
    command_options = {key: value for key, value in options.items() if key != "plot"}
    pieces = ["STUDY", "ALLEEG", python_literal(n_tmp), python_literal(index)]
    pieces.extend(f"{key}={python_literal(value)}" for key, value in command_options.items())
    command = f"CORRMAP, STUDY, ALLEEG = pop_corrmap({', '.join(pieces)})"
    return (result, study, datasets, command) if return_com else (result, study, datasets)


__all__ = ["pop_corrmap"]
