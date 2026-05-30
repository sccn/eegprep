"""Create a STUDY by browsing for dataset files."""

from __future__ import annotations

from typing import Any

from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.studyfunc._study_utils import build_python_call
from eegprep.functions.studyfunc.pop_study import pop_study


def pop_studywizard(
    filenames: list[str] | tuple[str, ...],
    *,
    name: str = "EEGPrep study",
    return_com: bool = False,
) -> Any:
    """Load selected datasets and create a STUDY."""
    datasets = [pop_loadset(str(filename)) for filename in filenames]
    study, alleeg, _command = pop_study(None, datasets, name=name, return_com=True)
    command = build_python_call(("STUDY", "ALLEEG"), "pop_studywizard", filenames=list(map(str, filenames)), name=name)
    return (study, alleeg, command) if return_com else (study, alleeg)
