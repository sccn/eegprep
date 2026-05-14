"""Create a STUDY by browsing for dataset files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.studyfunc.pop_study import pop_study


def pop_studywizard(filenames: list[str] | tuple[str, ...]) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Load selected datasets and create a STUDY."""
    datasets = [pop_loadset(str(Path(filename))) for filename in filenames]
    study, alleeg, _command = pop_study(None, datasets, name="EEGPrep study")
    return study, alleeg, "STUDY = pop_studywizard();"
