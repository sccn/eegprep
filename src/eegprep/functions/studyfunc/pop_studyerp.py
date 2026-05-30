"""Create a simple ERP STUDY."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._study_utils import build_python_call
from eegprep.functions.studyfunc.pop_study import pop_study


def pop_studyerp(
    ALLEEG: list[dict[str, Any]] | None = None,
    *,
    return_com: bool = False,
) -> Any:
    """Create a STUDY marked as a simple ERP design."""
    study, datasets, _command = pop_study(None, ALLEEG or [], name="Simple ERP STUDY", design="ERP", return_com=True)
    command = build_python_call(("STUDY", "ALLEEG"), "pop_studyerp", "ALLEEG")
    return (study, datasets, command) if return_com else (study, datasets)
