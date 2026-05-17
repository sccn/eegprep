"""Create a simple ERP STUDY."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc.pop_study import pop_study


def pop_studyerp(ALLEEG: list[dict[str, Any]] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Create a STUDY marked as a simple ERP design."""
    study, datasets, _command = pop_study(None, ALLEEG or [], name="Simple ERP STUDY", design="ERP")
    return study, datasets, "STUDY = pop_studyerp();"
