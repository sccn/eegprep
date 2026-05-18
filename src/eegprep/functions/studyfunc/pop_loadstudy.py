"""Load EEGPrep STUDY structures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eegprep.functions.popfunc._pop_utils import format_history_value


def pop_loadstudy(filename: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Load a STUDY JSON file saved by ``pop_savestudy``."""
    path = Path(filename)
    with path.open(encoding="utf-8") as stream:
        study = json.load(stream)
    study["filename"] = path.name
    study["filepath"] = str(path.parent)
    return study, [], f"STUDY = pop_loadstudy({format_history_value(path)});"
