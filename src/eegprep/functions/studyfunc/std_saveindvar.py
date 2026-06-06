"""Store STUDY independent-variable descriptors."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from eegprep.functions.studyfunc.pop_listfactors import pop_listfactors
from eegprep.functions.studyfunc._study_utils import build_python_call, ensure_study


def std_saveindvar(STUDY: dict[str, Any], *, return_com: bool = False) -> Any:
    """Save factor descriptors under ``STUDY.etc.eegprep.independent_variables``."""
    study = deepcopy(ensure_study(STUDY))
    study["etc"].setdefault("eegprep", {})["independent_variables"] = pop_listfactors(study, constant="off")
    study["saved"] = "no"
    command = build_python_call(("STUDY",), "std_saveindvar", "STUDY")
    return (study, command) if return_com else study


__all__ = ["std_saveindvar"]
