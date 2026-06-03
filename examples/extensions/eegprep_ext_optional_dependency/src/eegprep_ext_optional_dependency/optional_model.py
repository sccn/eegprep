"""Optional dependency and packaged model-data example."""

from __future__ import annotations

from importlib import resources
from typing import Any


def model_card_text() -> str:
    """Return the packaged model-card resource text."""
    return (
        resources.files("eegprep_ext_optional_dependency")
        .joinpath("resources/model/model-card.json")
        .read_text(encoding="utf-8")
    )


def pop_demo_optional_score(EEG: dict[str, Any] | None, *, return_com: bool = False):
    """Use an optional model dependency only when this action is called."""
    if EEG is None:
        return (None, "") if return_com else None
    try:
        import eegprep_template_optional_model  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install eegprep-ext-optional-dependency[model] to use pop_demo_optional_score.") from exc
    command = "EEG = pop_demo_optional_score(EEG);"
    return (EEG, command) if return_com else EEG
