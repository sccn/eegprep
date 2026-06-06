"""Check simple STUDY dataset uniformity by subject."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._study_utils import ensure_study


def std_checkconsist(
    STUDY: dict[str, Any],
    *args: Any,
    uniform: str | None = None,
    return_counts: bool = False,
    **kwargs: Any,
) -> Any:
    """Check whether each value of a STUDY factor has the same subject count."""
    study = ensure_study(STUDY)
    label = str(kwargs.get("uniform", uniform or _uniform_from_args(args)) or "").lower()
    if label not in {"condition", "group", "session"}:
        raise ValueError("uniform must be 'condition', 'group', or 'session'")
    values = [str(value) for value in study.get(label) or [] if str(value)]
    if not values:
        return (1, []) if return_counts else 1
    counts = []
    infos = [info for info in study.get("datasetinfo") or [] if isinstance(info, dict)]
    for value in values:
        subjects = {
            str(info.get("subject") or "")
            for info in infos
            if str(info.get(label) if label != "session" else info.get("session")) == value
        }
        counts.append(len(subjects))
    ok = int(len(set(counts)) <= 1)
    return (ok, counts) if return_counts else ok


def _uniform_from_args(args: tuple[Any, ...]) -> str:
    if len(args) >= 2 and str(args[0]).lower() == "uniform":
        return str(args[1])
    if len(args) == 1:
        return str(args[0])
    return ""


__all__ = ["std_checkconsist"]
