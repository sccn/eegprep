"""Check STUDY dataset/session metadata consistency."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._study_utils import as_alleeg_list, check_datasetinfo, ensure_study


def std_checkdatasession(
    STUDY: dict[str, Any], ALLEEG: list[dict[str, Any]] | None = None, *, return_report: bool = False
) -> Any:
    """Check dataset/session alignment for a STUDY and loaded ALLEEG."""
    study = ensure_study(STUDY)
    datasets = as_alleeg_list(ALLEEG)
    report = check_datasetinfo(study, datasets)
    sessions = []
    seen = set()
    duplicates = []
    for info in study.get("datasetinfo") or []:
        key = (info.get("subject"), repr(info.get("session")), repr(info.get("run")), info.get("condition"))
        sessions.append(key)
        if key in seen:
            duplicates.append(
                {
                    "subject": info.get("subject"),
                    "session": info.get("session"),
                    "run": info.get("run"),
                    "condition": info.get("condition"),
                }
            )
        seen.add(key)
    report["duplicate_subject_sessions"] = duplicates
    if duplicates:
        report["problems"].append("duplicate subject/session/run/condition entries")
    ok = not report["problems"]
    return (ok, report) if return_report else int(ok)


__all__ = ["std_checkdatasession"]
