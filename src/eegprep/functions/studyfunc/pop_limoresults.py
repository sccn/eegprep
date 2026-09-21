"""Store second-level LIMO-compatible results in a STUDY."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.studyfunc._study_utils import build_python_call, ensure_study
from eegprep.functions.studyfunc.std_limoresults import std_limoresults


def pop_limoresults(
    STUDY: dict[str, Any],
    source: Any = None,
    *args: Any,
    analysis: str = "one sample t-test",
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Compute and retain one standalone LIMO second-level result."""
    study = ensure_study(STUDY)
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    analysis = str(options.pop("analysis", analysis))
    if source is None:
        source = options.pop("source", None)
    if source is None:
        stored_limo = study.get("limo")
        limo = stored_limo if isinstance(stored_limo, dict) else {}
        source = limo.get("model_files")
    if source is None:
        raise ValueError("pop_limoresults requires model files or numerical result data")
    result = std_limoresults(source, analysis, **options)
    stored_limo = study.get("limo")
    limo = deepcopy(stored_limo) if isinstance(stored_limo, dict) else {}
    summaries = list(limo.get("results") or [])
    summaries.append(result)
    limo["results"] = summaries
    study["limo"] = limo
    study["saved"] = "no"
    command = build_python_call(
        ("STUDY", "result"),
        "pop_limoresults",
        "STUDY",
        "source",
        analysis=analysis,
        **options,
    )
    return (study, result, command) if return_com else (study, result)


__all__ = ["pop_limoresults"]
