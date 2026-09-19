"""Shared storage and cache invalidation for STUDY plotting parameters."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.popfunc.plot_utils import python_literal
from eegprep.functions.studyfunc._study_utils import ensure_study, equal_value


def update_study_params(
    STUDY: dict[str, Any],
    section: str,
    defaults: dict[str, Any],
    function_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    invalidate_on: tuple[str, ...] = (),
    invalidated_fields: tuple[str, ...] = (),
    return_com: bool = False,
) -> Any:
    """Fill defaults, apply known options, and invalidate affected caches."""
    study = ensure_study(STUDY)
    params = study["etc"].get(section)
    if not isinstance(params, dict):
        params = {}
    for key, value in defaults.items():
        params.setdefault(key, deepcopy(value))
    study["etc"][section] = params

    options = _options(args, kwargs)
    unknown = sorted(set(options) - set(defaults))
    if unknown:
        raise ValueError(f"Unknown {function_name} option(s): {', '.join(unknown)}")
    changed = {key for key, value in options.items() if not equal_value(params.get(key), value)}
    for key, value in options.items():
        params[key] = deepcopy(value)
    if changed.intersection(invalidate_on):
        _clear_measure_fields(study, invalidated_fields)
    if changed:
        study["saved"] = "no"

    command = _history_command(function_name, options)
    return (study, command) if return_com else study


def _options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if len(args) == 1 and str(args[0]).lower() == "default":
        if kwargs:
            raise ValueError("'default' cannot be combined with parameter options")
        return {}
    return parse_key_value_args(args, kwargs, lowercase_kwargs=True)


def _clear_measure_fields(study: dict[str, Any], fields: tuple[str, ...]) -> None:
    for collection_name in ("cluster", "changrp"):
        collection = study.get(collection_name) or []
        if isinstance(collection, dict):
            collection = [collection]
        for entry in collection:
            if not isinstance(entry, dict):
                continue
            for field in fields:
                entry.pop(field, None)


def _history_command(function_name: str, options: dict[str, Any]) -> str:
    pieces = ["STUDY"]
    pieces.extend(f"{key}={python_literal(value)}" for key, value in options.items())
    return f"STUDY = {function_name}({', '.join(pieces)})"


__all__ = ["update_study_params"]
