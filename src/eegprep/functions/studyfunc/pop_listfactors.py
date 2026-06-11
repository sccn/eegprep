"""List STUDY design factors."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from eegprep.functions.popfunc._plot_utils import python_literal
from eegprep.functions.popfunc._pop_utils import is_on, parse_key_value_args
from eegprep.functions.studyfunc._study_utils import build_python_call, ensure_study
from eegprep.functions.studyfunc.std_addvarlevel import std_addvarlevel


def pop_listfactors(
    des: dict[str, Any], *args: Any, return_com: bool = False, **kwargs: Any
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], str]:
    """Return factor descriptors for one STUDY or design structure."""
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    gui = options.pop("gui", "off")
    level = str(options.pop("level", "both")).lower()
    vartype = str(options.pop("vartype", "both")).lower()
    constant = options.pop("constant", "on")
    options.pop("splitreg", "off")
    options.pop("contrast", "off")
    options.pop("interaction", "off")
    if options:
        raise ValueError(f"Unknown pop_listfactors option(s): {', '.join(sorted(options))}")
    if is_on(gui):
        raise NotImplementedError("pop_listfactors GUI display is not available in standalone EEGPrep")
    if level not in {"one", "two", "both"}:
        raise ValueError("level must be 'one', 'two', or 'both'")
    if vartype not in {"categorical", "continuous", "both"}:
        raise ValueError("vartype must be 'categorical', 'continuous', or 'both'")

    designs = _designs(des)
    factors = []
    for design in designs:
        for variable in design.get("variable") or []:
            if not isinstance(variable, dict):
                continue
            current_level = str(variable.get("level") or "one")
            current_vartype = str(variable.get("vartype") or "categorical").lower()
            if level != "both" and current_level != level:
                continue
            if vartype != "both" and current_vartype != vartype:
                continue
            factors.extend(_factor_entries(variable, current_level, current_vartype))
    if is_on(constant):
        factors.append(
            {
                "description": "constant",
                "label": "constant",
                "vartype": "continuous",
                "level": "two",
            }
        )
    output = _deduplicate(factors)
    if return_com:
        return output, _history_command(des, level=level, vartype=vartype, constant=constant)
    return output


def _history_command(des: dict[str, Any], **kwargs: Any) -> str:
    source = "STUDY" if "design" in des else python_literal(des)
    return build_python_call(("factors",), "pop_listfactors", source, **kwargs)


def _designs(des: dict[str, Any]) -> list[dict[str, Any]]:
    if "design" in des:
        study = std_addvarlevel(ensure_study(des))
        return [deepcopy(design) for design in study.get("design") or [] if isinstance(design, dict)]
    if "variable" in des:
        return [deepcopy(des)]
    return []


def _factor_entries(variable: dict[str, Any], level: str, vartype: str) -> list[dict[str, Any]]:
    label = str(variable.get("label") or "")
    if vartype == "continuous":
        return [{"description": f"{label} - continuous variable", "label": label, "vartype": vartype, "level": level}]
    values = variable.get("value") if isinstance(variable.get("value"), list) else []
    return [
        {
            "description": f"{label} - {_value_label(value)}",
            "label": label,
            "vartype": "categorical",
            "level": level,
            "value": value,
        }
        for value in values
    ]


def _value_label(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "&".join(str(item) for item in value)
    return str(value)


def _deduplicate(factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for factor in factors:
        key = (
            factor.get("label"),
            factor.get("vartype"),
            factor.get("level"),
            repr(factor.get("value")),
            factor.get("description"),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(factor)
    return output


__all__ = ["pop_listfactors"]
