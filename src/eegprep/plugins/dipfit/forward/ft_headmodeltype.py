"""Head-model type detection for the DIPFIT forward dispatch."""

from __future__ import annotations

from typing import Any

import numpy as np


SPHERICAL_HEADMODELS = ("singlesphere", "concentricspheres")
BEM_HEADMODELS = ("bem", "dipoli", "asa", "bemcp", "openmeeg")

_TYPE_ALIASES = {
    "nolte": "singleshell",
    "concentric": "concentricspheres",
}


def ft_headmodeltype(headmodel: dict[str, Any]) -> str:
    """Return the FieldTrip volume-conductor type of ``headmodel``."""
    declared = headmodel.get("type")
    if isinstance(declared, str) and declared.strip():
        kind = declared.strip()
        return _TYPE_ALIASES.get(kind, kind)
    radii = np.asarray(headmodel.get("r", []), dtype=float).ravel()
    if radii.size == 1 and "label" not in headmodel:
        return "singlesphere"
    if radii.size >= 2 and "label" not in headmodel:
        return "concentricspheres"
    if "bnd" in headmodel and "mat" in headmodel:
        return "bem"
    if "bnd" in headmodel:
        return "singleshell"
    raise ValueError("unrecognized volume conductor model")


def normalize_headmodel(headmodel: dict[str, Any]) -> dict[str, Any]:
    """Return radii and conductivities as doubles, with FieldTrip's ``cond`` name."""
    out = dict(headmodel)
    if "cond" not in out and "c" in out:
        out["cond"] = out["c"]
    if "r" not in out or "cond" not in out:
        raise ValueError("spherical head models need radii and conductivities")
    out["r"] = np.asarray(out["r"], dtype=float).ravel()
    out["cond"] = np.asarray(out["cond"], dtype=float).ravel()
    if out["r"].size != out["cond"].size:
        raise ValueError("the number of radii and conductivities must match")
    if "o" in out and out["o"] is not None and np.size(out["o"]):
        out["o"] = np.asarray(out["o"], dtype=float).ravel()
    else:
        out["o"] = np.zeros(3)
    if not isinstance(out.get("type"), str) or not str(out.get("type")).strip():
        out["type"] = ft_headmodeltype(out)
    return out
