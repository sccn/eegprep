"""Reject dipole models by their relative residual variance."""

import numpy as np

from eegprep.plugins.dipfit._utils import normalize_model_list


def dipfit_reject(models, reject: float) -> list[dict]:
    """Empty fits exceeding an RV fraction (or a percentage greater than one)."""
    threshold = float(reject)
    if threshold > 1:
        threshold /= 100
    if not 0 <= threshold <= 1:
        raise ValueError("reject must be a fraction in 0..1 or a percentage in 0..100")
    result = []
    for model in normalize_model_list({"dipfit": {"model": models}}):
        rv = np.asarray(model.get("rv", np.nan), dtype=float).ravel()
        if rv.size == 1 and np.isfinite(rv[0]) and rv[0] <= threshold:
            result.append(dict(model))
        else:
            rejected = {key: [] for key in model}
            rejected.update(rv=1.0)
            if "component" in model:
                rejected["component"] = model["component"]
            result.append(rejected)
    return result
