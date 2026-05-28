"""Component envelope plotting helper."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.sigprocfunc.topoplot import topoplot


def envtopo(
    data: Any,
    weights: Any,
    *,
    times: Any = None,
    chanlocs: Any = None,
    icawinv: Any = None,
    components: Any = None,
    title: str = "",
):
    """Plot data envelope and largest component projection envelopes."""
    values = np.asarray(data, dtype=float)
    if values.ndim == 3:
        values = np.nanmean(values, axis=2)
    if values.ndim != 2:
        raise ValueError("envtopo data must be channels x points")
    weight_values = np.asarray(weights, dtype=float)
    activations = weight_values @ values
    maps = (
        np.asarray(icawinv, dtype=float)
        if icawinv is not None and np.asarray(icawinv).size
        else np.linalg.pinv(weight_values)
    )
    component_indices = _components(components, activations, max_components=min(7, activations.shape[0]))
    projections = [np.outer(maps[:, index], activations[index]) for index in component_indices]
    x_values = (
        np.asarray(times, dtype=float).ravel()
        if times is not None and len(np.asarray(times).ravel())
        else np.arange(values.shape[1])
    )

    fig = plt.figure(figsize=(8.5, 4.8))
    envelope_ax = fig.add_subplot(2, 1, 1)
    envelope_ax.fill_between(x_values, np.nanmin(values, axis=0), np.nanmax(values, axis=0), color="0.85", label="data")
    for index, projection in zip(component_indices, projections):
        envelope_ax.plot(x_values, np.nanmax(projection, axis=0), linewidth=1.0, label=f"IC {index + 1}")
        envelope_ax.plot(x_values, np.nanmin(projection, axis=0), linewidth=1.0)
    envelope_ax.set_xlabel("Time (ms)")
    envelope_ax.set_ylabel("uV")
    envelope_ax.set_title(title or "Largest ERP components")
    envelope_ax.legend(fontsize=7, ncols=2)

    for plot_index, component_index in enumerate(component_indices, start=1):
        ax = fig.add_subplot(2, max(len(component_indices), 1), max(len(component_indices), 1) + plot_index)
        topoplot(maps[:, component_index], chanlocs_as_list(chanlocs), axes=ax, electrodes="off")
        ax.set_title(f"IC {component_index + 1}")
    fig.tight_layout()
    return fig


def _components(components: Any, activations: np.ndarray, *, max_components: int) -> np.ndarray:
    if components is not None and len(np.asarray(components).ravel()):
        values = np.asarray(components, dtype=int).ravel()
        if np.any(values < 1) or np.any(values > activations.shape[0]):
            raise ValueError("component indices are outside available ICA components")
        return values - 1
    power = np.nanmax(activations * activations, axis=1)
    return np.argsort(power)[::-1][:max_components]


__all__ = ["envtopo"]
