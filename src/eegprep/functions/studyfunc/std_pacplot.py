"""Plot cached STUDY PAC measures."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.popfunc._pop_utils import is_on, parse_key_value_args
from eegprep.functions.studyfunc._study_utils import build_python_call, ensure_study
from eegprep.functions.studyfunc.std_readdata import std_readpac


def std_pacplot(
    STUDY: dict[str, Any],
    ALLEEG: list[dict[str, Any]] | None,
    *args: Any,
    channels: Any = None,
    channels1: Any = None,
    clusters: Any = None,
    components: Any = None,
    design: int | None = None,
    noplot: str | bool = "off",
    plotmode: str = "normal",
    return_com: bool = False,
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Read and optionally plot precomputed STUDY PAC measures."""
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    option_channels = options.pop("channels", None)
    option_channels1 = options.pop("channels1", None)
    if option_channels is not None:
        channels = option_channels
    elif option_channels1 is not None:
        channels = option_channels1
    elif channels is None:
        channels = channels1
    option_clusters = options.pop("clusters", None)
    option_clusters1 = options.pop("clusters1", None)
    if option_clusters is not None:
        clusters = option_clusters
    elif option_clusters1 is not None:
        clusters = option_clusters1
    option_components = options.pop("components", None)
    option_comps = options.pop("comps", None)
    if option_components is not None:
        components = option_components
    elif option_comps is not None:
        components = option_comps
    design = options.pop("design", design)
    noplot = options.pop("noplot", noplot)
    plotmode = str(options.pop("plotmode", plotmode) or "normal").lower()
    timerange = options.pop("timerange", None)
    freqrange = options.pop("freqrange", None)
    ignored = {"condition", "channels2", "clusters2", "onepersubj", "plotsubjects", "topoplotopt", "mode"}
    unsupported = sorted(key for key in options if key not in ignored)
    if unsupported:
        raise ValueError(f"Unknown std_pacplot option(s): {', '.join(unsupported)}")
    channels, clusters = _default_target(STUDY, channels, clusters, components)
    study, pacdata, pactimes, pacfreqs = std_readpac(
        STUDY,
        ALLEEG,
        channels=channels,
        clusters=clusters,
        components=components,
        design=design,
        timerange=timerange,
        freqrange=freqrange,
    )
    figure = None if is_on(noplot) or plotmode == "none" else _plot_pac(pacdata, pactimes, pacfreqs)
    command = build_python_call(
        ("STUDY", "PACDATA", "PACTIMES", "PACFREQS", "FIGURE"),
        "std_pacplot",
        "STUDY",
        "ALLEEG",
        channels=channels,
        clusters=clusters,
        components=components,
        design=design,
        noplot=noplot,
        plotmode=plotmode,
        timerange=timerange,
        freqrange=freqrange,
    )
    result = (study, pacdata, pactimes, pacfreqs, figure)
    return (*result, command) if return_com else result


def _default_target(study: dict[str, Any], channels: Any, clusters: Any, components: Any) -> tuple[Any, Any]:
    if channels is not None or clusters is not None or components is not None:
        return channels, clusters
    prepared = ensure_study(study)
    if any(isinstance(group, dict) and "pacdata" in group for group in prepared.get("changrp") or []):
        return "channels", clusters
    return channels, 1


def _plot_pac(data: list[np.ndarray], times: np.ndarray, freqs: np.ndarray) -> Any:
    images = []
    for values in data:
        array = np.abs(np.asarray(values, dtype=float))
        if array.ndim < 2:
            raise ValueError("PAC plot data must have frequency and time axes")
        while array.ndim > 2:
            array = np.nanmean(array, axis=0)
        images.append(array)
    image = np.nanmean(np.stack(images, axis=0), axis=0)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    mesh = ax.imshow(
        image,
        aspect="auto",
        origin="lower",
        extent=[float(times[0]), float(times[-1]), float(freqs[0]), float(freqs[-1])],
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude frequency (Hz)")
    ax.set_title("STUDY PAC")
    fig.colorbar(mesh, ax=ax, label="PAC magnitude")
    fig.tight_layout()
    return fig


__all__ = ["std_pacplot"]
