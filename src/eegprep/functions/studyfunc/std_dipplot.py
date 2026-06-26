"""2D visualization for STUDY-level dipole plotting."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.popfunc._pop_utils import is_on, parse_key_value_args
from eegprep.functions.studyfunc._cluster_utils import dataset_for_study_set
from eegprep.functions.studyfunc._study_utils import build_python_call
from eegprep.functions.studyfunc.std_preclust import _dipole_position


def std_dipplot(
    STUDY: dict[str, Any],
    ALLEEG: list[dict[str, Any]],
    *args: Any,
    clusters: Any = None,
    noplot: str | bool = "off",
    return_com: bool = False,
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Plot 2D visualization for group-level dipole clusters."""
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    clusters = options.pop("clusters", clusters)
    noplot = options.pop("noplot", noplot)

    # EEGLAB defaults to parent cluster (1) if no cluster is specified
    if clusters is None:
        clusters = [1]

    if not isinstance(clusters, (list, tuple, np.ndarray)):
        clusters = [clusters]

    fig = None
    if not is_on(noplot):
        fig, ax = plt.subplots(figsize=(6, 6))

        for clus in clusters:
            clus_idx = int(clus) - 1
            if clus_idx < 0 or clus_idx >= len(STUDY.get("cluster", [])):
                continue

            cluster = STUDY["cluster"][clus_idx]
            sets = np.asarray(cluster.get("sets") or []).ravel()
            comps = np.asarray(cluster.get("comps") or []).ravel()

            positions = []
            for study_set, comp in zip(sets, comps):
                try:
                    eeg = dataset_for_study_set(STUDY, ALLEEG, int(study_set))
                    pos = _dipole_position(eeg, int(comp), int(study_set))
                    positions.append(pos)
                except ValueError:
                    continue

            if positions:
                positions = np.asarray(positions)
                ax.scatter(positions[:, 0], positions[:, 1], label=cluster.get("name", f"Cluster {clus}"))

        ax.set_title("2D Group-Level Dipole Clusters")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

    command = build_python_call(("STUDY",), "std_dipplot", "STUDY", "ALLEEG", clusters=clusters)
    return (STUDY, fig, command) if return_com else (STUDY, fig)


__all__ = ["std_dipplot"]
