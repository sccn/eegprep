"""Write the EEGPrep side of the MNI BEM dipole-fit comparison.

Run from the repository root:

    uv run python scripts/dipfit_standard_bem_parity.py --output /tmp/eegprep_bem_parity.json

``scripts/dipfit_standard_bem_parity.m`` runs this and prints the leadfield
error and component distances against FieldTrip and EEGLAB.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eegprep import pop_loadset
from eegprep.plugins.dipfit._fitting import prepare_forward_data
from eegprep.plugins.dipfit.fieldtrip.ft_compute_leadfield import ft_compute_leadfield
from eegprep.plugins.dipfit.pop_dipfit_settings import pop_dipfit_settings
from eegprep.plugins.dipfit.pop_multifit import pop_multifit


PARITY_DIPOLES_MM = np.asarray([[0.0, -20.0, 40.0], [30.0, -50.0, 30.0], [-25.0, 10.0, 45.0]], dtype=float)
PARITY_COMPONENTS = (1, 2, 3)


def bem_parity_payload(dataset: str | Path) -> dict[str, object]:
    """Return leadfields and ``pop_multifit`` positions for the default MNI model."""
    eeg = pop_loadset(str(dataset))
    eeg = pop_dipfit_settings(eeg, model="standardBEM")
    forward = prepare_forward_data(eeg, list(PARITY_COMPONENTS))
    if forward.headmodel is None or forward.sens is None:
        raise RuntimeError("standardBEM did not prepare a boundary-element head model")
    leadfield = ft_compute_leadfield(PARITY_DIPOLES_MM, forward.sens, forward.headmodel)
    fitted = pop_multifit(eeg, list(PARITY_COMPONENTS), threshold=100)
    positions = []
    for component in PARITY_COMPONENTS:
        raw = np.asarray(fitted["dipfit"]["model"][component - 1].get("posxyz", []), dtype=float)
        if raw.size < 3:
            positions.append([float("nan"), float("nan"), float("nan")])
        else:
            positions.append(raw.reshape(-1, 3)[0].tolist())
    return {
        "labels": forward.labels,
        "dipoles": PARITY_DIPOLES_MM.tolist(),
        "leadfield": np.asarray(leadfield, dtype=float).tolist(),
        "positions": positions,
        "coord_transform": [float(value) for value in eeg["dipfit"]["coord_transform"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    dataset = args.dataset
    if dataset is None:
        dataset = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data_epochs_ica.set"
    text = json.dumps(bem_parity_payload(dataset))
    if args.output is None:
        print(text)
        return
    args.output.write_text(text)


if __name__ == "__main__":
    main()
