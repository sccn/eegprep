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

from eegprep.plugins.dipfit._bem_parity import bem_parity_payload


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
