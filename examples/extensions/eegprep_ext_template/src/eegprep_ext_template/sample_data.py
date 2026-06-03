"""Packaged sample EEG data for template tests and demos."""

from __future__ import annotations

from copy import deepcopy
from importlib import resources
import json
from typing import Any

import numpy as np


def load_sample_eeg() -> dict[str, Any]:
    """Return a fresh tiny EEG dictionary from this package's sample data."""
    text = (
        resources.files("eegprep_ext_template")
        .joinpath("resources/sample_data/template_eeg.json")
        .read_text(encoding="utf-8")
    )
    payload = json.loads(text)
    eeg = deepcopy(payload)
    eeg["data"] = np.asarray(payload["data"], dtype=float)
    eeg["times"] = np.asarray(payload["times"], dtype=float)
    return eeg
