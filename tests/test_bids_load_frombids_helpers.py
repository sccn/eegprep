from __future__ import annotations

from pathlib import Path

import numpy as np


def test_raw_set_loader_returns_eeg_and_timing_metadata() -> None:
    from eegprep.plugins.EEG_BIDS.raw import load_raw_eeg_file

    dataset = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data.set"
    warnings: list[str] = []

    eeg, srate, times_sec, report = load_raw_eeg_file(
        str(dataset),
        dtype=np.float64,
        numeric_null=np.array([]),
        warning=warnings.append,
        verbose=False,
    )

    assert report["ImporterUsed"] == "pop_loadset"
    assert warnings == []
    assert srate == eeg["srate"]
    assert eeg["data"].dtype == np.float64
    np.testing.assert_allclose(times_sec, np.asarray(eeg["times"], dtype=float) / 1000.0)


def test_montage_inference_uses_packaged_montage_resources() -> None:
    from eegprep.plugins.EEG_BIDS.montage import apply_montage_inference

    numeric_null = np.array([])
    eeg = {
        "chanlocs": [
            {
                "labels": "'Fp1'",
                "X": numeric_null,
                "Y": numeric_null,
                "Z": numeric_null,
                "sph_radius": numeric_null,
                "sph_theta": numeric_null,
                "sph_phi": numeric_null,
                "theta": numeric_null,
                "radius": numeric_null,
            },
            {
                "labels": "Fpz",
                "X": numeric_null,
                "Y": numeric_null,
                "Z": numeric_null,
                "sph_radius": numeric_null,
                "sph_theta": numeric_null,
                "sph_phi": numeric_null,
                "theta": numeric_null,
                "radius": numeric_null,
            },
            {
                "labels": "Fp2",
                "X": numeric_null,
                "Y": numeric_null,
                "Z": numeric_null,
                "sph_radius": numeric_null,
                "sph_theta": numeric_null,
                "sph_phi": numeric_null,
                "theta": numeric_null,
                "radius": numeric_null,
            },
        ],
        "chaninfo": {"nosedir": "+Y"},
        "etc": {},
    }
    report: dict[str, object] = {}
    warnings: list[str] = []
    errors: list[str] = []

    apply_montage_inference(
        eeg,
        "standard-10-5-342ch.locs",
        numeric_null=numeric_null,
        report=report,
        warning=warnings.append,
        error=errors.append,
    )

    assert warnings == []
    assert errors == []
    assert eeg["chaninfo"]["nosedir"] == "+X"
    assert eeg["etc"]["labelscheme"] == "10-20"
    assert report["ChanlocsFrom"] == "standard-10-5-342ch.locs"
    for chanloc in eeg["chanlocs"]:
        assert np.isfinite([chanloc["X"], chanloc["Y"], chanloc["Z"]]).all()
