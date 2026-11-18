"""Degenerate electrode positions: error, warning, and realistic montages."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from eegprep import clean_channels, eeg_checkset, eeg_emptyset
from eegprep.functions.popfunc.eeg_interp import eeg_interp, spheric_spline
from eegprep.functions.sigprocfunc.electrode_geometry import (
    HEMISPHERE_THICKNESS_RATIO,
    validate_electrode_positions,
)
from eegprep.functions.sigprocfunc.headplot import headplot_setup
from eegprep.functions.sigprocfunc.readlocs import readlocs
from eegprep.plugins.clean_rawdata.private.sphericalSplineInterpolate import sphericalSplineInterpolate

# Positions and generative model from issue 539. The recording is shortened;
# the geometry decision does not depend on the sample count.
_ISSUE_LOCS = {
    "Fp1": (-32, 77, 19),
    "Fp2": (32, 77, 19),
    "F3": (-52, 35, 57),
    "F4": (52, 35, 57),
    "C3": (-69, -29, 69),
    "C4": (69, -29, 69),
    "P3": (-54, -92, 44),
    "P4": (54, -92, 44),
    "O1": (-31, -116, -7),
    "O2": (31, -116, -7),
    "F7": (-72, 32, 5),
    "F8": (72, 32, 5),
    "T3": (-83, -21, -4),
    "T4": (83, -21, -4),
    "T5": (-73, -74, -8),
    "T6": (73, -74, -8),
    "Fz": (0, 37, 84),
    "Cz": (0, -34, 105),
    "Pz": (0, -98, 66),
}
_TEN_20 = {
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "Oz",
    "O2",
    "Fpz",
}
_TEN_10 = _TEN_20 | {
    "AF3",
    "AF4",
    "AF7",
    "AF8",
    "FC1",
    "FC2",
    "FC5",
    "FC6",
    "CP1",
    "CP2",
    "CP5",
    "CP6",
    "PO3",
    "PO4",
    "PO7",
    "PO8",
    "F1",
    "F2",
    "C1",
    "C2",
    "P1",
    "P2",
    "F5",
    "F6",
    "P5",
    "P6",
    "C5",
    "C6",
    "FT7",
    "FT8",
    "TP7",
    "TP8",
    "AFz",
    "FCz",
    "CPz",
    "POz",
    "Iz",
}
_RESOURCES = Path(__file__).resolve().parents[1] / "src" / "eegprep" / "resources"


def _issue_positions() -> np.ndarray:
    return np.array([_ISSUE_LOCS[name] for name in _ISSUE_LOCS], dtype=float)


def _issue_recording(positions: np.ndarray) -> dict:
    """Build the issue's dataset with an 8 s recording instead of 120 s."""
    positions = np.asarray(positions, dtype=float)
    names = list(_ISSUE_LOCS)
    xyz = _issue_positions()
    rng = np.random.default_rng(0)
    srate = 250.0
    n_samples = 250 * 8
    sources = rng.standard_normal((5, n_samples))
    source_pos = rng.standard_normal((5, 3))
    source_pos = 80 * source_pos / np.linalg.norm(source_pos, axis=1, keepdims=True)
    forward = np.exp(-np.linalg.norm(xyz[:, None, :] - source_pos[None], axis=2) / 60.0)
    data = forward @ sources + 0.2 * rng.standard_normal((len(names), n_samples))
    eeg = eeg_emptyset()
    eeg.update(
        data=data,
        nbchan=len(names),
        pnts=n_samples,
        trials=1,
        srate=srate,
        xmin=0.0,
        xmax=(n_samples - 1) / srate,
        chanlocs=[
            {"labels": name, "X": float(point[0]), "Y": float(point[1]), "Z": float(point[2])}
            for name, point in zip(names, positions, strict=True)
        ],
    )
    return eeg_checkset(eeg)


def _plain_eeg(positions: np.ndarray, n_samples: int = 64) -> dict:
    positions = np.asarray(positions, dtype=float)
    n_channels = len(positions)
    return {
        "data": np.random.default_rng(1).standard_normal((n_channels, n_samples)),
        "nbchan": n_channels,
        "pnts": n_samples,
        "trials": 1,
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": (n_samples - 1) / 100.0,
        "chanlocs": [
            {"labels": f"E{index + 1}", "X": float(point[0]), "Y": float(point[1]), "Z": float(point[2])}
            for index, point in enumerate(positions)
        ],
    }


def _ring(n_channels: int, z: float = 0.0) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n_channels, endpoint=False)
    return np.column_stack((80.0 * np.cos(angles), 80.0 * np.sin(angles), np.full(n_channels, z)))


def _line(n_channels: int) -> np.ndarray:
    return np.column_stack((10.0 + 5.0 * np.arange(n_channels), np.zeros(n_channels), np.zeros(n_channels)))


def _cartesian(locs: list[dict], labels: set[str] | None = None) -> np.ndarray:
    rows = []
    for loc in locs:
        label = str(loc.get("labels", ""))
        if labels is not None and label not in labels:
            continue
        rows.append(
            [
                float(np.asarray(loc["X"], dtype=float).ravel()[0]),
                float(np.asarray(loc["Y"], dtype=float).ravel()[0]),
                float(np.asarray(loc["Z"], dtype=float).ravel()[0]),
            ]
        )
    positions = np.asarray(rows, dtype=float)
    return positions[np.isfinite(positions).all(axis=1)]


def _with_missing_channels(positions: np.ndarray) -> list[tuple[str, np.ndarray]]:
    dropped_block = positions[8:]
    dropped_stride = positions[np.arange(len(positions)) % 4 != 0]
    return [("full", positions), ("drop-first-8", dropped_block), ("drop-every-fourth", dropped_stride)]


def _standard_montages() -> list[tuple[str, np.ndarray]]:
    cap = readlocs(str(_RESOURCES / "headplot" / "Standard-10-5-Cap385.sfp"))
    montages = [
        ("10-20", _cartesian(cap, _TEN_20)),
        ("10-10", _cartesian(cap, _TEN_10)),
        ("biosemi-128", _cartesian(readlocs(str(_RESOURCES / "montages" / "biosemi-128.locs")))),
        ("biosemi-256", _cartesian(readlocs(str(_RESOURCES / "montages" / "biosemi-sccn-256v1.locs")))),
        ("egi-129", _cartesian(readlocs(str(_RESOURCES / "montages" / "egi-gsn-hydrocell-129.locs")))),
        ("egi-257", _cartesian(readlocs(str(_RESOURCES / "montages" / "egi-gsn-hydrocell-257.locs")))),
    ]
    counts = {"10-20": 19, "10-10": 40, "biosemi-128": 128, "biosemi-256": 256, "egi-129": 128, "egi-257": 256}
    for name, positions in montages:
        if len(positions) < counts[name]:
            raise AssertionError(f"{name} loaded {len(positions)} positions")
    return montages


def _hemisphere_thickness_ratio() -> float:
    """σ3/σ1 of a uniform hemispherical shell, from its second moments."""
    cosine = np.cos(np.pi / 2.0)
    second_z = (1.0 + cosine + cosine**2) / 3.0
    mean_z = (1.0 + cosine) / 2.0
    variance_z = second_z - mean_z**2
    variance_x = 0.5 * (1.0 - second_z)
    return float(np.sqrt(variance_z / variance_x))


def test_hemisphere_thickness_ratio_is_one_half():
    """A uniform hemispherical shell has third-to-first singular-value ratio 1/2."""
    assert _hemisphere_thickness_ratio() == pytest.approx(0.5, abs=1e-12)
    assert HEMISPHERE_THICKNESS_RATIO == pytest.approx(_hemisphere_thickness_ratio())


def test_issue_539_reproduction_raises_for_collinear_and_identical_positions():
    true = _issue_positions()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        kept = clean_channels(_issue_recording(true))
    assert kept["nbchan"] >= 1
    assert kept["data"].shape[0] == kept["nbchan"]

    collinear = [(10 + 5 * index, 0, 0) for index in range(len(true))]
    with pytest.raises(ValueError, match="collinear"):
        clean_channels(_issue_recording(collinear))
    with pytest.raises(ValueError, match="identical"):
        clean_channels(_issue_recording([(10, 20, 30)] * len(true)))


def test_nearly_collinear_positions_raise():
    noisy_line = _line(len(_ISSUE_LOCS)) + np.random.default_rng(2).normal(0.0, 1e-6, size=(len(_ISSUE_LOCS), 3))
    with pytest.raises(ValueError, match="collinear"):
        clean_channels(_issue_recording(noisy_line))


def test_planar_ring_warns_and_clean_channels_still_runs():
    with pytest.warns(UserWarning, match="planar"):
        cleaned = clean_channels(
            _issue_recording(_ring(len(_ISSUE_LOCS))),
            num_samples=4,
            corr_threshold=-1.0,
            noise_threshold=100.0,
        )
    assert cleaned["data"].shape[0] == cleaned["nbchan"]
    assert cleaned["nbchan"] >= 1


def test_nearly_planar_positions_warn_and_clean_channels_still_runs():
    flat = _issue_positions()
    flat[:, 2] = 0.0
    jittered = flat + np.random.default_rng(3).normal(0.0, 0.1, size=flat.shape)
    with pytest.warns(UserWarning, match="planar"):
        cleaned = clean_channels(
            _issue_recording(jittered),
            num_samples=4,
            corr_threshold=-1.0,
            noise_threshold=100.0,
        )
    assert cleaned["nbchan"] >= 1


def test_eeg_interp_rejects_a_collinear_source_strip_when_bad_channels_add_spread():
    positions = _line(7)
    positions[5] = (10.0, 40.0, 0.0)
    positions[6] = (10.0, 0.0, 40.0)
    with pytest.raises(ValueError, match="collinear"):
        eeg_interp(_plain_eeg(positions), [6, 7])

    identical_sources = np.tile([10.0, 20.0, 30.0], (6, 1))
    identical_sources[4] = (0.0, 0.0, 80.0)
    identical_sources[5] = (0.0, 80.0, 0.0)
    with pytest.raises(ValueError, match="identical"):
        eeg_interp(_plain_eeg(identical_sources), [5, 6])

    flat = _line(6)
    flat[5, 1] = 50.0
    with pytest.raises(ValueError, match="collinear"):
        eeg_interp(_plain_eeg(flat), [6], method="invdist")


def test_eeg_interp_accepts_two_source_channels_from_a_nonplanar_montage():
    tetrahedron = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        interpolated = eeg_interp(_plain_eeg(tetrahedron), [3, 4])
    assert interpolated["data"].shape[0] == 4


def test_eeg_interp_rejects_identical_and_collinear_and_warns_on_a_ring():
    n_channels = 12
    with pytest.raises(ValueError, match="identical"):
        eeg_interp(_plain_eeg(np.tile([10.0, 20.0, 30.0], (n_channels, 1))), [1])
    with pytest.raises(ValueError, match="collinear"):
        eeg_interp(_plain_eeg(_line(n_channels)), [1])
    with pytest.warns(UserWarning, match="planar"):
        interpolated = eeg_interp(_plain_eeg(_ring(n_channels)), [1])
    assert interpolated["data"].shape == (n_channels, 64)
    assert np.isfinite(interpolated["data"]).all()


def test_invdist_accepts_a_planar_ring_and_rejects_a_line():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        interpolated = eeg_interp(_plain_eeg(_ring(12)), [1], method="invdist")
    assert interpolated["data"].shape[0] == 12
    with pytest.raises(ValueError, match="collinear"):
        eeg_interp(_plain_eeg(_line(12)), [1], method="invdist")


def test_spline_helpers_reject_a_line_and_warn_on_a_ring():
    ring = _ring(10)
    values = np.ones((8, 3))
    with pytest.warns(UserWarning, match="planar"):
        interpolated = spheric_spline(
            ring[:8, 0],
            ring[:8, 1],
            ring[:8, 2],
            ring[8:, 0],
            ring[8:, 1],
            ring[8:, 2],
            values,
            (0, 4, 7),
        )
    assert interpolated.shape == (2, 3)
    with pytest.raises(ValueError, match="collinear"):
        spheric_spline(
            np.arange(8, dtype=float),
            np.zeros(8),
            np.zeros(8),
            np.array([9.0]),
            np.zeros(1),
            np.zeros(1),
            np.ones((8, 3)),
            (0, 4, 7),
        )

    source = ring.T
    with pytest.warns(UserWarning, match="planar"):
        weights, _, _, _ = sphericalSplineInterpolate(source[:, :8], source)
    assert weights.shape == (10, 8)
    with pytest.raises(ValueError, match="collinear"):
        sphericalSplineInterpolate(np.vstack((np.arange(6, dtype=float), np.zeros(6), np.zeros(6))), np.zeros((3, 2)))


def test_headplot_rejects_a_line_and_warns_on_a_ring(tmp_path: Path):
    def locs(positions: np.ndarray) -> list[dict]:
        return [
            {"labels": f"E{index + 1}", "X": float(point[0]), "Y": float(point[1]), "Z": float(point[2])}
            for index, point in enumerate(positions)
        ]

    with pytest.raises(ValueError, match="collinear"):
        headplot_setup(locs(_line(12)), tmp_path / "line.spl")
    with pytest.warns(UserWarning, match="planar"):
        spline = headplot_setup(locs(_ring(12, z=0.5)), tmp_path / "ring.spl")
    assert Path(spline).is_file()


def test_standard_montages_with_missing_channels_stay_silent():
    for name, positions in _standard_montages():
        for subset_name, subset in _with_missing_channels(positions):
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                validate_electrode_positions(subset, operation=f"{name}:{subset_name}")

    ten_twenty = next(positions for name, positions in _standard_montages() if name == "10-20")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        interpolated = eeg_interp(_plain_eeg(ten_twenty, n_samples=16), [1])
        cleaned = clean_channels(
            _plain_eeg(ten_twenty, n_samples=800),
            num_samples=4,
            corr_threshold=-1.0,
            noise_threshold=100.0,
            window_len=2,
        )
    assert interpolated["nbchan"] == len(ten_twenty)
    assert cleaned["nbchan"] == len(ten_twenty)
