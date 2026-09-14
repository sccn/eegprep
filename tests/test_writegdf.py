from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pytest

import eegprep
from eegprep.functions.popfunc.pop_fileio import pop_fileio
from eegprep.functions.popfunc.pop_writeeeg import pop_writeeeg
from eegprep.functions.sigprocfunc.writegdf import writegdf


def _eeg() -> dict:
    samples = 257
    time = np.arange(samples) / 256
    data = np.vstack(
        (
            20 * np.sin(2 * np.pi * 10 * time) + 0.125,
            7 * np.cos(2 * np.pi * 3 * time) - 2.75,
        )
    )
    return {
        "data": data,
        "nbchan": 2,
        "pnts": samples,
        "trials": 1,
        "srate": 256.0,
        "setname": "GDF round trip",
        "subject": "subject-01",
        "chanlocs": [{"labels": "Fz"}, {"labels": "Cz"}],
        "event": [
            {"type": 10, "latency": 1.0, "duration": 4.0},
            {"type": "20", "latency": 64.25, "duration": 2.0, "channel": 2},
        ],
    }


def _write(eeg: dict, output: Path) -> dict[str, int]:
    return writegdf(
        output,
        eeg["data"],
        eeg["srate"],
        labels=[channel["labels"] for channel in eeg["chanlocs"]],
        events=eeg["event"],
        subject=eeg["subject"],
        recording=eeg["setname"],
    )


def test_writegdf_roundtrips_samples_rate_labels_and_events_through_mne(tmp_path: Path) -> None:
    eeg = _eeg()
    output = tmp_path / "roundtrip.gdf"

    mapping = _write(eeg, output)
    raw = mne.io.read_raw_gdf(output, preload=True, verbose=False)
    imported = pop_fileio(output)

    assert mapping == {}
    assert output.read_bytes().startswith(b"GDF 1.25")
    assert raw.ch_names == ["Fz", "Cz"]
    assert raw.info["sfreq"] == pytest.approx(eeg["srate"], rel=1e-12)
    np.testing.assert_allclose(raw.get_data(units="uV"), eeg["data"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(imported["data"], eeg["data"], rtol=0, atol=1e-12)
    assert [event["type"] for event in imported["event"]] == ["10", "20"]
    # MNE stores annotation seconds at microsecond resolution, which is below
    # one thousandth of a sample here.
    np.testing.assert_allclose([event["latency"] for event in imported["event"]], [1, 64], rtol=0, atol=1e-3)
    np.testing.assert_allclose([event["duration"] for event in imported["event"]], [4, 2], rtol=0, atol=1e-3)


def test_pop_writeeeg_gdf_history_and_free_text_event_mapping(tmp_path: Path) -> None:
    eeg = _eeg()
    eeg["event"] = [
        {"type": "stimulus", "latency": 20},
        {"type": "stimulus", "latency": 40},
        {"type": "response", "latency": 60},
    ]
    output = tmp_path / "events.gdf"

    with pytest.warns(RuntimeWarning, match="stimulus.*32768.*response.*32769"):
        command = pop_writeeeg(eeg, output, "TYPE", "GDF")
    raw = mne.io.read_raw_gdf(output, preload=False, verbose=False)

    assert list(raw.annotations.description) == ["32768", "32768", "32769"]
    assert list(raw.annotations.onset * eeg["srate"] + 1) == pytest.approx([20, 40, 60], abs=1e-3)
    # MNE normalizes zero-duration GDF events to one sample when reading.
    assert list(raw.annotations.duration) == [1 / eeg["srate"]] * 3
    assert command == f"LASTCOM = pop_writeeeg(EEG, '{output}', 'TYPE', 'GDF');"


def test_writegdf_preserves_fractional_sampling_rates_without_events(tmp_path: Path) -> None:
    eeg = _eeg()
    eeg["srate"] = 256.0175
    eeg["event"] = []
    output = tmp_path / "fractional-rate.gdf"

    _write(eeg, output)
    raw = mne.io.read_raw_gdf(output, preload=True, verbose=False)

    assert raw.info["sfreq"] == pytest.approx(256.0175, rel=1e-12)
    np.testing.assert_allclose(raw.get_data(units="uV"), eeg["data"], rtol=0, atol=1e-12)


def test_writegdf_supports_sub_hertz_sampling_without_events(tmp_path: Path) -> None:
    eeg = _eeg()
    eeg["srate"] = 0.5
    eeg["event"] = []
    output = tmp_path / "sub-hertz.gdf"

    _write(eeg, output)
    raw = mne.io.read_raw_gdf(output, preload=False, verbose=False)

    assert raw.info["sfreq"] == pytest.approx(0.5, rel=1e-12)


def test_writegdf_avoids_collisions_between_numeric_and_text_event_codes(tmp_path: Path) -> None:
    eeg = _eeg()
    eeg["event"] = [
        {"type": 32768, "latency": 10},
        {"type": "stimulus", "latency": 20},
    ]
    output = tmp_path / "codes.gdf"

    with pytest.warns(RuntimeWarning, match="stimulus.*32769"):
        mapping = _write(eeg, output)
    raw = mne.io.read_raw_gdf(output, preload=False, verbose=False)

    assert mapping == {"stimulus": 32769}
    assert list(raw.annotations.description) == ["32768", "32769"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda eeg: eeg.update(data=eeg["data"][:, :, None], trials=2), "continuous"),
        (lambda eeg: eeg.update(data=eeg["data"].astype(complex) + 1j), "real"),
        (lambda eeg: eeg["data"].__setitem__((0, 0), np.nan), "finite"),
        (lambda eeg: eeg.update(srate=0), "sampling_rate"),
        (lambda eeg: eeg["chanlocs"][0].update(labels="label-is-more-than-sixteen-bytes"), "16 bytes"),
        (lambda eeg: eeg["event"][0].update(latency=1000), "outside"),
        (lambda eeg: eeg["event"][0].update(type=2.5), "integers"),
        (lambda eeg: eeg.update(srate=256.5), "integer sampling rate"),
    ],
)
def test_writegdf_rejects_data_that_cannot_be_represented(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    eeg = _eeg()
    mutation(eeg)

    with pytest.raises(ValueError, match=message):
        _write(eeg, tmp_path / "invalid.gdf")


def test_writegdf_and_pop_writeeeg_reject_mismatched_output_types(tmp_path: Path) -> None:
    eeg = _eeg()

    with pytest.raises(ValueError, match="end in .gdf"):
        _write(eeg, tmp_path / "invalid.edf")
    with pytest.raises(ValueError, match="TYPE must match"):
        pop_writeeeg(eeg, tmp_path / "invalid.gdf", "TYPE", "EDF")


def test_writegdf_is_public() -> None:
    assert eegprep.writegdf is writegdf
