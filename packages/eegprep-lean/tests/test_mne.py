"""Tests for ``to_mne_raw``: a Window as an ``mne.io.RawArray``.

Offline, on hand-built windows. The tests that build a ``RawArray`` need the ``mne``
extra and skip without it; the missing-extra test does not, because it is the one that
runs where mne is absent.
"""

from __future__ import annotations

import sys

import pytest

np = pytest.importorskip("numpy", reason="needs numpy, which the zarr and plot extras bring")

import eegprep_lean  # noqa: E402
from eegprep_lean.mne import to_mne_raw  # noqa: E402
from eegprep_lean.window import Window  # noqa: E402

RATE = 250.0
LABELS = ("E1", "E2", "E3")
DATA = [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0], [-1.0, -2.0, -3.0, -4.0]]


def window(data=DATA, *, unit: str | None = "uV", labels=LABELS, physical: bool = True, rate: float = RATE) -> Window:
    array = np.asarray(data, dtype=np.float64)
    return Window(
        data=array,
        channels=tuple(range(array.shape[0])),
        start_sample=0,
        rate=rate,
        group_name="eeg_250hz",
        physical=physical,
        labels=labels,
        unit=unit,
        original_rate=500.0,
    )


class TestWithMne:
    @pytest.fixture(autouse=True)
    def _needs_mne(self) -> None:
        pytest.importorskip("mne", reason="needs the mne extra")

    def test_returns_a_raw_array_in_volts(self) -> None:
        import mne

        raw = to_mne_raw(window())

        assert isinstance(raw, mne.io.RawArray)
        np.testing.assert_allclose(raw.get_data(), np.asarray(DATA) * 1e-6, rtol=1e-12)

    @pytest.mark.parametrize(("unit", "factor"), [("uV", 1e-6), ("mV", 1e-3), ("V", 1.0)])
    def test_each_accepted_unit_becomes_volts(self, unit: str, factor: float) -> None:
        raw = to_mne_raw(window(unit=unit))

        np.testing.assert_allclose(raw.get_data(), np.asarray(DATA) * factor, rtol=1e-12)

    def test_the_rate_names_and_default_types(self) -> None:
        raw = to_mne_raw(window())

        assert raw.info["sfreq"] == RATE
        assert raw.ch_names == list(LABELS)
        assert raw.get_channel_types() == ["eeg", "eeg", "eeg"]
        assert raw.n_times == 4

    def test_a_non_integer_rate_is_kept_exactly(self) -> None:
        assert to_mne_raw(window(rate=256.5)).info["sfreq"] == 256.5

    def test_ch_types_can_be_one_string_or_one_per_channel(self) -> None:
        assert to_mne_raw(window(), ch_types="eog").get_channel_types() == ["eog"] * 3
        assert to_mne_raw(window(), ch_types=["eeg", "eog", "eeg"]).get_channel_types() == ["eeg", "eog", "eeg"]

    @pytest.mark.parametrize("unit", ["T", "nT", "degC", "", "UV"])
    def test_any_other_unit_is_refused(self, unit: str) -> None:
        with pytest.raises(ValueError, match="unit"):
            to_mne_raw(window(unit=unit))

    def test_a_window_without_a_single_unit_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unit"):
            to_mne_raw(window(unit=None))

    def test_stored_counts_are_refused(self) -> None:
        with pytest.raises(ValueError, match="physical"):
            to_mne_raw(window(physical=False))

    def test_a_window_without_labels_is_refused(self) -> None:
        with pytest.raises(ValueError, match="label"):
            to_mne_raw(window(labels=None))

    def test_the_package_exports_it(self) -> None:
        assert eegprep_lean.to_mne_raw is to_mne_raw


class TestWithoutMne:
    def test_the_error_names_the_mne_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``None`` in ``sys.modules`` makes ``import mne`` fail the way an absent
        package does, so this runs the real function in the state it is for."""
        monkeypatch.setitem(sys.modules, "mne", None)

        with pytest.raises(ImportError, match=r"eegprep-lean\[mne\]"):
            to_mne_raw(window())

    def test_an_unrelated_import_error_is_not_blamed_on_the_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """mne present but broken inside is a bug to see, not an extra to install."""
        import builtins

        real_import = builtins.__import__

        def broken(name, *args, **kwargs):
            if name == "mne":
                raise ImportError("cannot import name 'x' from 'somewhere_else'", name="somewhere_else")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", broken)

        with pytest.raises(ImportError) as caught:
            to_mne_raw(window())
        monkeypatch.undo()

        assert "extra" not in str(caught.value)
        assert caught.value.name == "somewhere_else"
