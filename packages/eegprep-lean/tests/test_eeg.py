"""Tests for ``to_eeg``: a Window as an EEGLAB-compatible dictionary.

Offline, on hand-built windows (a real :class:`~eegprep_lean.Window`, with real numbers),
because the conversion is arithmetic and bookkeeping and needs no server. The one test
that needs eegprep's own ``eeg_checkset`` skips where eegprep is not installed, which is
the isolated lean tiers; the repository's main environment has it, so it runs in CI.

The ``network`` test closes the loop: a live recording, through the dictionary, into
eegprep's own checker.
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys

import pytest

np = pytest.importorskip("numpy", reason="needs numpy, which the zarr and plot extras bring")

import eegprep_lean  # noqa: E402
from eegprep_lean.eeg import to_eeg  # noqa: E402
from eegprep_lean.window import Window  # noqa: E402

RATE = 250.0
LABELS = ("E1", "E2", "E3")
IDS = {"dataset_id": "xx099999", "store": "sub-01/eeg/sub-01_task-test_eeg.set", "index_digest": "ab" * 32}

# The key set of eegprep's eeg_emptyset(), written out here so a key added there is a
# failing test here rather than a silent difference. The lean package must not import
# eegprep to learn it; test_the_keys_are_eegpreps_own checks this list against the source.
EMPTYSET_KEYS = {
    "setname",
    "filename",
    "filepath",
    "subject",
    "group",
    "condition",
    "session",
    "comments",
    "nbchan",
    "trials",
    "pnts",
    "srate",
    "xmin",
    "xmax",
    "times",
    "data",
    "chanlocs",
    "urchanlocs",
    "ref",
    "event",
    "urevent",
    "eventdescription",
    "epoch",
    "epochdescription",
    "history",
    "saved",
    "run",
    "icaact",
    "icawinv",
    "icasphere",
    "icaweights",
    "icachansind",
    "chaninfo",
    "reject",
    "stats",
    "specdata",
    "specicaact",
    "splinefile",
    "icasplinefile",
    "dipfit",
    "datfile",
    "roi",
    "etc",
}


def window(data, *, unit: str | None = "uV", labels=LABELS, physical: bool = True, rate: float = RATE) -> Window:
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


DATA = [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0], [-1.0, -2.0, -3.0, -4.0]]


class TestTheDictionary:
    def test_has_exactly_the_keys_of_an_empty_set(self) -> None:
        assert set(to_eeg(window(DATA), **IDS)) == EMPTYSET_KEYS

    def test_the_keys_are_eegpreps_own(self) -> None:
        eeg_emptyset = pytest.importorskip("eegprep").eeg_emptyset

        assert set(eeg_emptyset()) == EMPTYSET_KEYS

    def test_the_core_fields(self) -> None:
        eeg = to_eeg(window(DATA), **IDS)

        assert eeg["nbchan"] == 3
        assert eeg["pnts"] == 4
        assert eeg["srate"] == RATE
        assert eeg["trials"] == 1
        assert eeg["xmin"] == 0.0
        assert eeg["xmax"] == pytest.approx(3 / RATE)
        assert eeg["data"].shape == (3, 4)
        assert eeg["data"].dtype == np.float64

    def test_times_are_milliseconds_like_eeglabs(self) -> None:
        eeg = to_eeg(window(DATA), **IDS)

        assert eeg["times"].tolist() == pytest.approx([0.0, 4.0, 8.0, 12.0])

    def test_chanlocs_are_labels_in_row_order(self) -> None:
        eeg = to_eeg(window(DATA), **IDS)

        assert eeg["chanlocs"] == [{"labels": "E1"}, {"labels": "E2"}, {"labels": "E3"}]

    def test_the_name_and_provenance(self) -> None:
        eeg = to_eeg(window(DATA), **IDS)

        assert eeg["setname"] == "xx099999/sub-01/eeg/sub-01_task-test_eeg.set"
        assert eeg["etc"] == {
            "nemar": {
                "dataset_id": "xx099999",
                "store": "sub-01/eeg/sub-01_task-test_eeg.set",
                "index_sha256": "ab" * 32,
                # The marker is the API's own, because the version field is not bumped when
                # this API is added and 0.5.0 predates it.
                "reader": f"eegprep-lean@{eegprep_lean.__version__} recording-v1",
            }
        }

    def test_the_data_is_a_copy_even_when_no_conversion_is_needed(self) -> None:
        """A microvolt window needs no arithmetic, which is exactly when returning the
        window's own array is tempting, and an in-place step on the dictionary would then
        rewrite the Window it came from."""
        source = window(DATA, unit="uV")

        eeg = to_eeg(source, **IDS)
        eeg["data"][0, 0] = 999.0

        assert source.data[0, 0] == 1.0
        assert not np.shares_memory(eeg["data"], source.data)

    def test_the_data_is_float64_whatever_the_window_holds(self) -> None:
        source = dataclasses.replace(window(DATA, unit="uV"), data=np.asarray(DATA, dtype=np.float32))
        assert source.data.dtype == np.float32

        assert to_eeg(source, **IDS)["data"].dtype == np.float64

    def test_a_non_integer_rate_is_kept_exactly(self) -> None:
        """int() on the rate would turn 256.5 Hz into 256 and shift every time after the
        first sample, with nothing raising."""
        eeg = to_eeg(window(DATA, rate=256.5), **IDS)

        assert eeg["srate"] == 256.5
        assert eeg["xmax"] == pytest.approx(3 / 256.5)
        assert eeg["times"][1] == pytest.approx(1000 / 256.5)

    def test_each_call_returns_fresh_containers(self) -> None:
        first = to_eeg(window(DATA), **IDS)
        first["event"].append("x")
        first["etc"]["extra"] = 1

        second = to_eeg(window(DATA), **IDS)

        assert second["event"] == []
        assert "extra" not in second["etc"]


class TestWithoutNumpy:
    def test_the_error_names_an_extra_that_brings_numpy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """numpy arrives with the zarr or the plot extra, and to_eeg is not behind zarr:
        it works under plot alone, so naming zarr would send a plot-only install to
        install the wrong thing."""
        monkeypatch.setitem(sys.modules, "numpy", None)
        monkeypatch.delitem(sys.modules, "eegprep_lean.eeg", raising=False)
        monkeypatch.delitem(sys.modules, "eegprep_lean.window", raising=False)

        with pytest.raises(ImportError, match="to_eeg needs the plot extra") as caught:
            eegprep_lean.to_eeg
        monkeypatch.undo()

        assert "zarr" not in str(caught.value).split(":")[0]


class TestUnits:
    @pytest.mark.parametrize(("unit", "factor"), [("uV", 1.0), ("mV", 1e3), ("V", 1e6)])
    def test_data_is_converted_to_microvolts(self, unit: str, factor: float) -> None:
        eeg = to_eeg(window(DATA, unit=unit), **IDS)

        np.testing.assert_allclose(eeg["data"], np.asarray(DATA) * factor, rtol=1e-12)

    @pytest.mark.parametrize("unit", ["T", "nT", "degC", "", "UV"])
    def test_any_other_unit_is_refused(self, unit: str) -> None:
        with pytest.raises(ValueError, match="unit"):
            to_eeg(window(DATA, unit=unit), **IDS)

    def test_a_window_without_a_single_unit_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unit"):
            to_eeg(window(DATA, unit=None), **IDS)

    def test_stored_counts_are_refused(self) -> None:
        """Counts are not in any unit, whatever the channel's own says."""
        with pytest.raises(ValueError, match="physical"):
            to_eeg(window(DATA, physical=False), **IDS)

    def test_a_window_without_labels_is_refused_rather_than_named_by_index(self) -> None:
        with pytest.raises(ValueError, match="label"):
            to_eeg(window(DATA, labels=None), **IDS)


class TestEegprepAcceptsIt:
    def test_eeg_checkset_keeps_the_core_fields(self) -> None:
        eeg_checkset = pytest.importorskip("eegprep").eeg_checkset
        eeg = to_eeg(window(DATA), **IDS)

        checked = eeg_checkset(eeg)

        assert (checked["nbchan"], checked["srate"], checked["pnts"]) == (3, RATE, 4)
        assert checked["data"].shape == (3, 4)


LIVE_DATASET = "nm000132"


@pytest.mark.network
class TestAgainstTheLiveArchive:
    def test_a_live_recording_passes_eegprep_checkset_with_the_indexs_values(self) -> None:
        pytest.importorskip("zarr", reason="needs the zarr extra")
        eeg_checkset = pytest.importorskip("eegprep").eeg_checkset
        from eegprep_lean import read_index
        from eegprep_lean.recording import index_sha256, load_recording

        index = asyncio.run(read_index(LIVE_DATASET))
        store = min(index.stores, key=lambda s: sum(g.n_channels * g.n_samples for g in s.groups))
        group = store.group()

        eeg = to_eeg(
            load_recording(LIVE_DATASET, store.path, index=index),
            dataset_id=LIVE_DATASET,
            store=store.path,
            index_digest=index_sha256(LIVE_DATASET),
        )
        checked = eeg_checkset(eeg)

        assert checked["nbchan"] == group.n_channels
        assert checked["srate"] == group.rate
        assert checked["pnts"] == group.n_samples
        assert checked["data"].shape == (group.n_channels, group.n_samples)
        assert len(checked["chanlocs"]) == group.n_channels
        assert checked["etc"]["nemar"]["dataset_id"] == LIVE_DATASET
