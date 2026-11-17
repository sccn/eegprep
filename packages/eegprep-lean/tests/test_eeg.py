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
import json
import math
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="needs numpy, which the zarr and plot extras bring")

import eegprep_lean  # noqa: E402
from eegprep_lean.eeg import to_eeg  # noqa: E402
from eegprep_lean.electrodes import ElectrodeAttributes, ElectrodePositionsNotAnObject  # noqa: E402
from eegprep_lean.index import IndexError_  # noqa: E402
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
                # this API is added and 0.5.0 predates it. v2 is the first reader that fills
                # chanlocs from a store's electrode attributes, so a record that names it
                # can be told from one made by a reader that never looked for them.
                "reader": f"eegprep-lean@{eegprep_lean.__version__} recording-v2",
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

    @pytest.mark.parametrize("unit", ["uV", "\u00b5V", "\u03bcV"], ids=["ascii-u", "micro-sign", "greek-mu"])
    def test_every_spelling_of_microvolts_gives_the_same_values(self, unit: str) -> None:
        """The archive's own index says the micro sign (U+00B5), and the Greek small letter
        mu (U+03BC) is what some tools write for the same unit. Refusing either would make
        most real stores unreadable while the plain ascii one, which the first dataset
        tested happened to use, passed."""
        eeg = to_eeg(window(DATA, unit=unit), **IDS)

        np.testing.assert_array_equal(eeg["data"], np.asarray(DATA))

    @pytest.mark.parametrize("unit", ["T", "nT", "degC", "", "UV", "mu", "\u00b5v", "\u00b5", "m\u00b5V"])
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

    def test_eeg_checkset_keeps_the_filled_locations(self) -> None:
        eeg_checkset = pytest.importorskip("eegprep").eeg_checkset
        attributes = stated({"E1": ALS_MM["E1"], "E3": ALS_MM["E3"]}, "EEGLAB", "mm")
        eeg = to_eeg(located(attributes), **IDS)

        checked = eeg_checkset(eeg)

        for before, after in zip(eeg["chanlocs"], checked["chanlocs"], strict=True):
            assert after["labels"] == before["labels"]
            if np.size(before["X"]):
                assert [after[field] for field in FIELDS] == [before[field] for field in FIELDS]
        assert np.size(checked["chanlocs"][1]["X"]) == 0


# Three electrodes in EEGLAB's ALS frame (+X toward the nose, +Y to the left, +Z up), in
# millimeters: on the midline toward the nose, on the vertex, and on the left. Every test of
# the conversion states them in another frame or unit and expects these back.
ALS_MM = {"E1": (98.7, 0.0, 52.1), "E2": (0.0, 0.0, 110.5), "E3": (0.0, 83.2, 0.0)}
FIELDS = ("X", "Y", "Z", "sph_theta", "sph_phi", "sph_radius", "theta", "radius")
# What eegprep's own coords_ALS_to_angular gives for E1, kept as numbers so a test can
# disagree with the code under test.
E1_SPH_PHI = 27.827879865538932
E1_RADIUS = 0.3454006674136726

# The BIDS coordinate-systems appendix: ALS is EEGLAB's own frame, RAS is turned into it.
ALS_SYSTEMS = ["EEGLAB", "EEGLAB-HJ", "CTF", "4DBti", "KitYokogawa"]
RAS_SYSTEMS = [
    "CapTrak",
    "NeuromagElektaMEGIN",
    "ElektaNeuromag",
    "ChietiItab",
    # The standard template identifiers, and the deprecated spellings the appendix keeps.
    "ICBM452AirSpace",
    "ICBM452Warp5Space",
    "IXI549Space",
    "fsaverage",
    "fsaverageSym",
    "fsLR",
    "MNIColin27",
    "MNI152Lin",
    *(f"MNI152NLin2009{version}{symmetry}" for version in "abc" for symmetry in ("Sym", "Asym")),
    "MNI152NLin6Sym",
    "MNI152NLin6ASym",
    "MNI305",
    "NIHPD",
    "OASIS30AntsOASISAnts",
    "OASIS30Atropos",
    "Talairach",
    "UNCInfant",
    "fsaverage3",
    "fsaverage4",
    "fsaverage5",
    "fsaverage6",
    "fsaveragesym",
    *(f"UNCInfant{age}V{volume}" for age in "012" for volume in ("21", "22", "23")),
]
UNIT_SCALES = {"mm": 1.0, "cm": 0.1, "m": 0.001}


def declared(system: str, unit: str) -> dict[str, tuple[float, float, float]]:
    """``ALS_MM`` as a dataset in ``system`` and ``unit`` would state it: a RAS frame names the
    left of the head by a negative x and the nose by +y."""
    scale = UNIT_SCALES[unit]
    ras = system in RAS_SYSTEMS
    return {
        label: tuple(value * scale for value in ((-y, x, z) if ras else (x, y, z)))
        for label, (x, y, z) in ALS_MM.items()
    }


def stated(positions, system: str = "EEGLAB", units: str = "mm", **optional: str) -> ElectrodeAttributes:
    return ElectrodeAttributes(positions=positions, coordinate_system=system, coordinate_units=units, **optional)


def located(attributes: ElectrodeAttributes | None) -> Window:
    return dataclasses.replace(window(DATA), electrodes=attributes)


def locations_of(eeg: dict) -> dict:
    return eeg["etc"]["nemar"]["locations"]


class TestElectrodeAttributes:
    def test_entries_that_are_not_three_finite_numbers_are_dropped(self) -> None:
        stored = {
            "E1": [1.0, 2.0, 3.0],
            "E2": [float("nan"), 0.0, 0.0],
            "E3": [0.0, float("inf"), 0.0],
            "E4": [1.0, 2.0],
            "E5": ["1", 2, 3],
            "E6": [True, 0, 0],
            "E7": None,
            "E8": [1, 2, 3, 4],
            "E9": [4, 5, 6],
        }

        attributes = ElectrodeAttributes.from_attributes({"electrode_positions": stored})

        assert attributes.positions == {"E1": (1.0, 2.0, 3.0), "E9": (4.0, 5.0, 6.0)}

    def test_an_integer_too_large_for_a_float_is_a_channel_with_no_position(self) -> None:
        """``float`` raises ``OverflowError`` from 2**1024 - 2**970 up, which a reader of JSON
        integers as binary64 sees as an infinity: a channel without a position, and no refusal
        of the store. Below that it is the largest float."""
        stored = {
            "E1": [1, 2, 10**399],
            "E2": [-(10**399), 2, 3],
            "E3": [0, 0, 2**1024 - 2**970],
            "E4": [0, 0, 2**1024 - 2**970 - 1],
            "E5": [4, 5, 6],
        }

        attributes = ElectrodeAttributes.from_attributes({"electrode_positions": stored})

        assert attributes.positions == {"E4": (0.0, 0.0, sys.float_info.max), "E5": (4.0, 5.0, 6.0)}

    @pytest.mark.parametrize("stored", [[], ["E1", "E2"], [[1, 2, 3]], "E1 1 2 3", 3, 1.5, True, False])
    def test_a_position_map_that_is_not_an_object_is_refused_by_name(self, stored) -> None:
        """A store that states positions in a shape no entry can be read from is not a store
        without positions: NEMAR's locations digest refuses it too. The refusal has a type of
        its own, so a caller records why the read failed without parsing a message."""
        with pytest.raises(ElectrodePositionsNotAnObject, match=f"is a {type(stored).__name__}") as caught:
            ElectrodeAttributes.from_attributes({"electrode_positions": stored})

        assert isinstance(caught.value, IndexError_)

    def test_a_null_position_map_states_no_positions(self) -> None:
        assert ElectrodeAttributes.from_attributes({"electrode_positions": None}) is None

    def test_no_position_map_is_no_attributes_whatever_else_is_there(self) -> None:
        assert ElectrodeAttributes.from_attributes({"electrode_coordinate_system": "CapTrak"}) is None
        assert ElectrodeAttributes.from_attributes({}) is None

    def test_a_missing_system_or_unit_is_the_empty_string_the_converter_writes(self) -> None:
        attributes = ElectrodeAttributes.from_attributes({"electrode_positions": {"E1": [1, 2, 3]}})

        assert (attributes.coordinate_system, attributes.coordinate_units) == ("", "")
        assert (attributes.source, attributes.file) == (None, None)

    def test_the_source_word_and_file_are_kept_only_when_they_are_text(self) -> None:
        stored = {"electrode_positions": {"E1": [1, 2, 3]}}

        text = ElectrodeAttributes.from_attributes(
            {**stored, "electrode_positions_source": "dataset", "electrode_positions_file": "electrodes.tsv"}
        )
        junk = ElectrodeAttributes.from_attributes(
            {**stored, "electrode_positions_source": 3, "electrode_positions_file": ""}
        )

        assert (text.source, text.file) == ("dataset", "electrodes.tsv")
        assert (junk.source, junk.file) == (None, None)


class TestLocations:
    def test_a_window_without_attributes_is_labels_only_as_before(self) -> None:
        eeg = to_eeg(window(DATA), **IDS)

        assert eeg["chanlocs"] == [{"labels": "E1"}, {"labels": "E2"}, {"labels": "E3"}]
        assert eeg["chaninfo"] == {}
        assert "locations" not in eeg["etc"]["nemar"]

    @pytest.mark.parametrize("unit", UNIT_SCALES)
    @pytest.mark.parametrize("system", ALS_SYSTEMS + RAS_SYSTEMS)
    def test_every_accepted_system_and_unit_gives_the_same_locations(self, system: str, unit: str) -> None:
        eeg = to_eeg(located(stated(declared(system, unit), system, unit)), **IDS)

        by_label = {loc["labels"]: loc for loc in eeg["chanlocs"]}
        for label, expected in ALS_MM.items():
            assert [by_label[label][axis] for axis in "XYZ"] == pytest.approx(expected, abs=1e-9)
        # The nose, the vertex and the left ear, whose angles are known without arithmetic.
        e1, e2, e3 = (by_label[label] for label in ("E1", "E2", "E3"))
        assert (e1["sph_theta"], e1["theta"]) == pytest.approx((0.0, 0.0), abs=1e-9)
        assert (e1["sph_phi"], e1["radius"]) == pytest.approx((E1_SPH_PHI, E1_RADIUS), abs=1e-9)
        assert e1["sph_radius"] == pytest.approx(math.hypot(98.7, 52.1), abs=1e-9)
        assert (e2["sph_phi"], e2["radius"]) == pytest.approx((90.0, 0.0), abs=1e-9)
        assert (e3["sph_theta"], e3["theta"], e3["radius"]) == pytest.approx((90.0, -90.0, 0.5), abs=1e-9)
        assert locations_of(eeg)["located"] == 3

    @pytest.mark.parametrize(
        ("spelling", "factor"),
        [("MM", 1.0), ("Cm", 10.0), ("M", 1000.0), ("millimeters", 1.0), ("centimeters", 10.0), ("meters", 1000.0)],
    )
    def test_unit_names_are_read_as_the_bids_import_reads_them(self, spelling: str, factor: float) -> None:
        eeg = to_eeg(located(stated({"E1": (1.0, 2.0, 3.0)}, "EEGLAB", spelling)), **IDS)

        assert [eeg["chanlocs"][0][axis] for axis in "XYZ"] == pytest.approx([factor, 2 * factor, 3 * factor])

    def test_a_channel_without_an_entry_gets_no_position_while_the_others_do(self) -> None:
        attributes = stated({"E1": ALS_MM["E1"], "E3": ALS_MM["E3"]})

        eeg = to_eeg(located(attributes), **IDS)

        e1, e2, e3 = eeg["chanlocs"]
        assert (e1["X"], e3["Y"]) == (98.7, 83.2)
        # What the BIDS import leaves for it: its label and an empty array for each field.
        assert set(e2) == {"labels", *FIELDS}
        assert all(np.size(e2[field]) == 0 for field in FIELDS)
        assert locations_of(eeg)["located"] == 2
        assert "reason" not in locations_of(eeg)

    def test_an_entry_belongs_to_exactly_its_label(self) -> None:
        attributes = stated({"e1": ALS_MM["E1"], "E1 ": ALS_MM["E1"], "E2": ALS_MM["E2"]})

        eeg = to_eeg(located(attributes), **IDS)

        assert [np.size(loc.get("X", [])) for loc in eeg["chanlocs"]] == [0, 1, 0]
        assert locations_of(eeg)["located"] == 1

    def test_electrodes_the_recording_lacks_are_ignored(self) -> None:
        attributes = stated({**ALS_MM, "Cz": (0.0, 0.0, 99.0)})

        eeg = to_eeg(located(attributes), **IDS)

        assert [loc["labels"] for loc in eeg["chanlocs"]] == ["E1", "E2", "E3"]
        assert locations_of(eeg)["located"] == 3

    def test_no_entry_matching_any_channel_is_no_positions_and_says_why(self) -> None:
        eeg = to_eeg(located(stated({"Cz": (0.0, 0.0, 99.0)})), **IDS)

        assert eeg["chanlocs"] == [{"labels": "E1"}, {"labels": "E2"}, {"labels": "E3"}]
        assert eeg["chaninfo"] == {}
        assert locations_of(eeg) == {
            "coordinate_system": "EEGLAB",
            "coordinate_units": "mm",
            "located": 0,
            "reason": "no_matching_labels",
        }

    @pytest.mark.parametrize("units", ["", "n/a", "ft", "inches", "µm"])
    def test_an_unknown_unit_gives_no_positions_and_says_why(self, units: str) -> None:
        eeg = to_eeg(located(stated(declared("CapTrak", "m"), "CapTrak", units)), **IDS)

        assert eeg["chanlocs"] == [{"labels": "E1"}, {"labels": "E2"}, {"labels": "E3"}]
        assert eeg["chaninfo"] == {}
        assert locations_of(eeg) == {
            "coordinate_system": "CapTrak",
            "coordinate_units": units,
            "located": 0,
            "reason": "units_unknown",
        }

    @pytest.mark.parametrize(
        "system",
        [
            "",
            "Other",
            "individual",
            "study",
            "scanner",
            "Pixels",
            "ACPC",
            "ScanRAS",
            "captrak",
            "eeglab",
            "EEGLAB ",
            "MNI",
        ],
    )
    def test_an_unknown_system_gives_no_positions_and_says_why(self, system: str) -> None:
        eeg = to_eeg(located(stated(declared("CapTrak", "m"), system, "m")), **IDS)

        assert eeg["chanlocs"] == [{"labels": "E1"}, {"labels": "E2"}, {"labels": "E3"}]
        assert eeg["chaninfo"] == {}
        assert locations_of(eeg) == {
            "coordinate_system": system,
            "coordinate_units": "m",
            "located": 0,
            "reason": "coordinate_system_unknown",
        }

    def test_a_store_with_neither_declared_says_the_unit_first(self) -> None:
        """A dataset with no coordsystem file has both empty. One reason is recorded, and
        the order is the order the conversion needs them: millimeters, then the frame."""
        eeg = to_eeg(located(stated(ALS_MM, "", "")), **IDS)

        assert locations_of(eeg)["reason"] == "units_unknown"

    def test_the_record_carries_what_the_store_said(self) -> None:
        attributes = stated(declared("CapTrak", "m"), "CapTrak", "m", source="dataset", file="electrodes.tsv")

        eeg = to_eeg(located(attributes), **IDS)

        assert locations_of(eeg) == {
            "source": "dataset",
            "file": "electrodes.tsv",
            "coordinate_system": "CapTrak",
            "coordinate_units": "m",
            "located": 3,
        }

    def test_nosedir_is_plus_x_once_locations_are_filled(self) -> None:
        assert to_eeg(located(stated(ALS_MM)), **IDS)["chaninfo"] == {"nosedir": "+X"}

    def test_the_stores_positions_are_not_changed(self) -> None:
        positions = declared("CapTrak", "m")
        attributes = stated(positions, "CapTrak", "m")
        before = dict(attributes.positions)

        to_eeg(located(attributes), **IDS)

        assert attributes.positions == before


def _scalp_electrodes(names: list[str], unit: str) -> dict[str, tuple[float, float, float]]:
    """Points on the upper half of a 90 mm sphere, written as a short decimal in ``unit``,
    as an electrodes file would hold them."""
    rng = np.random.default_rng(7)
    azimuth = rng.uniform(-np.pi, np.pi, len(names))
    elevation = rng.uniform(0.1, np.pi / 2, len(names))
    points = 90.0 * np.stack(
        [np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)], axis=1
    )
    digits = {"mm": 2, "cm": 3, "m": 5}[unit]
    return {
        name: tuple(round(float(value) * UNIT_SCALES[unit], digits) for value in point)
        for name, point in zip(names, points, strict=True)
    }


def _write_bids_edf(root: Path, names: list[str], rows: dict, system: str, unit: str) -> Path:
    """A one-recording BIDS dataset whose electrodes file holds ``rows``."""
    pyedflib = pytest.importorskip("pyedflib", reason="needs pyedflib, which eegprep installs")
    eeg_dir = root / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.9.0", "DatasetType": "raw"})
    )
    edf = eeg_dir / "sub-01_task-x_eeg.edf"
    writer = pyedflib.EdfWriter(str(edf), len(names), file_type=pyedflib.FILETYPE_EDFPLUS)
    writer.setSignalHeaders(
        [
            {
                "label": name,
                "dimension": "uV",
                "sample_frequency": 100,
                "physical_min": -100,
                "physical_max": 100,
                "digital_min": -32768,
                "digital_max": 32767,
                "transducer": "",
                "prefilter": "",
            }
            for name in names
        ]
    )
    writer.writeSamples(list(np.random.default_rng(0).standard_normal((len(names), 500)) * 20))
    writer.close()
    (eeg_dir / "sub-01_task-x_channels.tsv").write_text(
        "name\ttype\tunits\n" + "".join(f"{name}\tEEG\tuV\n" for name in names)
    )
    (eeg_dir / "sub-01_task-x_electrodes.tsv").write_text(
        "name\tx\ty\tz\n" + "".join(f"{name}\t{x}\t{y}\t{z}\n" for name, (x, y, z) in rows.items())
    )
    (eeg_dir / "sub-01_task-x_coordsystem.json").write_text(
        json.dumps({"EEGCoordinateSystem": system, "EEGCoordinateUnits": unit})
    )
    return edf


# Systems and units the BIDS import reads correctly: it turns every system whose name has
# EEGLAB in it into ALS and every other into RAS. CTF, 4DBti and KitYokogawa are ALS in the
# appendix, so the Zarr route differs from the import there by design; see the pull request.
SAME_AS_THE_BIDS_IMPORT = [
    ("CapTrak", "m"),
    ("CapTrak", "cm"),
    ("CapTrak", "mm"),
    ("EEGLAB", "mm"),
    ("EEGLAB", "cm"),
    ("EEGLAB-HJ", "m"),
    ("NeuromagElektaMEGIN", "m"),
    ("ElektaNeuromag", "mm"),
    ("ChietiItab", "cm"),
    ("MNI305", "mm"),
    ("MNI152NLin2009cAsym", "mm"),
    ("Talairach", "mm"),
    ("fsaverage", "m"),
]


class TestTheZarrRouteGivesTheBidsRoutesAnswer:
    """One set of coordinates, written once as an ``electrodes.tsv`` that ``pop_load_frombids``
    reads and once as the root attributes a store carries, which ``to_eeg`` reads."""

    @pytest.mark.parametrize(("system", "unit"), SAME_AS_THE_BIDS_IMPORT)
    def test_same_chanlocs_as_pop_load_frombids(self, tmp_path: Path, system: str, unit: str) -> None:
        pytest.importorskip("bids", reason="needs pybids, the eegprep[pybids] extra")
        pop_load_frombids = pytest.importorskip("eegprep").pop_load_frombids
        names = [f"E{number}" for number in range(1, 13)]
        # E12 has no row, and "Extra" has a row and no channel.
        rows = _scalp_electrodes([*names[:-1], "Extra"], unit)
        edf = _write_bids_edf(tmp_path, names, rows, system, unit)

        from_bids = pop_load_frombids(str(edf), bidsevent=False, infer_locations=False, dtype=np.float64, verbose=False)

        stored = json.loads(
            json.dumps(
                {
                    "electrode_positions": {name: list(xyz) for name, xyz in rows.items()},
                    "electrode_coordinate_system": system,
                    "electrode_coordinate_units": unit,
                }
            )
        )
        source = dataclasses.replace(
            window(np.zeros((len(names), 4)), labels=tuple(names)),
            electrodes=ElectrodeAttributes.from_attributes(stored),
        )
        from_zarr = to_eeg(source, **IDS)

        assert [loc["labels"] for loc in from_zarr["chanlocs"]] == [loc["labels"] for loc in from_bids["chanlocs"]]
        for zarr_loc, bids_loc in zip(from_zarr["chanlocs"], from_bids["chanlocs"], strict=True):
            for field in FIELDS:
                if bids_loc["labels"] == "E12":
                    assert np.size(zarr_loc[field]) == 0
                    assert np.size(bids_loc[field]) == 0
                    continue
                assert zarr_loc[field] == bids_loc[field], (bids_loc["labels"], field)
        assert from_zarr["chaninfo"]["nosedir"] == from_bids["chaninfo"]["nosedir"]
        assert locations_of(from_zarr)["located"] == 11


class TestICLabelRunsOnALocatedRecording:
    """The reason the locations are wanted: ICLabel draws a scalp map per component, and
    ``topoplot`` has nothing to draw from a channel with no ``theta`` and ``radius``."""

    SAMPLE = Path(__file__).resolve().parents[3] / "sample_data" / "eeglab_data_with_ica_tmp.set"

    @staticmethod
    def _recording(eeg: dict, *, with_locations: bool) -> dict:
        """EEGLAB's own sample recording, as a store would give it: data and labels, and the
        electrode positions in the attributes when asked for. The decomposition is carried
        over unchanged, because a store has none."""
        labels = tuple(loc["labels"] for loc in eeg["chanlocs"])
        # The two EOG channels have no position in the sample, as an electrodes file lacks a row.
        positions = {loc["labels"]: (loc["X"], loc["Y"], loc["Z"]) for loc in eeg["chanlocs"] if np.size(loc["X"])}
        source = Window(
            data=eeg["data"],
            channels=tuple(range(len(labels))),
            start_sample=0,
            rate=eeg["srate"],
            group_name="eeg_128hz",
            physical=True,
            labels=labels,
            unit="uV",
            electrodes=stated(positions, "EEGLAB", "mm") if with_locations else None,
        )
        recording = to_eeg(source, **IDS)
        for key in ("icawinv", "icasphere", "icaweights", "icachansind", "icaact"):
            recording[key] = eeg[key]
        return recording

    @pytest.fixture(scope="class")
    def sample(self) -> dict:
        pop_loadset = pytest.importorskip("eegprep").pop_loadset
        return pop_loadset(str(self.SAMPLE))

    def test_without_locations_topoplot_raises(self, sample: dict) -> None:
        ICL_feature_extractor = pytest.importorskip("eegprep").ICL_feature_extractor

        with pytest.raises(ValueError, match="theta"):
            ICL_feature_extractor(self._recording(sample, with_locations=False))

    def test_with_locations_the_scalp_maps_are_those_of_eeglabs_own_channel_locations(self, sample: dict) -> None:
        ICL_feature_extractor = pytest.importorskip("eegprep").ICL_feature_extractor

        features = ICL_feature_extractor(self._recording(sample, with_locations=True))
        reference = ICL_feature_extractor(sample)

        topo = features[0]
        assert topo.shape == (32, 32, 1, 32)
        assert np.isfinite(topo).all()
        assert np.abs(topo).max() > 0
        np.testing.assert_array_equal(topo, reference[0])


LIVE_DATASET = "nm000132"
# A second public dataset, because the first one's index spells microvolts in ascii and the
# archive's other stores do not.
MICRO_SIGN_DATASET = "nm000274"


@pytest.mark.network
class TestAgainstTheLiveArchive:
    def test_the_unit_string_a_live_index_really_carries_is_accepted(self) -> None:
        """The string is read from the live group, not typed here: what the archive serves
        is the thing under test, and a typed copy would pass for the wrong reason."""
        from eegprep_lean import read_group_metadata, read_index

        async def run():
            index = await read_index(MICRO_SIGN_DATASET)
            store = min(index.stores, key=lambda s: sum(g.n_channels * g.n_samples for g in s.groups))
            return await read_group_metadata(index, store)

        metadata = asyncio.run(run())
        rows = tuple(range(metadata.n_channels))
        unit = metadata.unit(rows)
        assert unit is not None, "the live group's channels do not share one unit"

        source = window(DATA)
        eeg = to_eeg(dataclasses.replace(source, unit=unit), **IDS)

        np.testing.assert_array_equal(eeg["data"], np.asarray(DATA))

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
