"""Tests for the NEMAR index reader.

Two tiers. The offline ones drive the real parsing and path-building code against a real
index document, captured verbatim from ``zarr.nemar.org`` and trimmed to two stores. The
``network`` ones reach the live host, because what is under test is conformance to a
contract another team serves, and a stand-in that always agrees with this reader would
prove exactly nothing about that.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from eegprep_lean import (
    DatasetIndex,
    IndexError_,
    Store,
    UnsupportedFormatVersion,
    read_index,
)
from eegprep_lean.transport import Response, _range_header

FIXTURE = Path(__file__).parent / "nm000103_index_v3.json"
LIVE_DATASET = "nm000103"

# Literal values from the captured document, so an assertion can disagree with the parser.
# A test that compares a parsed object to itself passes whatever from_entry does with the
# fields, which is how a name/modality swap survives a suite.
STORE0_PATH = "sub-NDARAA075AMK/eeg/sub-NDARAA075AMK_task-DespicableMe_eeg.set"
STORE0_ZARR = "sub-NDARAA075AMK/eeg/sub-NDARAA075AMK_task-DespicableMe_eeg.zarr"
STORE1_PATH = "sub-NDARAA075AMK/eeg/sub-NDARAA075AMK_task-DiaryOfAWimpyKid_eeg.set"
GROUP0_NAME = "eeg_250hz"
GROUP0_MODALITY = "EEG"
GROUP0_RATE = 250.0
GROUP0_CHANNELS = 129
GROUP0_SAMPLES = 43034
GROUP0_VIEW_LEVELS = 4
CONTRACT_BASE = "https://zarr.nemar.org/nm000103/zarr/"


class ReplayTransport:
    """Serves one captured document, and records what was asked for.

    This stands in for the network at the process boundary, the same way an HTTP response
    fixture does. Everything under test still runs: validation, version branching and
    path building are the real implementations.
    """

    def __init__(self, body: bytes, *, status: int = 200) -> None:
        self.body = body
        self.status = status
        self.requested: list[str] = []

    async def get(self, url: str, *, start: int | None = None, end: int | None = None) -> Response:
        self.requested.append(url)
        return Response(status=self.status, body=self.body)


def _document() -> dict:
    return json.loads(FIXTURE.read_text())


def _read(document: dict) -> tuple[DatasetIndex, ReplayTransport]:
    transport = ReplayTransport(json.dumps(document).encode())
    index = asyncio.run(read_index(LIVE_DATASET, transport=transport))
    return index, transport


class TestContractConformance:
    def test_reads_the_index_through_the_contract_host(self) -> None:
        """The one URL a client may build. Not the bucket."""
        _, transport = _read(_document())

        assert transport.requested == ["https://zarr.nemar.org/nm000103/zarr/index.json"]

    def test_refuses_a_format_version_it_does_not_implement(self) -> None:
        """An older index stays served until its dataset reconverts, so guessing is wrong."""
        document = _document()
        document["format_version"] = 2

        with pytest.raises(UnsupportedFormatVersion) as caught:
            _read(document)

        assert caught.value.found == 2
        assert "format_version 2" in str(caught.value)

    def test_a_non_json_body_is_reported_as_such(self) -> None:
        """A CDN or bucket error page arriving where the index was expected."""
        transport = ReplayTransport(b"<html>503 Service Unavailable</html>")

        with pytest.raises(IndexError_, match="not JSON"):
            asyncio.run(read_index(LIVE_DATASET, transport=transport))

    def test_an_index_without_contract_base_is_refused(self) -> None:
        """Without it there is no URL this reader is allowed to build."""
        document = _document()
        del document["contract_base"]

        with pytest.raises(IndexError_, match="no contract_base"):
            _read(document)

    def test_store_count_comes_from_the_field_not_the_array_length(self) -> None:
        """The contract says store_count is authoritative; stores[] may be a page of it.

        The fixture is trimmed to two stores from a dataset serving 3522, so a reader
        that returned len(stores) would disagree with the archive by three orders of
        magnitude and look plausible doing it.
        """
        index, _ = _read(_document())

        assert index.store_count == 3522
        assert len(index.stores) == 2

    def test_paths_are_built_from_contract_base_and_never_from_the_bucket(self) -> None:
        """data_base names the bucket the bytes sit in today. We do not follow it."""
        document = _document()
        data_base = document["data_base"]
        index, _ = _read(document)
        store = index.stores[0]

        url = index.level0_url(store)

        assert url.startswith("https://zarr.nemar.org/")
        assert "s3" not in url
        assert not url.startswith(data_base)

    def test_level0_and_view_follow_the_declared_layout_templates(self) -> None:
        """layout.level0 is <zarr>/<group>/0 and layout.view is <zarr>/<group>/view/<L>.

        Spelled out literally rather than rebuilt from the parsed object. Comparing a URL
        to one assembled from the same fields it was assembled from passes no matter what
        those fields hold, which is how a name/modality swap goes unnoticed.
        """
        index, _ = _read(_document())
        store = index.stores[0]

        assert index.level0_url(store) == f"{CONTRACT_BASE}{STORE0_ZARR}/{GROUP0_NAME}/0"
        assert index.view_url(store, 1) == f"{CONTRACT_BASE}{STORE0_ZARR}/{GROUP0_NAME}/view/1"

    def test_parses_every_group_field_into_the_field_it_names(self) -> None:
        """Catches a transposition in from_entry, which no URL assertion can see."""
        group = _read(_document())[0].stores[0].group()

        assert group.name == GROUP0_NAME
        assert group.modality == GROUP0_MODALITY
        assert group.rate == GROUP0_RATE
        assert group.n_channels == GROUP0_CHANNELS
        assert group.n_samples == GROUP0_SAMPLES
        assert group.n_view_levels == GROUP0_VIEW_LEVELS
        assert group.duration_s == GROUP0_SAMPLES / GROUP0_RATE

    def test_parses_store_identity_fields_into_the_fields_they_name(self) -> None:
        """`path` is the source recording and `zarr` is the store; swapping them builds
        URLs that look right and resolve to nothing."""
        store = _read(_document())[0].stores[0]

        assert store.path == STORE0_PATH
        assert store.zarr == STORE0_ZARR

    def test_contract_base_is_normalized_so_joins_cannot_lose_a_separator(self) -> None:
        document = _document()
        document["contract_base"] = document["contract_base"].rstrip("/")

        index, _ = _read(document)

        assert index.contract_base.endswith("/")
        assert "zarr//" not in index.level0_url(index.stores[0])


class TestViewLevels:
    def test_accepts_the_highest_level_the_group_declares(self) -> None:
        """The boundary itself. Only testing one past it leaves `<=` vs `<` undetected."""
        index, _ = _read(_document())
        store = index.stores[0]

        url = index.view_url(store, GROUP0_VIEW_LEVELS)

        assert url.endswith(f"/view/{GROUP0_VIEW_LEVELS}")

    def test_rejects_a_view_level_the_group_does_not_have(self) -> None:
        index, _ = _read(_document())
        store = index.stores[0]
        group = store.group()

        with pytest.raises(IndexError_, match="outside 1"):
            index.view_url(store, group.n_view_levels + 1)

    def test_rejects_view_level_zero_rather_than_treating_it_as_full_resolution(self) -> None:
        """Level 0 has its own template. Silently aliasing it would return the wrong array."""
        index, _ = _read(_document())

        with pytest.raises(IndexError_, match="level0_url"):
            index.view_url(index.stores[0], 0)


class TestGroupSelection:
    def test_a_single_group_needs_no_name(self) -> None:
        index, _ = _read(_document())

        assert index.stores[0].group().name == GROUP0_NAME

    def test_refuses_to_guess_between_several_groups(self) -> None:
        """Groups are one recording at different rates, so picking the first returns real
        data at the wrong sampling rate, which survives a plot and is never questioned."""
        base = _read(_document())[0].stores[0].groups[0]
        two = Store(path="x.set", zarr="x.zarr", groups=(base, replace(base, name="eeg_500hz", rate=500.0)))

        with pytest.raises(IndexError_, match="different rates"):
            two.group()

    def test_a_store_with_no_groups_says_so_rather_than_offering_a_choice(self) -> None:
        """Zero groups is not an ambiguity; "pick one: none" is not an instruction."""
        empty = Store(path="x.set", zarr="x.zarr", groups=())

        with pytest.raises(IndexError_, match="no channel groups"):
            empty.group()

    def test_an_explicit_group_is_honored_rather_than_silently_replaced(self) -> None:
        """A caller that resolved an ambiguity must get the group it named.

        Two groups at genuinely different rates, so a reader that ignores the argument and
        falls back to `store.group()` raises on the ambiguity instead of quietly returning
        the wrong rate.
        """
        base = _read(_document())[0].stores[0].groups[0]
        fast = replace(base, name="eeg_500hz", rate=500.0)
        store = Store(path=STORE0_PATH, zarr=STORE0_ZARR, groups=(base, fast))
        index = _read(_document())[0]

        assert index.level0_url(store, fast).endswith("/eeg_500hz/0")
        assert index.level0_url(store, base).endswith(f"/{GROUP0_NAME}/0")
        assert index.view_url(store, 1, fast).endswith("/eeg_500hz/view/1")

    def test_selects_by_name_among_groups_at_different_rates(self) -> None:
        base = _read(_document())[0].stores[0].groups[0]
        fast = replace(base, name="eeg_500hz", rate=500.0)
        store = Store(path=STORE0_PATH, zarr=STORE0_ZARR, groups=(base, fast))

        assert store.group("eeg_500hz").rate == 500.0
        assert store.group(GROUP0_NAME).rate == GROUP0_RATE

    def test_names_what_is_available_when_the_name_is_wrong(self) -> None:
        index, _ = _read(_document())
        store = index.stores[0]

        with pytest.raises(IndexError_, match=store.groups[0].name):
            store.group("eeg_nonexistent")


class TestLookup:
    def test_finds_a_store_by_its_source_path(self) -> None:
        index, _ = _read(_document())

        assert index.store(STORE1_PATH) is index.stores[1]
        assert index.store(STORE0_PATH).zarr == STORE0_ZARR

    def test_missing_store_names_the_dataset(self) -> None:
        index, _ = _read(_document())

        with pytest.raises(IndexError_, match=LIVE_DATASET):
            index.store("sub-nobody/eeg/nothing.set")


class TestRangeHeaders:
    def test_suffix_range_for_the_shard_index(self) -> None:
        """Zarr's sharding codec reads its chunk index from the tail of the object."""
        assert _range_header(None, 128) == "bytes=-128"

    def test_closed_open_and_absent_ranges(self) -> None:
        assert _range_header(0, 255) == "bytes=0-255"
        assert _range_header(1024, None) == "bytes=1024-"
        assert _range_header(None, None) is None


@pytest.mark.network
class TestAgainstTheLiveHost:
    """Reaches zarr.nemar.org. These are the tests that can actually fail when the
    contract changes underneath this reader, which is the point of having them."""

    def test_reads_the_live_index(self) -> None:
        index = asyncio.run(read_index(LIVE_DATASET))

        assert index.dataset_id == LIVE_DATASET
        assert index.format_version == 3
        assert index.contract_base == f"https://zarr.nemar.org/{LIVE_DATASET}/zarr/"
        assert index.store_count > 0
        assert index.stores

    def test_a_level0_url_built_from_the_live_index_serves_array_metadata(self) -> None:
        """End to end: read the index, build a path from its own layout, fetch what is
        there. Proves the templates resolve to something real rather than merely to a
        well-formed string."""
        from eegprep_lean.transport import default_transport

        async def run() -> dict:
            transport = default_transport()
            index = await read_index(LIVE_DATASET, transport=transport)
            url = index.level0_url(index.stores[0])
            response = await transport.get(f"{url}/zarr.json")
            return json.loads(response.body)

        metadata = asyncio.run(run())

        assert len(metadata["shape"]) == 2, "a level-0 EEG array is channels by samples"
        assert metadata["data_type"] == "int16"
        # Axis order, not just rank: a transposed array also has rank 2, and reading it as
        # channels-by-samples would return one channel's whole recording as a time slice.
        channels, samples = metadata["shape"]
        assert channels == GROUP0_CHANNELS, "first axis is channels"
        assert samples == GROUP0_SAMPLES, "second axis is samples"
        # Physical values need these; a reader returning digital counts is wrong in a way
        # that raises nothing, so check they are usable and not a degenerate identity.
        scale = metadata["attributes"]["scale"]
        offset = metadata["attributes"]["offset"]
        assert len(scale) == channels
        assert len(offset) == channels
        assert all(s > 0 for s in scale), "a zero or negative scale cannot convert to volts"
        assert any(o != 0 for o in offset), "an all-zero offset would make the field pointless"

    def test_the_host_honors_the_suffix_range_the_sharding_codec_needs(self) -> None:
        """If suffix ranges stop working, every chunk read fails and metadata still
        passes, so this is checked directly rather than inferred."""
        from eegprep_lean.transport import default_transport

        async def run() -> Response:
            transport = default_transport()
            index = await read_index(LIVE_DATASET, transport=transport)
            store = index.stores[0]
            url = index.level0_url(store)
            return await transport.get(f"{url}/c/0/0", start=None, end=128)

        response = asyncio.run(run())

        assert response.status == 206, "a 200 means the range was ignored and a whole shard is coming"
        assert len(response.body) == 128
