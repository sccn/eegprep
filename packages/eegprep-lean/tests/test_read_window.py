"""Tests for `read_window`, against a real Zarr store served over real HTTP.

Needs the ``zarr`` extra.

Everything else offline either builds a :class:`Window` by hand or tests a piece
``read_window`` calls. Nothing exercised the function that composes one, so a mutation
that carried the channel's unit onto stored integer counts survived every offline tier
and was caught only by the live test (issue #410).

The store here is built by zarr itself and served by a real HTTP server that honors
``Range``, so the reader under test is the real one end to end: its transport, its byte
ranges, its store, the codec pipeline, the conversion and the composition. Nothing is
stood in for.

**The array is sharded, and that is load-bearing rather than incidental.** An unsharded
array stores each chunk as its own object, so zarr fetches whole objects and never asks
for a byte range at all. A fixture built that way exercises none of the range handling
while looking like it does: against one, the half-open to closed adjustment in
:meth:`~eegprep_lean.store.NemarHttpStore._http_range` survives deletion and every test
here still passes. Production level-0 arrays are sharded, so the shape here matches them,
and :class:`TestTheReadActuallyUsesByteRanges` asserts the ranges are issued rather than
trusting this paragraph.

What this file does **not** cover, deliberately: the transport's refusal of a ranged
request answered in full. That is the transport's own contract, it needs a server that
ignores ``Range``, and ``test_transport.py`` owns it.
"""

from __future__ import annotations

import asyncio
import pathlib
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

zarr = pytest.importorskip("zarr", reason="needs the zarr extra")
np = pytest.importorskip("numpy", reason="needs the zarr extra")

from zarr.storage import LocalStore  # noqa: E402

from eegprep_lean.channels import read_group_metadata  # noqa: E402
from eegprep_lean.index import ChannelGroup, DatasetIndex, IndexError_, Store  # noqa: E402
from eegprep_lean.transport import FetchTransport, set_default_transport  # noqa: E402
from eegprep_lean.window import read_window  # noqa: E402

# The store's geometry and constants, written out so an assertion can disagree with the
# code that built it.
N_CHANNELS = 3
N_SAMPLES = 1000
RATE = 250.0
ORIGINAL_RATE = 500.0
GROUP = "eeg_250hz"
CHUNK_SAMPLES = 125
SHARD_SAMPLES = 500

# A second recording carrying two groups, so the `group=` parameter is observable. A
# store with one group resolves the same way whether or not the parameter is honored.
SECOND_STORE_PATH = "sub-02/eeg/sub-02_task-test_eeg.set"
SECOND_ZARR = "rec2.zarr"
SLOW_GROUP, SLOW_RATE = "eeg_125hz", 125.0
FAST_GROUP, FAST_RATE = "eeg_250hz", 250.0
ZARR_PATH = "rec.zarr"
SCALE = [2.0, 10.0, 100.0]
OFFSET = [1.0, -5.0, 0.0]
LABELS = ("E1", "E2", "E3")
UNIT = "uV"


def _digital(channel: int, sample: int) -> int:
    """The value written at one cell. Distinct per cell, so a transposed or offset read
    produces numbers that are wrong rather than merely rearranged."""
    return channel * N_SAMPLES + sample


class _RangeHandler(BaseHTTPRequestHandler):
    """Serves files and honors Range, which SimpleHTTPRequestHandler does not."""

    root: pathlib.Path
    #: (path, Range header or None) per request, so a test can assert what was asked for
    #: rather than only what came back.
    requested: list[tuple[str, str | None]]

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        type(self).requested.append((self.path, self.headers.get("Range")))
        target = self.root / self.path.lstrip("/")
        if not target.is_file():
            self.send_response(404)
            self.end_headers()
            return
        body = target.read_bytes()
        spec = (self.headers.get("Range") or "").removeprefix("bytes=")
        status = 200
        if spec:
            status = 206
            if spec.startswith("-"):
                body = body[-int(spec[1:]) :]
            elif spec.endswith("-"):
                body = body[int(spec[:-1]) :]
            else:
                first, _, last = spec.partition("-")
                body = body[int(first) : int(last) + 1]
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        """Quiet."""


@pytest.fixture(scope="module")
def served(tmp_path_factory) -> tuple[DatasetIndex, Store, list[str]]:
    root = tmp_path_factory.mktemp("served")
    group = zarr.create_group(store=LocalStore(root / ZARR_PATH), overwrite=True).create_group(GROUP)
    group.attrs.update(
        {
            "modality": "EEG",
            "rate": RATE,
            "original_rate": ORIGINAL_RATE,
            "n_channels": N_CHANNELS,
            "channels": [
                {
                    "label": LABELS[i],
                    "unit": UNIT,
                    "channel_type": "EEG",
                    "row_index": i,
                    "original_rate": ORIGINAL_RATE,
                    "target_rate": RATE,
                    "usable_for_inference": True,
                }
                for i in range(N_CHANNELS)
            ],
        }
    )
    # Sharded, like production. Two shards of four inner chunks each, so a window read
    # takes a suffix range for the shard index and then a range for the inner chunk.
    array = group.create_array(
        "0",
        shape=(N_CHANNELS, N_SAMPLES),
        dtype="int16",
        chunks=(N_CHANNELS, CHUNK_SAMPLES),
        shards=(N_CHANNELS, SHARD_SAMPLES),
    )
    array[:] = np.array([[_digital(c, s) for s in range(N_SAMPLES)] for c in range(N_CHANNELS)], dtype=np.int16)
    array.attrs.update(
        {
            "physical_formula": "physical = digital * scale + offset",
            "scale": SCALE,
            "offset": OFFSET,
            "level": 0,
            "rate": RATE,
            "source_rate_hz": ORIGINAL_RATE,
        }
    )

    # A second recording with two groups at different rates, so `group=` is observable.
    second = zarr.create_group(store=LocalStore(root / SECOND_ZARR), overwrite=True)
    for name, rate in ((SLOW_GROUP, SLOW_RATE), (FAST_GROUP, FAST_RATE)):
        g = second.create_group(name)
        g.attrs.update(
            {
                "modality": "EEG",
                "rate": rate,
                "original_rate": ORIGINAL_RATE,
                "n_channels": 1,
                "channels": [
                    {
                        "label": "E1",
                        "unit": UNIT,
                        "channel_type": "EEG",
                        "row_index": 0,
                        "original_rate": ORIGINAL_RATE,
                        "target_rate": rate,
                        "usable_for_inference": True,
                    }
                ],
            }
        )
        a = g.create_array("0", shape=(1, 100), dtype="int16", chunks=(1, 50))
        a[:] = np.arange(100, dtype=np.int16).reshape(1, 100)
        a.attrs.update(
            {
                "physical_formula": "physical = digital * scale + offset",
                "scale": [1.0],
                "offset": [0.0],
                "level": 0,
                "rate": rate,
                "source_rate_hz": ORIGINAL_RATE,
            }
        )

    requested: list[tuple[str, str | None]] = []
    handler = type("Handler", (_RangeHandler,), {"root": root, "requested": requested})
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    index = DatasetIndex(
        dataset_id="xx099999",
        format_version=3,
        contract_base=f"http://127.0.0.1:{server.server_port}/",
        store_count=2,
        stores=(
            Store(
                path="sub-01/eeg/sub-01_task-test_eeg.set",
                zarr=ZARR_PATH,
                groups=(
                    ChannelGroup(
                        name=GROUP,
                        modality="EEG",
                        rate=RATE,
                        n_channels=N_CHANNELS,
                        n_samples=N_SAMPLES,
                        n_view_levels=0,
                    ),
                ),
            ),
            Store(
                path=SECOND_STORE_PATH,
                zarr=SECOND_ZARR,
                groups=tuple(
                    ChannelGroup(
                        name=name,
                        modality="EEG",
                        rate=rate,
                        n_channels=1,
                        n_samples=100,
                        n_view_levels=0,
                    )
                    for name, rate in ((SLOW_GROUP, SLOW_RATE), (FAST_GROUP, FAST_RATE))
                ),
            ),
        ),
    )
    try:
        yield index, index.stores[0], requested
    finally:
        server.shutdown()
        server.server_close()


def read(index: DatasetIndex, store: Store, **kwargs):
    return asyncio.run(read_window(index, store, **kwargs))


class TestTheValuesThemselves:
    def test_physical_units_use_each_channels_own_constants(self, served) -> None:
        """digital * scale + offset, per channel. Applying one channel's constants to
        all of them returns the right shape and the wrong magnitude."""
        index, store, _ = served

        window = read(index, store, start_sample=0, n_samples=2)

        assert window.data.tolist() == [
            [_digital(0, 0) * 2.0 + 1.0, _digital(0, 1) * 2.0 + 1.0],
            [_digital(1, 0) * 10.0 - 5.0, _digital(1, 1) * 10.0 - 5.0],
            [_digital(2, 0) * 100.0, _digital(2, 1) * 100.0],
        ]

    def test_physical_false_returns_the_stored_counts(self, served) -> None:
        index, store, _ = served

        window = read(index, store, start_sample=10, n_samples=3, physical=False)

        assert window.data.dtype == np.int16
        assert window.data.tolist() == [[_digital(c, s) for s in (10, 11, 12)] for c in range(3)]


class TestWhatTheWindowSaysAboutItself:
    def test_stored_counts_never_carry_the_channels_unit(self, served) -> None:
        """Issue #410's surviving mutation. int16 counts are not in the channel's unit,
        and a window that claimed otherwise would make every downstream label wrong by
        the scale factor while the numbers still looked like signal."""
        index, store, _ = served

        assert read(index, store, start_sample=0, n_samples=4).unit == UNIT
        assert read(index, store, start_sample=0, n_samples=4, physical=False).unit is None

    def test_labels_and_acquisition_rate_come_from_the_group(self, served) -> None:
        index, store, _ = served

        window = read(index, store, start_sample=0, n_samples=4)

        assert window.labels == LABELS
        assert window.rate == RATE
        assert window.original_rate == ORIGINAL_RATE
        assert window.was_resampled is True

    def test_times_are_absolute_in_the_recording(self, served) -> None:
        index, store, _ = served

        window = read(index, store, start_sample=500, n_samples=4)

        assert window.times_s[0] == pytest.approx(2.0)

    def test_it_records_which_group_it_read_and_whether_it_converted(self, served) -> None:
        """Both are part of the Window a caller reads back, and a plot titles itself
        from the first."""
        index, store, _ = served

        assert read(index, store, start_sample=0, n_samples=2).group_name == GROUP
        assert read(index, store, start_sample=0, n_samples=2).physical is True
        assert read(index, store, start_sample=0, n_samples=2, physical=False).physical is False

    def test_an_explicit_group_is_honored(self, served) -> None:
        """Against a store with two groups, so the parameter is observable.

        Passing the group a store would have resolved anyway proves nothing: dropping
        the override entirely leaves such a test passing. These are the same recording
        at two rates, so taking the wrong one returns real data at the wrong sampling
        rate, which is the kind of wrong that survives a plot.
        """
        index, _, _ = served
        store = index.store(SECOND_STORE_PATH)

        slow = read(index, store, start_sample=0, n_samples=2, group=store.group(SLOW_GROUP))
        fast = read(index, store, start_sample=0, n_samples=2, group=store.group(FAST_GROUP))

        assert (slow.group_name, slow.rate) == (SLOW_GROUP, SLOW_RATE)
        assert (fast.group_name, fast.rate) == (FAST_GROUP, FAST_RATE)

    def test_a_store_with_several_groups_refuses_to_guess(self, served) -> None:
        """They are the same recording at different rates, so there is no sensible
        default and picking the first would be silently wrong."""
        index, _, _ = served
        store = index.store(SECOND_STORE_PATH)

        with pytest.raises(IndexError_, match="channel groups and no name was given"):
            read(index, store, start_sample=0, n_samples=2)


class TestChannelSelection:
    def test_non_adjacent_channels_come_back_in_the_order_asked_for(self, served) -> None:
        """read_window reads the contiguous span covering the channels and takes rows
        afterwards, because zarr's async getitem does basic indexing only. Getting that
        reindexing wrong returns real data from the wrong channels."""
        index, store, _ = served

        window = read(index, store, start_sample=0, n_samples=2, channels=[2, 0], physical=False)

        assert window.channels == (2, 0)
        assert window.data.tolist() == [[_digital(2, 0), _digital(2, 1)], [_digital(0, 0), _digital(0, 1)]]
        assert window.labels == ("E3", "E1")

    def test_a_slice_selects_channels_too(self, served) -> None:
        """`channels` takes a sequence, a slice, or None. The slice branch was reachable
        from the public API and tested nowhere."""
        index, store, _ = served

        window = read(index, store, start_sample=0, n_samples=2, channels=slice(1, 3), physical=False)

        assert window.channels == (1, 2)
        assert window.data.tolist() == [[_digital(1, 0), _digital(1, 1)], [_digital(2, 0), _digital(2, 1)]]

    def test_no_channels_means_all_of_them(self, served) -> None:
        index, store, _ = served

        window = read(index, store, start_sample=0, n_samples=2, channels=None)

        assert window.channels == tuple(range(N_CHANNELS))

    def test_a_channel_outside_the_group_is_refused(self, served) -> None:
        index, store, _ = served

        with pytest.raises(IndexError_, match="outside 0.."):
            read(index, store, start_sample=0, n_samples=2, channels=[99])


class TestBoundaries:
    def test_a_window_running_past_the_end_is_clamped(self, served) -> None:
        """Asking for the last ten seconds of a recording that has eight is ordinary,
        and the window reports what it actually holds.

        This pins the contract, not one line: zarr's basic indexing clamps an
        out-of-range stop the way numpy does, so deleting `read_window`'s own clamp
        leaves this passing. The explicit clamp stays because a documented behavior
        should not rest on another library's slicing semantics, and this test would
        still catch the day that changes.
        """
        index, store, _ = served

        window = read(index, store, start_sample=N_SAMPLES - 5, n_samples=100)

        assert window.n_samples == 5

    def test_a_negative_start_is_refused(self, served) -> None:
        """Its own branch with its own message. Python would read a negative index from
        the end of the recording rather than failing."""
        index, store, _ = served

        with pytest.raises(IndexError_, match="must not be negative"):
            read(index, store, start_sample=-1, n_samples=2)

    def test_starting_past_the_end_is_refused(self, served) -> None:
        index, store, _ = served

        with pytest.raises(IndexError_, match="past the end"):
            read(index, store, start_sample=N_SAMPLES, n_samples=1)

    @pytest.mark.parametrize("n_samples", [0, -1])
    def test_a_non_positive_length_is_refused(self, served, n_samples: int) -> None:
        index, store, _ = served

        with pytest.raises(IndexError_, match="must be positive"):
            read(index, store, start_sample=0, n_samples=n_samples)


class TestTheReadActuallyUsesByteRanges:
    """Without this the file's central claim is unenforced.

    An unsharded fixture never provokes a range request, and then the byte-range
    translation and the transport's range checking are dead code under test: both survive
    deletion while every other test here still passes.
    """

    def test_a_window_read_issues_range_requests(self, served) -> None:
        index, store, requested = served
        requested.clear()

        read(index, store, start_sample=0, n_samples=4)

        ranges = [spec for _, spec in requested if spec]
        assert ranges, "no request carried a Range header, so no byte-range code ran"

    def test_the_shard_index_is_read_as_a_suffix_range(self, served) -> None:
        """The sharding codec reads the chunk index from the end of the shard. That is
        the one form whose translation carries no off-by-one, and the one a client must
        support to read a sharded store at all."""
        index, store, requested = served
        requested.clear()

        read(index, store, start_sample=0, n_samples=4)

        assert any(spec and spec.startswith("bytes=-") for _, spec in requested), (
            "the shard index was not read as a suffix range"
        )

    def test_the_inner_chunk_is_fetched_as_a_bounded_range(self, served) -> None:
        """A shard holds four inner chunks. Reading four samples takes one of them, by a
        closed range inside the shard, which is the whole reason the store speaks ranges
        and the one form whose translation has an off-by-one to get wrong."""
        index, store, requested = served
        requested.clear()

        read(index, store, start_sample=0, n_samples=4)

        shard = f"/{ZARR_PATH}/{GROUP}/0/c/0/0"
        closed = [
            spec.removeprefix("bytes=")
            for path, spec in requested
            if path == shard and spec and re.fullmatch(r"bytes=\d+-\d+", spec)
        ]
        assert closed, "the inner chunk was not fetched with a bounded range inside the shard"
        first, last = (int(v) for v in closed[0].split("-"))
        assert last >= first


class TestFetching:
    def test_passing_metadata_skips_the_group_fetch(self, served) -> None:
        """One small document per call otherwise. A caller reading many windows from one
        recording holds it already, and this is the parameter that lets them say so."""
        index, store, requested = served
        metadata = asyncio.run(read_group_metadata(index, store))

        group_doc = f"/{ZARR_PATH}/{GROUP}/zarr.json"
        requested.clear()
        read(index, store, start_sample=0, n_samples=2, metadata=metadata)
        without = sum(1 for path, _ in requested if path == group_doc)

        requested.clear()
        read(index, store, start_sample=0, n_samples=2)
        with_fetch = sum(1 for path, _ in requested if path == group_doc)

        assert without == 0, "metadata was supplied, so the group document was not needed"
        assert with_fetch == 1


class TestAHostTransport:
    def test_a_registered_transport_serves_a_read_that_names_none(self, served, host_fetch, host_default) -> None:
        """The browser case end to end. A sandboxed runtime registers its own client once,
        and a read written with no ``transport=`` argument, as NEMAR's recipe is, goes
        through it: shard index, inner chunk and metadata alike."""
        index, store, _ = served
        set_default_transport(FetchTransport(host_fetch))

        window = read(index, store, start_sample=0, n_samples=2, physical=False)

        assert window.data.tolist() == [[_digital(c, s) for s in (0, 1)] for c in range(N_CHANNELS)]
        ranges = [headers.get("Range", "") for _, headers in host_fetch.seen]
        assert any(spec.startswith("bytes=-") for spec in ranges), "the shard index did not go through the host"
        assert any(re.fullmatch(r"bytes=\d+-\d+", spec) for spec in ranges), "nor did the inner chunk"
