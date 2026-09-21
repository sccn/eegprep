"""Tests for `read_window`, against a real Zarr store served over real HTTP.

Needs the ``zarr`` extra.

Everything else offline either builds a :class:`Window` by hand or tests a piece
``read_window`` calls. Nothing exercised the function that composes one, so a mutation
that carried the channel's unit onto stored integer counts survived every offline tier
and was caught only by the live test (issue #410).

The store here is built by zarr itself and served by a real HTTP server that honors
``Range``, so the reader under test is the real one end to end: its transport, its byte
ranges, its store, the codec pipeline, the conversion and the composition. Nothing is
stood in for. A handler that ignored ``Range`` would not do: the transport refuses a
ranged request answered in full, which is the behavior that keeps a silently truncated
read from looking like a short window.
"""

from __future__ import annotations

import asyncio
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

zarr = pytest.importorskip("zarr", reason="needs the zarr extra")
np = pytest.importorskip("numpy", reason="needs the zarr extra")

from zarr.storage import LocalStore  # noqa: E402

from eegprep_lean.channels import read_group_metadata  # noqa: E402
from eegprep_lean.index import ChannelGroup, DatasetIndex, IndexError_, Store  # noqa: E402
from eegprep_lean.window import read_window  # noqa: E402

# The store's geometry and constants, written out so an assertion can disagree with the
# code that built it.
N_CHANNELS = 3
N_SAMPLES = 1000
RATE = 250.0
ORIGINAL_RATE = 500.0
GROUP = "eeg_250hz"
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
    requested: list[str]

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        type(self).requested.append(self.path)
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
    array = group.create_array("0", shape=(N_CHANNELS, N_SAMPLES), dtype="int16", chunks=(N_CHANNELS, 250))
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

    requested: list[str] = []
    handler = type("Handler", (_RangeHandler,), {"root": root, "requested": requested})
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    index = DatasetIndex(
        dataset_id="xx099999",
        format_version=3,
        contract_base=f"http://127.0.0.1:{server.server_port}/",
        store_count=1,
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

    def test_starting_past_the_end_is_refused(self, served) -> None:
        index, store, _ = served

        with pytest.raises(IndexError_, match="past the end"):
            read(index, store, start_sample=N_SAMPLES, n_samples=1)

    @pytest.mark.parametrize("n_samples", [0, -1])
    def test_a_non_positive_length_is_refused(self, served, n_samples: int) -> None:
        index, store, _ = served

        with pytest.raises(IndexError_, match="must be positive"):
            read(index, store, start_sample=0, n_samples=n_samples)


class TestFetching:
    def test_passing_metadata_skips_the_group_fetch(self, served) -> None:
        """One small document per call otherwise. A caller reading many windows from one
        recording holds it already, and this is the parameter that lets them say so."""
        index, store, requested = served
        metadata = asyncio.run(read_group_metadata(index, store))

        group_doc = f"/{ZARR_PATH}/{GROUP}/zarr.json"
        requested.clear()
        read(index, store, start_sample=0, n_samples=2, metadata=metadata)
        without = requested.count(group_doc)

        requested.clear()
        read(index, store, start_sample=0, n_samples=2)
        with_fetch = requested.count(group_doc)

        assert without == 0, "metadata was supplied, so the group document was not needed"
        assert with_fetch == 1
