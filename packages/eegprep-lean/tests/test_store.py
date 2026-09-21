"""Tests for the Zarr store and the window reader.

Needs the ``zarr`` extra, so the whole module skips without it: the base install is meant
to work with nothing else present, and a hard import here would make that untrue.

The byte-range translation is tested against a real HTTP server rather than by asserting
on a computed header, because the failure it guards is an off-by-one that decodes cleanly
and returns wrong numbers. A test that checks the arithmetic against the same arithmetic
would not notice.
"""

from __future__ import annotations

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

zarr = pytest.importorskip("zarr", reason="needs the zarr extra")
np = pytest.importorskip("numpy", reason="needs the zarr extra")

from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest  # noqa: E402
from zarr.core.buffer import default_buffer_prototype  # noqa: E402

from eegprep_lean.index import IndexError_  # noqa: E402
from eegprep_lean.store import NemarHttpStore, ReadOnlyStoreError  # noqa: E402
from eegprep_lean.window import EXPECTED_FORMULA, Window, to_physical  # noqa: E402

BODY = bytes(range(256))


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        if self.path.endswith("/absent"):
            self.send_response(404)
            self.end_headers()
            return
        spec = (self.headers.get("Range") or "").removeprefix("bytes=")
        if spec.startswith("-"):
            chunk, status = BODY[-int(spec[1:]) :], 206
        elif spec.endswith("-") and spec[:-1]:
            chunk, status = BODY[int(spec[:-1]) :], 206
        elif "-" in spec:
            first, _, last = spec.partition("-")
            chunk, status = BODY[int(first) : int(last) + 1], 206
        else:
            chunk, status = BODY, 200
        self.send_response(status)
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)

    def log_message(self, *args: object) -> None:
        """Quiet."""


@pytest.fixture(scope="module")
def store():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield NemarHttpStore(f"http://127.0.0.1:{server.server_port}/base")
    finally:
        server.shutdown()
        server.server_close()


def _get(store: NemarHttpStore, key: str, byte_range=None) -> bytes | None:
    buffer = asyncio.run(store.get(key, default_buffer_prototype(), byte_range))
    return None if buffer is None else buffer.to_bytes()


class TestByteRanges:
    def test_range_request_is_half_open_and_http_is_closed(self, store: NemarHttpStore) -> None:
        """RangeByteRequest excludes `end`; `Range: bytes=a-b` includes `b`.

        Translating without losing one returns an extra byte at every chunk boundary,
        which decodes without complaint and yields wrong values.
        """
        got = _get(store, "obj", RangeByteRequest(10, 20))

        assert got == BODY[10:20]
        assert len(got) == 10

    def test_offset_request_returns_the_rest(self, store: NemarHttpStore) -> None:
        assert _get(store, "obj", OffsetByteRequest(200)) == BODY[200:]

    def test_suffix_request_returns_the_tail(self, store: NemarHttpStore) -> None:
        """The form the sharding codec reads its chunk index with."""
        assert _get(store, "obj", SuffixByteRequest(16)) == BODY[-16:]

    def test_no_range_returns_everything(self, store: NemarHttpStore) -> None:
        assert _get(store, "obj") == BODY

    def test_an_unknown_byte_request_type_is_refused(self, store: NemarHttpStore) -> None:
        with pytest.raises(TypeError, match="unsupported byte request"):
            store._http_range(object())


class TestAbsence:
    def test_a_missing_key_is_none_not_an_exception(self, store: NemarHttpStore) -> None:
        """Zarr probes for keys that need not exist, the sharding codec routinely.

        Raising would turn an ordinary absence into a failed read.
        """
        assert _get(store, "absent") is None

    def test_exists_reports_both_ways(self, store: NemarHttpStore) -> None:
        assert asyncio.run(store.exists("obj")) is True
        assert asyncio.run(store.exists("absent")) is False


class TestReadOnly:
    def test_declares_itself_read_only(self, store: NemarHttpStore) -> None:
        assert store.supports_writes is False
        assert store.supports_deletes is False

    def test_writing_and_deleting_raise(self, store: NemarHttpStore) -> None:
        buffer = default_buffer_prototype().buffer.from_bytes(b"x")

        with pytest.raises(ReadOnlyStoreError):
            asyncio.run(store.set("obj", buffer))
        with pytest.raises(ReadOnlyStoreError):
            asyncio.run(store.delete("obj"))

    def test_listing_is_unsupported_rather_than_empty(self, store: NemarHttpStore) -> None:
        """Anonymous listing is denied on the bucket, so there is nothing to enumerate.

        Declaring support and yielding nothing would let a caller conclude a dataset is
        empty when it is only unlistable.
        """
        assert store.supports_listing is False

        async def drain() -> None:
            async for _ in store.list():
                pass

        with pytest.raises(NotImplementedError, match="cannot be listed"):
            asyncio.run(drain())

    def test_equality_and_hashing_follow_the_base_url(self) -> None:
        a, b = NemarHttpStore("https://x/y"), NemarHttpStore("https://x/y/")

        assert a == b
        assert hash(a) == hash(b)
        assert a != NemarHttpStore("https://x/z")


class TestPhysicalConversion:
    ATTRS = {
        "physical_formula": EXPECTED_FORMULA,
        "scale": [2.0, 10.0, 100.0],
        "offset": [1.0, -5.0, 0.0],
    }

    def test_each_channel_uses_its_own_scale_and_offset(self) -> None:
        """Per channel, not global. Applying one channel's constants to all of them
        returns numbers of the right shape and the wrong magnitude."""
        digital = np.array([[1, 2], [1, 2], [1, 2]], dtype=np.int16)

        physical = to_physical(digital, self.ATTRS, [0, 1, 2])

        assert physical.tolist() == [[3.0, 5.0], [5.0, 15.0], [100.0, 200.0]]

    def test_constants_follow_the_channels_actually_read(self) -> None:
        """A window of channels 2 and 0 must use those two channels' constants, in that
        order, not the first two rows of the arrays."""
        digital = np.array([[1, 1], [1, 1]], dtype=np.int16)

        physical = to_physical(digital, self.ATTRS, [2, 0])

        assert physical.tolist() == [[100.0, 100.0], [3.0, 3.0]]

    def test_a_different_declared_formula_is_refused(self) -> None:
        """Applying this formula to an array that declares another one returns plausible
        numbers, so it refuses rather than guessing."""
        attrs = dict(self.ATTRS, physical_formula="physical = (digital - offset) / scale")

        with pytest.raises(IndexError_, match="declares its conversion"):
            to_physical(np.zeros((1, 1), dtype=np.int16), attrs, [0])


class TestWindowGeometry:
    def test_times_are_absolute_in_the_recording_not_the_window(self) -> None:
        """A window starting at sample 500 of a 250 Hz recording begins at t=2 s.

        Times relative to the window would misplace every event annotation on a plot.
        """
        window = Window(
            data=np.zeros((1, 4)),
            channels=(0,),
            start_sample=500,
            rate=250.0,
            group_name="eeg_250hz",
            physical=True,
        )

        assert window.n_samples == 4
        assert window.duration_s == pytest.approx(0.016)
        assert window.times_s[0] == pytest.approx(2.0)
        assert window.times_s[-1] == pytest.approx(2.012)


@pytest.mark.network
class TestAgainstTheLiveArchive:
    """Reads real signal from a real sharded store. This is the test that would notice
    if the store geometry, the codec chain or the conversion changed underneath us."""

    def test_reads_a_window_of_real_signal_in_physical_units(self) -> None:
        from eegprep_lean import read_index
        from eegprep_lean.window import read_window

        async def run():
            index = await read_index("nm000103")
            store = index.stores[0]
            physical = await read_window(index, store, start_sample=2500, n_samples=500, channels=[0, 1, 2])
            digital = await read_window(
                index, store, start_sample=2500, n_samples=500, channels=[0, 1, 2], physical=False
            )
            return physical, digital

        physical, digital = asyncio.run(run())

        assert physical.data.shape == (3, 500)
        assert physical.rate == 250.0
        assert physical.duration_s == pytest.approx(2.0)
        # Read from sample 2500 at 250 Hz, so the window starts ten seconds in.
        assert physical.times_s[0] == pytest.approx(10.0)

        assert digital.data.dtype == np.int16, "physical=False returns the stored counts"
        assert physical.data.dtype == np.float64
        # The conversion must actually have happened. Equal arrays would mean scale 1 and
        # offset 0 were applied, which is the silent failure this guards.
        assert not np.allclose(physical.data, digital.data)
        assert np.isfinite(physical.data).all()

    def test_only_the_requested_window_is_fetched(self) -> None:
        """The array is sharded and 129 by 43034; a reader that pulled the whole store
        would still return the right window, just after moving megabytes."""
        from eegprep_lean import read_index
        from eegprep_lean.transport import UrllibTransport

        class Counting(UrllibTransport):
            def __init__(self) -> None:
                super().__init__(timeout_s=60)
                self.bytes_read = 0

            async def get(self, url: str, *, start=None, end=None):
                response = await super().get(url, start=start, end=end)
                self.bytes_read += len(response.body)
                return response

        async def run() -> int:
            from eegprep_lean.window import read_window

            transport = Counting()
            index = await read_index("nm000103", transport=transport)
            store = index.stores[0]
            before = transport.bytes_read
            await read_window(index, store, start_sample=0, n_samples=250, channels=[0], transport=transport)
            return transport.bytes_read - before

        signal_bytes = asyncio.run(run())

        # The whole shard is over 7 MB. One second of one channel must cost far less,
        # even counting the metadata and shard-index reads that precede it.
        assert 0 < signal_bytes < 2_000_000, f"read {signal_bytes} bytes for a one-second window"
