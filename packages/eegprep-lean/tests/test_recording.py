"""Tests for ``load_recording`` and ``index_sha256``: one whole recording, read in one call.

The offline tier serves a Zarr store built by zarr itself over real HTTP that honors
``Range``, the way ``test_read_window.py`` does, so the reader under test is the real one
end to end. The only stand-in is :class:`RoutingTransport`, a transport that answers the
production index address with the bytes of a local index document and sends every other
request over real HTTP to the local server. It is a stand-in for the one external address
(``zarr.nemar.org``) a test must not reach, and it records what was asked for.

``index_sha256`` is base tier (standard library only) and its tests need no zarr; the
tests that read signal skip without it.

The ``network`` tier reads the smallest recording of a live public dataset, because what
is under test there is conformance to a contract another team serves.
"""

from __future__ import annotations

import asyncio
import dataclasses
import gc
import hashlib
import json
import pathlib
import threading
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from eegprep_lean import IndexError_, Store, UrllibTransport, read_index
from eegprep_lean.index import INDEX_URL_TEMPLATE
from eegprep_lean.recording import index_sha256, load_recording
from eegprep_lean.transport import Response

DATASET = "xx099999"
N_CHANNELS = 3
N_SAMPLES = 1000
RATE = 250.0
SCALE = [2.0, 10.0, 100.0]
OFFSET = [1.0, -5.0, 0.0]
LABELS = ("E1", "E2", "E3")
FIRST_PATH = "sub-01/eeg/sub-01_task-test_eeg.set"
TWO_EEG_PATH = "sub-02/eeg/sub-02_task-test_eeg.set"
MIXED_PATH = "sub-03/meg/sub-03_task-test_meg.fif"
RAGGED_PATH = "sub-04/eeg/sub-04_task-test_eeg.set"
# 700 samples in chunks of 125 and shards of 500: the second shard holds 200 samples, whose
# last chunk holds 75. Neither the shard nor the chunk grid divides the length.
RAGGED_SAMPLES = 700
# Written out here and not imported from the code: the peak bytes per channel-sample that
# max_bytes is counted in. Float64 window (8) + float64 dictionary data (8) + one channel's
# worth of times (8).
PEAK = 24
# The fixture's group names, for the stores that have several.
SLOW, FAST = "eeg_125hz", "eeg_250hz"


def _ragged(channel: int, sample: int) -> int:
    """Distinct per cell of the ragged store, and in range for int16."""
    return channel * 1000 + sample


def _digital(channel: int, sample: int) -> int:
    """Distinct per cell, so a transposed or truncated read is wrong rather than rearranged."""
    return channel * N_SAMPLES + sample


class _Handler(BaseHTTPRequestHandler):
    """Serves files and in-memory documents, honors Range, and records each request."""

    root: pathlib.Path
    documents: dict[str, bytes]
    requested: list[tuple[str, str | None, str | None]]

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        type(self).requested.append((self.path, self.headers.get("Range"), self.headers.get("User-Agent")))
        if self.path in self.documents:
            body = self.documents[self.path]
        else:
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


def _channel_entries(n: int, unit: str, rate: float, channel_type: str) -> list[dict]:
    return [
        {
            "label": LABELS[i] if n == N_CHANNELS else f"C{i + 1}",
            "unit": unit,
            "channel_type": channel_type,
            "row_index": i,
            "original_rate": 500.0,
            "target_rate": rate,
            "usable_for_inference": True,
        }
        for i in range(n)
    ]


def _write_group(root, name: str, *, modality: str, rate: float, unit: str, data, scale, offset, sharded=False):
    import numpy as np

    n_channels, n_samples = data.shape
    group = root.create_group(name)
    group.attrs.update(
        {
            "modality": modality,
            "rate": rate,
            "original_rate": 500.0,
            "n_channels": n_channels,
            "channels": _channel_entries(n_channels, unit, rate, modality),
        }
    )
    kwargs = {"chunks": (n_channels, 125), "shards": (n_channels, 500)} if sharded else {"chunks": (n_channels, 50)}
    array = group.create_array("0", shape=data.shape, dtype="int16", **kwargs)
    array[:] = data.astype(np.int16)
    array.attrs.update(
        {
            "physical_formula": "physical = digital * scale + offset",
            "scale": scale,
            "offset": offset,
            "level": 0,
            "rate": rate,
            "source_rate_hz": 500.0,
        }
    )


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """(index document bytes, base URL, requests) for a local copy of three recordings."""
    zarr = pytest.importorskip("zarr", reason="needs the zarr extra")
    np = pytest.importorskip("numpy", reason="needs the zarr extra")
    from zarr.storage import LocalStore

    root = tmp_path_factory.mktemp("served")
    first = np.array([[_digital(c, s) for s in range(N_SAMPLES)] for c in range(N_CHANNELS)])
    _write_group(
        zarr.create_group(store=LocalStore(root / "rec.zarr"), overwrite=True),
        FAST,
        modality="EEG",
        rate=RATE,
        unit="uV",
        data=first,
        scale=SCALE,
        offset=OFFSET,
        sharded=True,
    )

    # Two EEG groups of the same recording at different rates, holding different values,
    # so the group that was read is visible in the numbers.
    two = zarr.create_group(store=LocalStore(root / "rec2.zarr"), overwrite=True)
    for name, rate, base in ((SLOW, 125.0, 0), (FAST, 250.0, 1000)):
        _write_group(
            two,
            name,
            modality="EEG",
            rate=rate,
            unit="uV",
            data=np.arange(100).reshape(1, 100) + base,
            scale=[1.0],
            offset=[0.0],
        )

    # An EEG group next to a MEG group: "the store's EEG group" has to skip the other.
    mixed = zarr.create_group(store=LocalStore(root / "rec3.zarr"), overwrite=True)
    _write_group(
        mixed,
        "meg_250hz",
        modality="MEG",
        rate=250.0,
        unit="T",
        data=np.full((2, 100), 7),
        scale=[1.0, 1.0],
        offset=[0.0, 0.0],
    )
    _write_group(
        mixed,
        FAST,
        modality="EEG",
        rate=250.0,
        unit="uV",
        data=np.arange(100).reshape(1, 100) + 5000,
        scale=[1.0],
        offset=[0.0],
    )

    _write_group(
        zarr.create_group(store=LocalStore(root / "rec4.zarr"), overwrite=True),
        FAST,
        modality="EEG",
        rate=RATE,
        unit="uV",
        data=np.array([[_ragged(c, s) for s in range(RAGGED_SAMPLES)] for c in range(N_CHANNELS)]),
        scale=SCALE,
        offset=OFFSET,
        sharded=True,
    )

    requested: list[tuple[str, str | None, str | None]] = []
    documents: dict[str, bytes] = {}
    handler = type("Handler", (_Handler,), {"root": root, "documents": documents, "requested": requested})
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_port}/"

    def group_entry(name: str, modality: str, rate: float, n_channels: int, n_samples: int) -> dict:
        return {
            "name": name,
            "modality": modality,
            "rate": rate,
            "n_channels": n_channels,
            "n_samples": n_samples,
            "n_view_levels": 0,
        }

    document = {
        "format": "nemar-zarr-index",
        "format_version": 3,
        "dataset_id": DATASET,
        "contract_base": base_url,
        "store_count": 4,
        "stores": [
            {
                "path": FIRST_PATH,
                "zarr": "rec.zarr",
                "groups": [group_entry(FAST, "EEG", RATE, N_CHANNELS, N_SAMPLES)],
            },
            {
                "path": TWO_EEG_PATH,
                "zarr": "rec2.zarr",
                "groups": [group_entry(SLOW, "EEG", 125.0, 1, 100), group_entry(FAST, "EEG", 250.0, 1, 100)],
            },
            {
                "path": MIXED_PATH,
                "zarr": "rec3.zarr",
                "groups": [group_entry("meg_250hz", "MEG", 250.0, 2, 100), group_entry(FAST, "EEG", 250.0, 1, 100)],
            },
            {
                "path": RAGGED_PATH,
                "zarr": "rec4.zarr",
                "groups": [group_entry(FAST, "EEG", RATE, N_CHANNELS, RAGGED_SAMPLES)],
            },
        ],
    }
    # Indented, with a trailing newline and non-sorted keys: re-serializing the parsed
    # document gives different bytes, so a digest of anything but the bytes read differs.
    body = (json.dumps(document, indent=1) + "\n").encode()
    documents["/index.json"] = body
    try:
        yield body, base_url, requested
    finally:
        server.shutdown()
        server.server_close()


class RoutingTransport:
    """Real HTTP to the local server, except the production index address, which is
    answered with the local index document. Records every URL it was asked for."""

    def __init__(self, index_body: bytes) -> None:
        self.index_body = index_body
        self.inner = UrllibTransport()
        self.urls: list[str] = []

    async def get(self, url: str, *, start: int | None = None, end: int | None = None) -> Response:
        self.urls.append(url)
        if url == INDEX_URL_TEMPLATE.format(dataset_id=DATASET):
            return Response(status=200, body=self.index_body)
        return await self.inner.get(url, start=start, end=end)


@pytest.fixture
def routed(served) -> RoutingTransport:
    return RoutingTransport(served[0])


@pytest.fixture
def local_index(served):
    body, base_url, _ = served
    return asyncio.run(read_index(DATASET, index_url=f"{base_url}index.json"))


class TestIndexSha256:
    def test_is_the_digest_of_the_exact_bytes_read(self, served) -> None:
        body, base_url, _ = served

        digest = index_sha256(DATASET, index_url=f"{base_url}index.json")

        assert digest == hashlib.sha256(body).hexdigest()
        # Not the digest of the parsed document: that is a different, and unprovable, value.
        assert digest != hashlib.sha256(json.dumps(json.loads(body)).encode()).hexdigest()

    def test_reads_through_the_given_transport(self, served, routed) -> None:
        body, _, _ = served

        digest = index_sha256(DATASET, transport=routed)

        assert routed.urls == [INDEX_URL_TEMPLATE.format(dataset_id=DATASET)]
        assert digest == hashlib.sha256(body).hexdigest()

    def test_sends_the_native_user_agent(self, served) -> None:
        """The hosts have been seen refusing ``Python-urllib``, so the digest must go
        through the package's own transport rather than a bare ``urlopen``."""
        _, base_url, requested = served
        requested.clear()

        index_sha256(DATASET, index_url=f"{base_url}index.json")

        assert [agent for _, _, agent in requested] == ["eegprep-lean"]

    def test_a_running_event_loop_is_refused_clearly_and_leaves_no_coroutine_behind(self, served) -> None:
        _, base_url, _ = served

        async def inside() -> None:
            with pytest.raises(RuntimeError, match="await read_index and read_window"):
                index_sha256(DATASET, index_url=f"{base_url}index.json")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            asyncio.run(inside())
            gc.collect()

        assert [str(w.message) for w in caught if "never awaited" in str(w.message)] == []


class TestLoadRecording:
    def test_returns_the_whole_recording_in_physical_units(self, served, local_index) -> None:
        window = load_recording(DATASET, FIRST_PATH, index=local_index)

        assert window.data.shape == (N_CHANNELS, N_SAMPLES)
        assert window.data[0, 0] == _digital(0, 0) * 2.0 + 1.0
        assert window.data[1, 999] == _digital(1, 999) * 10.0 - 5.0
        assert window.data[2, 500] == _digital(2, 500) * 100.0
        assert window.start_sample == 0
        assert window.physical is True
        assert (window.unit, window.rate, window.labels) == ("uV", RATE, LABELS)
        assert window.group_name == FAST

    def test_reads_the_index_through_the_transport_when_none_is_given(self, served, routed) -> None:
        window = load_recording(DATASET, FIRST_PATH, transport=routed)

        assert routed.urls[0] == INDEX_URL_TEMPLATE.format(dataset_id=DATASET)
        assert window.data.shape == (N_CHANNELS, N_SAMPLES)
        # The signal and the group metadata went through the same transport.
        assert any(url.endswith("/rec.zarr/eeg_250hz/0/zarr.json") for url in routed.urls)
        assert any(url.endswith("/rec.zarr/eeg_250hz/zarr.json") for url in routed.urls)

    def test_an_index_passed_in_is_not_fetched_again(self, served, local_index, routed) -> None:
        load_recording(DATASET, FIRST_PATH, index=local_index, transport=routed)

        assert INDEX_URL_TEMPLATE.format(dataset_id=DATASET) not in routed.urls
        assert routed.urls, "the signal was not read through the given transport"

    def test_a_store_with_one_eeg_group_among_others_reads_the_eeg_one(self, served, local_index) -> None:
        window = load_recording(DATASET, MIXED_PATH, index=local_index)

        assert window.group_name == FAST
        assert window.data.shape == (1, 100)
        assert window.data[0, 0] == 5000

    def test_two_eeg_groups_are_not_guessed_between(self, served, local_index) -> None:
        with pytest.raises(IndexError_, match=f"{SLOW}.*{FAST}"):
            load_recording(DATASET, TWO_EEG_PATH, index=local_index)

    def test_a_store_with_no_eeg_group_says_so_rather_than_reading_another_modality(self, local_index) -> None:
        """Only a MEG group: reading it would hand a magnetic field to a caller who asked
        for EEG, and the error has to say what the store does hold."""
        meg_only = Store(path=MIXED_PATH, zarr="rec3.zarr", groups=(local_index.store(MIXED_PATH).groups[0],))
        index = dataclasses.replace(local_index, stores=(meg_only,))

        with pytest.raises(IndexError_, match=r"no EEG channel group.*meg_250hz \(MEG\)"):
            load_recording(DATASET, MIXED_PATH, index=index)

    def test_an_explicit_group_is_honored(self, served, local_index) -> None:
        slow = load_recording(DATASET, TWO_EEG_PATH, group=SLOW, index=local_index)
        fast = load_recording(DATASET, TWO_EEG_PATH, group=FAST, index=local_index)

        assert (slow.group_name, slow.rate, slow.data[0, 0]) == (SLOW, 125.0, 0)
        assert (fast.group_name, fast.rate, fast.data[0, 0]) == (FAST, 250.0, 1000)

    def test_an_unknown_store_is_refused(self, served, local_index) -> None:
        with pytest.raises(IndexError_, match="no store for"):
            load_recording(DATASET, "sub-99/eeg/missing.set", index=local_index)

    def test_a_recording_over_max_bytes_is_refused_before_any_signal_is_read(self, served, local_index) -> None:
        _, _, requested = served
        requested.clear()
        needed = N_CHANNELS * N_SAMPLES * PEAK

        with pytest.raises(ValueError, match=str(needed)):
            load_recording(DATASET, FIRST_PATH, index=local_index, max_bytes=needed - 1)

        assert requested == [], "the size was checked after reading rather than before"

    def test_a_recording_exactly_max_bytes_is_read(self, served, local_index) -> None:
        needed = N_CHANNELS * N_SAMPLES * PEAK

        window = load_recording(DATASET, FIRST_PATH, index=local_index, max_bytes=needed)

        assert window.data.shape == (N_CHANNELS, N_SAMPLES)

    def test_the_cap_counts_peak_memory_and_not_only_the_decoded_array(self, served, local_index) -> None:
        """A cap of the float64 array alone (8 bytes per channel-sample) lets through a
        recording whose conversion peaks at three times that."""
        float64_array = N_CHANNELS * N_SAMPLES * 8

        with pytest.raises(ValueError, match="peak"):
            load_recording(DATASET, FIRST_PATH, index=local_index, max_bytes=float64_array)

    def test_the_whole_ragged_recording_comes_back_with_every_value(self, served, local_index) -> None:
        """700 samples in shards of 500 and chunks of 125, so the last shard and the last
        chunk are short. Shape alone would not notice a short chunk padded with fill
        values or a last shard dropped: every cell is checked."""
        np = pytest.importorskip("numpy")

        window = load_recording(DATASET, RAGGED_PATH, index=local_index)

        expected = np.array(
            [[_ragged(c, s) * SCALE[c] + OFFSET[c] for s in range(RAGGED_SAMPLES)] for c in range(N_CHANNELS)]
        )
        assert window.data.shape == (N_CHANNELS, RAGGED_SAMPLES)
        assert (window.data == expected).all()
        # The two seams by name: the first sample of the short shard and the last of the
        # short chunk.
        assert window.data[0, 500] == _ragged(0, 500) * SCALE[0] + OFFSET[0]
        assert window.data[2, 699] == _ragged(2, 699) * SCALE[2] + OFFSET[2]

    def test_a_store_shorter_than_its_index_says_is_refused(self, served, local_index) -> None:
        """read_window clamps to the array's real length, so a store reconverted shorter
        than the index read earlier comes back as a smaller window and no error."""
        group = local_index.store(RAGGED_PATH).groups[0]
        longer = dataclasses.replace(group, n_samples=1000)
        stale = dataclasses.replace(local_index, stores=(Store(path=RAGGED_PATH, zarr="rec4.zarr", groups=(longer,)),))

        with pytest.raises(IndexError_, match=r"3 channels by 700 samples.*3 by 1000"):
            load_recording(DATASET, RAGGED_PATH, index=stale)

    def test_index_url_names_the_index_that_is_read(self, served) -> None:
        _, base_url, requested = served
        requested.clear()

        window = load_recording(DATASET, FIRST_PATH, index_url=f"{base_url}index.json")

        assert window.data.shape == (N_CHANNELS, N_SAMPLES)
        assert requested[0][0] == "/index.json"

    def test_a_running_event_loop_is_refused_clearly_and_leaves_no_coroutine_behind(self, local_index) -> None:
        async def inside() -> str:
            with pytest.raises(RuntimeError, match="await read_index and read_window") as caught:
                load_recording(DATASET, FIRST_PATH, index=local_index)
            return str(caught.value)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            asyncio.run(inside())
            gc.collect()

        assert [str(w.message) for w in caught if "never awaited" in str(w.message)] == []

    def test_the_default_cap_is_two_gibibytes(self) -> None:
        import inspect

        assert inspect.signature(load_recording).parameters["max_bytes"].default == 2 * 1024**3

    def test_nothing_is_written_to_disk(self, served, local_index, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        for name in ("TMPDIR", "TEMP", "TMP"):
            monkeypatch.setenv(name, str(tmp_path))

        load_recording(DATASET, FIRST_PATH, index=local_index)

        assert list(tmp_path.iterdir()) == []


# The smallest recording of a live public dataset with a verified Zarr copy.
LIVE_DATASET = "nm000132"


def _smallest_store(index):
    """The recording with the fewest channel-samples, so the download stays small."""
    return min(index.stores, key=lambda s: sum(g.n_channels * g.n_samples for g in s.groups))


@pytest.mark.network
class TestAgainstTheLiveArchive:
    def test_reads_the_whole_smallest_recording(self) -> None:
        pytest.importorskip("zarr", reason="needs the zarr extra")
        index = asyncio.run(read_index(LIVE_DATASET))
        store = _smallest_store(index)
        group = store.group()

        window = load_recording(LIVE_DATASET, store.path, index=index)

        assert window.data.shape == (group.n_channels, group.n_samples)
        assert window.rate == group.rate
        assert window.unit == "uV"
        assert window.labels is not None and len(window.labels) == group.n_channels
        # EEG in microvolts: a conversion left at the stored counts, or in volts, is
        # orders of magnitude outside this.
        assert 1.0 < float(window.data.std()) < 1e5

    def test_the_digest_is_the_live_indexs_own(self) -> None:
        digest = index_sha256(LIVE_DATASET)

        assert len(digest) == 64
        assert int(digest, 16) >= 0
        assert digest == index_sha256(LIVE_DATASET), "two reads of an unchanged index disagree"
