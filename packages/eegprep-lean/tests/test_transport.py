"""Tests for the transport, against a real HTTP server on loopback.

Nothing here is stubbed. ``UrllibTransport`` makes genuine requests over a socket to a
``http.server`` running in this process, which is what makes the error branches real
rather than asserted: a 404 is produced by a server deciding to send one, and a refused
connection by there being nothing to connect to.

``FetchTransport`` is exercised against the same server, through a host client that
makes real requests (``conftest.HostFetch``), so its range handling and its refusals are
checked against real responses too.

``PyfetchTransport`` cannot be exercised here. It needs ``pyodide.http``, which exists
only in a browser, so it ships covered by the Pyodide harness rather than by this file.
Saying so plainly is better than three ``no cover`` pragmas implying it was considered.
"""

from __future__ import annotations

import asyncio
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from eegprep_lean.transport import (
    FetchTransport,
    PyfetchTransport,
    TransportError,
    UrllibTransport,
    default_transport,
    running_in_pyodide,
    set_default_transport,
)

BODY = b"0123456789" * 32


class _Handler(BaseHTTPRequestHandler):
    """Serves BODY, honoring Range unless the path asks it not to."""

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        if self.path == "/missing":
            self.send_response(404)
            self.end_headers()
            return

        range_header = self.headers.get("Range")
        # The path a client cannot detect without checking the status: the server was
        # asked for a range and answered with everything.
        if range_header and self.path == "/ignores-range":
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY)))
            self.end_headers()
            self.wfile.write(BODY)
            return

        if range_header and range_header.startswith("bytes="):
            spec = range_header.removeprefix("bytes=")
            if spec.startswith("-"):
                chunk = BODY[-int(spec[1:]) :]
                start = len(BODY) - len(chunk)
            else:
                first, _, last = spec.partition("-")
                start = int(first)
                chunk = BODY[start : int(last) + 1] if last else BODY[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{start + len(chunk) - 1}/{len(BODY)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return

        self.send_response(200)
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)

    def log_message(self, *args: object) -> None:
        """Quiet. The default writes every request to stderr."""


@pytest.fixture(scope="module")
def base_url():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def _get(url: str, **kwargs):
    return asyncio.run(UrllibTransport(timeout_s=10).get(url, **kwargs))


class TestRanges:
    def test_whole_body(self, base_url: str) -> None:
        response = _get(f"{base_url}/data")

        assert response.status == 200
        assert response.body == BODY

    def test_closed_range_returns_only_that_slice(self, base_url: str) -> None:
        response = _get(f"{base_url}/data", start=10, end=19)

        assert response.status == 206
        assert response.body == BODY[10:20]

    def test_suffix_range_returns_the_tail(self, base_url: str) -> None:
        """The form Zarr's sharding codec reads its chunk index with."""
        response = _get(f"{base_url}/data", end=16)

        assert response.status == 206
        assert response.body == BODY[-16:]

    def test_open_ended_range_returns_the_rest(self, base_url: str) -> None:
        response = _get(f"{base_url}/data", start=300)

        assert response.status == 206
        assert response.body == BODY[300:]


class TestFailures:
    def test_a_range_answered_in_full_is_refused(self, base_url: str) -> None:
        """A 200 to a Range request is a success to HTTP and a failure here.

        Without this the caller gets the whole object and slices it by the offsets it
        asked for, reading the wrong bytes and getting plausible numbers out. That is why
        it raises rather than returning what arrived.
        """
        with pytest.raises(TransportError) as caught:
            _get(f"{base_url}/ignores-range", start=10, end=19)

        assert caught.value.status == 200
        assert "answered in full" in str(caught.value)

    def test_an_unranged_200_is_not_refused(self, base_url: str) -> None:
        """The guard must key on whether a range was asked for, not on the status alone."""
        assert _get(f"{base_url}/ignores-range").status == 200

    def test_http_error_carries_its_status(self, base_url: str) -> None:
        with pytest.raises(TransportError) as caught:
            _get(f"{base_url}/missing")

        assert caught.value.status == 404
        assert caught.value.url.endswith("/missing")
        assert "[404]" in str(caught.value)

    def test_connection_failure_has_no_status_and_still_names_the_url(self) -> None:
        """The other TransportError branch: there was no response to take a status from."""
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            dead_port = probe.getsockname()[1]

        with pytest.raises(TransportError) as caught:
            _get(f"http://127.0.0.1:{dead_port}/data")

        assert caught.value.status is None
        assert str(dead_port) in str(caught.value)


class TestFetchTransport:
    """The host's client in place of the platform's, held to the same contract."""

    def test_whole_body(self, base_url: str, host_fetch) -> None:
        response = asyncio.run(FetchTransport(host_fetch).get(f"{base_url}/data"))

        assert response.status == 200
        assert response.body == BODY

    def test_closed_range_returns_only_that_slice(self, base_url: str, host_fetch) -> None:
        response = asyncio.run(FetchTransport(host_fetch).get(f"{base_url}/data", start=10, end=19))

        assert response.status == 206
        assert response.body == BODY[10:20]

    def test_suffix_range_returns_the_tail(self, base_url: str, host_fetch) -> None:
        """The form the sharding codec reads its chunk index with."""
        response = asyncio.run(FetchTransport(host_fetch).get(f"{base_url}/data", end=16))

        assert response.status == 206
        assert response.body == BODY[-16:]

    def test_open_ended_range_returns_the_rest(self, base_url: str, host_fetch) -> None:
        response = asyncio.run(FetchTransport(host_fetch).get(f"{base_url}/data", start=300))

        assert response.status == 206
        assert response.body == BODY[300:]

    def test_range_is_the_only_header_sent(self, base_url: str, host_fetch) -> None:
        """Anything else costs a CORS preflight in a browser or is refused by the host's
        client, and an unranged request sends nothing at all."""
        transport = FetchTransport(host_fetch)

        asyncio.run(transport.get(f"{base_url}/data", start=10, end=19))
        asyncio.run(transport.get(f"{base_url}/data"))

        assert [headers for _, headers in host_fetch.seen] == [{"Range": "bytes=10-19"}, {}]

    def test_a_range_answered_in_full_is_refused(self, base_url: str, host_fetch) -> None:
        with pytest.raises(TransportError) as caught:
            asyncio.run(FetchTransport(host_fetch).get(f"{base_url}/ignores-range", start=10, end=19))

        assert caught.value.status == 200
        assert "answered in full" in str(caught.value)

    def test_an_unranged_200_is_not_refused(self, base_url: str, host_fetch) -> None:
        assert asyncio.run(FetchTransport(host_fetch).get(f"{base_url}/ignores-range")).status == 200

    def test_http_error_carries_its_status(self, base_url: str, host_fetch) -> None:
        """The host's client returns a status rather than raising, so this is the only
        place a 404 becomes the TransportError the store maps to absence."""
        with pytest.raises(TransportError) as caught:
            asyncio.run(FetchTransport(host_fetch).get(f"{base_url}/missing"))

        assert caught.value.status == 404
        assert caught.value.url.endswith("/missing")

    def test_a_client_that_raises_has_no_status_and_still_names_the_url(self, host_fetch) -> None:
        """No response arrived, which is what the host's client raising means."""
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            dead_port = probe.getsockname()[1]

        with pytest.raises(TransportError) as caught:
            asyncio.run(FetchTransport(host_fetch).get(f"http://127.0.0.1:{dead_port}/data"))

        assert caught.value.status is None
        assert str(dead_port) in str(caught.value)


class TestPlatformSelection:
    def test_native_gets_the_urllib_transport(self) -> None:
        assert isinstance(default_transport(), UrllibTransport)
        assert running_in_pyodide() is False

    def test_emscripten_gets_the_pyfetch_transport(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pyodide reports sys.platform == "emscripten". Selecting on anything else, or
        getting the comparison backwards, means the browser silently takes the path whose
        first act is to start a thread it cannot start."""
        monkeypatch.setattr(sys, "platform", "emscripten")

        assert running_in_pyodide() is True
        assert isinstance(default_transport(), PyfetchTransport)

    def test_a_host_transport_wins_in_pyodide(self, monkeypatch: pytest.MonkeyPatch, host_fetch, host_default) -> None:
        """The case it exists for: a sandbox without ``pyodide.http`` registers its own
        client, and platform selection must not hand back the transport that imports it."""
        monkeypatch.setattr(sys, "platform", "emscripten")
        registered = FetchTransport(host_fetch)

        set_default_transport(registered)

        assert default_transport() is registered

    def test_a_host_transport_wins_natively_too(self, host_fetch, host_default) -> None:
        registered = FetchTransport(host_fetch)

        set_default_transport(registered)

        assert default_transport() is registered

    def test_none_restores_platform_selection(self, host_fetch, host_default) -> None:
        set_default_transport(FetchTransport(host_fetch))

        set_default_transport(None)

        assert isinstance(default_transport(), UrllibTransport)
