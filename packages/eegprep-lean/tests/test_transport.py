"""Tests for the transport, against a real HTTP server on loopback.

Nothing here is stubbed. ``UrllibTransport`` makes genuine requests over a socket to a
``http.server`` running in this process, which is what makes the error branches real
rather than asserted: a 404 is produced by a server deciding to send one, and a refused
connection by there being nothing to connect to.

``FetchTransport`` and ``PyfetchTransport`` are held to the same contract by the same
tests, parametrized over all three, through host clients that make real requests
(``conftest.HostFetch`` for ``FetchTransport``, ``conftest.FakePyfetch`` for
``PyfetchTransport``, installed at ``sys.modules["pyodide.http"]`` so
``PyfetchTransport.get`` runs its real code path off Pyodide). Only what is specific to a
given client has tests of its own.
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
    Transport,
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

        if self.path == "/moved":
            self.send_response(302)
            self.send_header("Location", "/data")
            self.send_header("Content-Length", "0")
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


@pytest.fixture(params=["urllib", "fetch", "pyfetch"])
def transport(request: pytest.FixtureRequest, host_fetch, pyodide_http) -> Transport:
    """Each transport that can run here, held to one contract."""
    if request.param == "fetch":
        return FetchTransport(host_fetch)
    if request.param == "pyfetch":
        return PyfetchTransport()
    return UrllibTransport(timeout_s=10)


def _get(transport: Transport, url: str, **kwargs):
    return asyncio.run(transport.get(url, **kwargs))


class TestRanges:
    def test_whole_body(self, base_url: str, transport: Transport) -> None:
        response = _get(transport, f"{base_url}/data")

        assert response.status == 200
        assert response.body == BODY

    def test_closed_range_returns_only_that_slice(self, base_url: str, transport: Transport) -> None:
        response = _get(transport, f"{base_url}/data", start=10, end=19)

        assert response.status == 206
        assert response.body == BODY[10:20]

    def test_suffix_range_returns_the_tail(self, base_url: str, transport: Transport) -> None:
        """The form Zarr's sharding codec reads its chunk index with."""
        response = _get(transport, f"{base_url}/data", end=16)

        assert response.status == 206
        assert response.body == BODY[-16:]

    def test_open_ended_range_returns_the_rest(self, base_url: str, transport: Transport) -> None:
        response = _get(transport, f"{base_url}/data", start=300)

        assert response.status == 206
        assert response.body == BODY[300:]


class TestFailures:
    def test_a_range_answered_in_full_is_refused(self, base_url: str, transport: Transport) -> None:
        """A 200 to a Range request is a success to HTTP and a failure here.

        Without this the caller gets the whole object and slices it by the offsets it
        asked for, reading the wrong bytes and getting plausible numbers out. That is why
        it raises rather than returning what arrived.
        """
        with pytest.raises(TransportError) as caught:
            _get(transport, f"{base_url}/ignores-range", start=10, end=19)

        assert caught.value.status == 200
        assert "answered in full" in str(caught.value)

    def test_an_unranged_200_is_not_refused(self, base_url: str, transport: Transport) -> None:
        """The guard must key on whether a range was asked for, not on the status alone."""
        assert _get(transport, f"{base_url}/ignores-range").status == 200

    def test_http_error_carries_its_status(self, base_url: str, transport: Transport) -> None:
        """For FetchTransport and PyfetchTransport this is the only place a 404 becomes
        the TransportError the store maps to absence, since the host's client returns a
        status, not an error."""
        with pytest.raises(TransportError) as caught:
            _get(transport, f"{base_url}/missing")

        assert caught.value.status == 404
        assert caught.value.url.endswith("/missing")
        assert "[404]" in str(caught.value)

    def test_connection_failure_has_no_status_and_still_names_the_url(self, transport: Transport) -> None:
        """The other TransportError branch: there was no response to take a status from."""
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            dead_port = probe.getsockname()[1]

        with pytest.raises(TransportError) as caught:
            _get(transport, f"http://127.0.0.1:{dead_port}/data")

        assert caught.value.status is None
        assert str(dead_port) in str(caught.value)


class TestFetchTransport:
    """What is specific to a client the host supplies."""

    def test_range_is_the_only_header_sent(self, base_url: str, host_fetch) -> None:
        """Anything else risks a CORS preflight in a browser or is refused by the host's
        client, and an unranged request sends nothing at all."""
        transport = FetchTransport(host_fetch)

        _get(transport, f"{base_url}/data", start=10, end=19)
        _get(transport, f"{base_url}/data")

        assert [headers for _, headers in host_fetch.seen] == [{"Range": "bytes=10-19"}, {}]

    def test_a_redirect_handed_back_is_refused(self, base_url: str, host_fetch) -> None:
        """A client that does not follow redirects returns the 3xx itself. Read as data,
        it would give zarr an empty document with no error anywhere."""
        with pytest.raises(TransportError) as caught:
            _get(FetchTransport(host_fetch), f"{base_url}/moved")

        assert caught.value.status == 302

    def test_an_opaque_response_is_refused(self) -> None:
        """Status 0 is what a browser reports for an opaque response, which carries no
        readable body. A host's client at the network boundary, answering that way."""

        async def opaque(url: str, *, headers: dict[str, str]) -> tuple[int, bytes]:
            return 0, b""

        with pytest.raises(TransportError) as caught:
            _get(FetchTransport(opaque), "http://example.invalid/data")

        assert caught.value.status == 0

    def test_a_bytes_like_body_arrives_as_bytes(self) -> None:
        """A client over a JavaScript buffer may hand back a bytearray or a memoryview."""

        async def buffer(url: str, *, headers: dict[str, str]) -> tuple[int, memoryview]:
            return 200, memoryview(bytearray(b"payload"))

        response = _get(FetchTransport(buffer), "http://example.invalid/data")

        assert type(response.body) is bytes
        assert response.body == b"payload"


class TestPyfetchTransport:
    """What is specific to the browser transport: what ``pyodide.http.pyfetch`` is asked
    to send. This is the regression check for #419: a script-set ``User-Agent`` here
    turned every read into a CORS-preflighted request in Safari and Firefox, which
    ``zarr.nemar.org`` refuses."""

    def test_only_range_is_sent_and_never_user_agent(self, base_url: str, pyodide_http) -> None:
        transport = PyfetchTransport()

        _get(transport, f"{base_url}/data", start=10, end=19)
        _get(transport, f"{base_url}/data", end=16)
        _get(transport, f"{base_url}/data")

        assert [headers for _, headers in pyodide_http.seen] == [
            {"Range": "bytes=10-19"},
            {"Range": "bytes=-16"},
            {},
        ]


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

    def test_a_host_transport_wins_in_pyodide(self, monkeypatch: pytest.MonkeyPatch, host_fetch) -> None:
        """The case it exists for: a sandbox without ``pyodide.http`` registers its own
        client, and platform selection must not hand back the transport that imports it."""
        monkeypatch.setattr(sys, "platform", "emscripten")
        registered = FetchTransport(host_fetch)

        set_default_transport(registered)

        assert default_transport() is registered

    def test_a_host_transport_wins_natively_too(self, host_fetch) -> None:
        registered = FetchTransport(host_fetch)

        set_default_transport(registered)

        assert default_transport() is registered

    def test_none_restores_platform_selection(self, host_fetch) -> None:
        set_default_transport(FetchTransport(host_fetch))

        set_default_transport(None)

        assert isinstance(default_transport(), UrllibTransport)
