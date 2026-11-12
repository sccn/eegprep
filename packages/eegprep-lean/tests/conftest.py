"""Shared by the transport and read-window tests: a host's own client, and a clean default."""

from __future__ import annotations

import asyncio
import contextlib
import sys
import types
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Awaitable, Callable, Iterator
from typing import TypeVar

import pytest

from eegprep_lean.transport import set_default_transport


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Hand a redirect back as the response, as a sandbox's client may."""

    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _urllib_get(url: str, headers: dict[str, str]) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers=headers)
    try:
        with _OPENER.open(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


class HostFetch:
    """A client of the shape :class:`~eegprep_lean.FetchTransport` takes, as a host supplies one.

    Real HTTP over ``urllib``, returning the status rather than raising for an HTTP error,
    which is the contract, and not following redirects. Records each request's headers,
    so a test can assert what was sent rather than only what came back.
    """

    def __init__(self) -> None:
        self.seen: list[tuple[str, dict[str, str]]] = []

    async def __call__(self, url: str, *, headers: dict[str, str]) -> tuple[int, bytes]:
        self.seen.append((url, dict(headers)))
        return await asyncio.to_thread(_urllib_get, url, headers)


@pytest.fixture
def host_fetch() -> HostFetch:
    return HostFetch()


class _AbortSignal:
    """The slice of ``AbortSignal`` :class:`~eegprep_lean.PyfetchTransport` drives.

    ``timeout`` starts when it is called, in milliseconds, matching ``AbortSignal.timeout``.
    ``aborted`` is set before waiters resume, so a caller can tell this deadline from any
    other failure. ``cancel_timer`` stops the deadline once the body has been read.
    """

    def __init__(self) -> None:
        self.aborted = False
        self._event = asyncio.Event()
        self._handle: asyncio.TimerHandle | None = None

    def abort(self) -> None:
        self.aborted = True
        self._event.set()

    def cancel_timer(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None

    async def wait(self) -> None:
        await self._event.wait()

    @staticmethod
    def timeout(milliseconds: float) -> _AbortSignal:
        signal = _AbortSignal()
        signal._handle = asyncio.get_running_loop().call_later(milliseconds / 1000, signal.abort)
        return signal


class _FakePyodideResponse:
    """The slice of ``pyodide.http.FetchResponse`` that :class:`~eegprep_lean.PyfetchTransport` reads.

    Headers are already in hand. ``bytes`` reads the body, and an abort signal that fires
    during that read closes the socket instead of returning a short body.
    """

    def __init__(
        self,
        status: int,
        headers: dict[str, str],
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        leftover: bytes,
        signal: _AbortSignal | None,
    ) -> None:
        self.status = status
        self.headers = headers
        self._reader = reader
        self._writer = writer
        self._leftover = leftover
        self._signal = signal

    async def bytes(self) -> bytes:
        async def read_body() -> bytes:
            chunks = [self._leftover] if self._leftover else []
            try:
                while True:
                    block = await self._reader.read(65536)
                    if not block:
                        break
                    chunks.append(block)
            finally:
                self._writer.close()
            return b"".join(chunks)

        try:
            return await _await_fetch(read_body(), self._signal, self._writer.close)
        finally:
            if self._signal is not None:
                self._signal.cancel_timer()

    def close(self) -> None:
        self._writer.close()


def _request_target(url: str) -> tuple[str, int, str]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname is None or parsed.port is None:
        raise OSError(f"unsupported url: {url}")
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return parsed.hostname, parsed.port, path


def _encode_request(path: str, host: str, headers: dict[str, str]) -> bytes:
    lines = [f"GET {path} HTTP/1.0", f"Host: {host}", "Connection: close"]
    lines.extend(f"{name}: {value}" for name, value in headers.items())
    return ("\r\n".join(lines) + "\r\n\r\n").encode("ascii")


async def _read_response_head(reader: asyncio.StreamReader) -> tuple[int, dict[str, str], bytes]:
    buffer = bytearray()
    while b"\r\n\r\n" not in buffer:
        chunk = await reader.read(4096)
        if not chunk:
            raise ConnectionError("connection closed before headers")
        buffer.extend(chunk)
        if len(buffer) > 65536:
            raise ConnectionError("response headers are too large")
    head, _, leftover = bytes(buffer).partition(b"\r\n\r\n")
    lines = head.decode("iso-8859-1").split("\r\n")
    status = int(lines[0].split()[1])
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        name, _, value = line.partition(":")
        headers[name.lower()] = value.strip()
    return status, headers, leftover


_T = TypeVar("_T")


async def _await_fetch(awaitable: Awaitable[_T], signal: _AbortSignal | None, close: Callable[[], None]) -> _T:
    """Await ``awaitable`` until it finishes or ``signal`` aborts the socket."""
    task = asyncio.ensure_future(awaitable)
    abort_task = asyncio.create_task(signal.wait()) if signal is not None else None
    try:
        if abort_task is None:
            return await task
        done, _pending = await asyncio.wait({task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
        if task in done and (abort_task not in done or task.exception() is None):
            return task.result()
        raise OSError("The operation was aborted.")
    except asyncio.CancelledError:
        task.cancel()
        close()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await task
        raise
    finally:
        if abort_task is not None:
            abort_task.cancel()
        if signal is not None and signal.aborted and not task.done():
            task.cancel()
            close()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await task


_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_MAX_REDIRECTS = 10


class FakePyfetch:
    """Stands in for ``pyodide.http.pyfetch``: a real request to the loopback server, wrapped
    in the shape Pyodide's binding returns, so :class:`~eegprep_lean.PyfetchTransport` can be
    exercised off Pyodide.

    Follows redirects and never raises for an HTTP error status, which is what a browser's
    ``fetch()`` does; raises only when no response arrives at all. Headers come back before
    the body, so a stall in either place is visible. An abort ``signal`` closes the socket
    instead of letting that stall run out. Records each request's headers, so a test can
    assert what was actually sent.
    """

    def __init__(self) -> None:
        self.seen: list[tuple[str, dict[str, str]]] = []

    async def __call__(
        self,
        url: str,
        *,
        headers: dict[str, str],
        signal: _AbortSignal | None = None,
    ) -> _FakePyodideResponse:
        self.seen.append((url, dict(headers)))
        try:
            current = url
            for _redirect in range(_MAX_REDIRECTS):
                response = await self._once(current, headers, signal)
                location = response.headers.get("location")
                if response.status not in _REDIRECT_STATUSES or location is None:
                    return response
                response.close()
                current = urllib.parse.urljoin(current, location)
            raise OSError("too many redirects")
        except BaseException:
            if signal is not None:
                signal.cancel_timer()
            raise

    async def _once(self, url: str, headers: dict[str, str], signal: _AbortSignal | None) -> _FakePyodideResponse:
        host, port, path = _request_target(url)
        reader, writer = await _await_fetch(asyncio.open_connection(host, port), signal, lambda: None)
        writer.write(_encode_request(path, f"{host}:{port}", headers))
        try:
            await _await_fetch(writer.drain(), signal, writer.close)
            status, response_headers, leftover = await _await_fetch(_read_response_head(reader), signal, writer.close)
        except BaseException:
            writer.close()
            raise
        return _FakePyodideResponse(status, response_headers, reader, writer, leftover, signal)


@pytest.fixture
def pyodide_http(monkeypatch: pytest.MonkeyPatch) -> FakePyfetch:
    """Installs a fake ``pyodide.http`` module at ``sys.modules``, so
    ``PyfetchTransport.get`` runs its real code path off Pyodide instead of skipping.

    ``js.AbortSignal`` is the same stand-in the transport's deadline calls: off Pyodide
    there is no browser global, and the signal has to actually fire for a stalled read
    to stop.
    """
    fake = FakePyfetch()
    pyodide_pkg = types.ModuleType("pyodide")
    http_module = types.ModuleType("pyodide.http")
    http_module.pyfetch = fake
    pyodide_pkg.http = http_module
    js_module = types.ModuleType("js")
    js_module.AbortSignal = _AbortSignal
    monkeypatch.setitem(sys.modules, "pyodide", pyodide_pkg)
    monkeypatch.setitem(sys.modules, "pyodide.http", http_module)
    monkeypatch.setitem(sys.modules, "js", js_module)
    return fake


@pytest.fixture(autouse=True)
def _platform_default_afterwards() -> Iterator[None]:
    """Restore platform selection after every test, so a registered transport cannot
    leak into a later one that reads with no ``transport=`` argument."""
    yield
    set_default_transport(None)
