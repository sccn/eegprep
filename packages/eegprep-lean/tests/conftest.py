"""Shared by the transport and read-window tests: a host's own client, and a clean default."""

from __future__ import annotations

import asyncio
import sys
import threading
import types
import urllib.error
import urllib.request
from collections.abc import Iterator

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


def _forget(future: asyncio.Future) -> None:
    if not future.cancelled():
        future.exception()


def _deliver(
    loop: asyncio.AbstractEventLoop,
    future: asyncio.Future,
    result: object,
    error: BaseException | None,
) -> None:
    def apply() -> None:
        if future.done():
            return
        if error is not None:
            future.set_exception(error)
            return
        future.set_result(result)

    try:
        loop.call_soon_threadsafe(apply)
    except RuntimeError:
        return


async def _await_in_thread(fn, signal: _AbortSignal | None):
    """Run ``fn`` on a daemon thread, and stop waiting if ``signal`` aborts.

    The thread is a daemon so a stall that outlives the test cannot hold
    ``asyncio.run`` open. ``urlopen`` still follows redirects and returns HTTP
    errors as responses.
    """
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    future.add_done_callback(_forget)

    def work() -> None:
        try:
            result = fn()
        except Exception as err:
            _deliver(loop, future, None, err)
        else:
            _deliver(loop, future, result, None)

    threading.Thread(target=work, daemon=True).start()
    abort_task = asyncio.create_task(signal.wait()) if signal is not None else None
    try:
        if abort_task is None:
            return await future
        done, _pending = await asyncio.wait({future, abort_task}, return_when=asyncio.FIRST_COMPLETED)
        if future in done and (abort_task not in done or future.exception() is None):
            return future.result()
        raise OSError("The operation was aborted.")
    finally:
        if abort_task is not None:
            abort_task.cancel()


def _open(url: str, headers: dict[str, str]):
    request = urllib.request.Request(url, headers=headers)
    try:
        return urllib.request.urlopen(request, timeout=10)
    except urllib.error.HTTPError as err:
        return err


class _FakePyodideResponse:
    """The slice of ``pyodide.http.FetchResponse`` that :class:`~eegprep_lean.PyfetchTransport` reads.

    ``urlopen`` has already returned the headers. ``bytes`` reads the body, and an
    abort signal ends that wait instead of letting a stalled body finish.
    """

    def __init__(self, raw, signal: _AbortSignal | None) -> None:
        self.status = raw.status
        self._raw = raw
        self._signal = signal

    async def bytes(self) -> bytes:
        try:
            return await _await_in_thread(self._read, self._signal)
        finally:
            if self._signal is not None:
                self._signal.cancel_timer()

    def _read(self) -> bytes:
        try:
            return self._raw.read()
        finally:
            self._raw.close()


class FakePyfetch:
    """Stands in for ``pyodide.http.pyfetch``: a real ``urllib`` request, wrapped in the
    shape Pyodide's binding returns, so :class:`~eegprep_lean.PyfetchTransport` can be
    exercised off Pyodide, against the same loopback server the other transports use.

    Follows redirects and never raises for an HTTP error status, which is what a
    browser's ``fetch()`` does; raises only when no response arrives at all. Headers
    come back before the body, so a stall in either place is visible. An abort
    ``signal`` ends the wait; the blocking read is a daemon thread, so the test does
    not sit out the stall. Records each request's headers, so a test can assert what
    was actually sent.
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
            raw = await _await_in_thread(lambda: _open(url, headers), signal)
        except BaseException:
            if signal is not None:
                signal.cancel_timer()
            raise
        return _FakePyodideResponse(raw, signal)


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
