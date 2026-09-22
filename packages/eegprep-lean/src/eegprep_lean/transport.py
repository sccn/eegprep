"""HTTP transport for the browser and for everywhere else.

Several implementations of one small interface, because the browser and a workstation
disagree about how a request is made and there is no portable answer.

In Pyodide there is no socket layer and no thread to run one on, so a request has to go
through the host's own ``fetch``, reached as ``pyodide.http.pyfetch``. Off Pyodide there
is no ``pyfetch``, so the request goes through ``urllib`` on a worker thread, which keeps
the caller's ``await`` honest without pulling in an async HTTP stack. Adding one would
cost more download than the reader itself, which is the thing this package exists to
avoid.

A host that owns the interpreter can also own the network. A sandboxed browser runtime
may remove ``pyodide.http`` and offer its own client instead, one that enforces what the
code may reach. :class:`FetchTransport` reads through such a client, and
:func:`set_default_transport` is how the host makes it the default, so reader code written
without a ``transport=`` argument still works there.

Only ``GET`` is needed, and only ever with a byte range. Nothing here writes.
"""

from __future__ import annotations

import asyncio
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

#: Sent on every request. The NEMAR hosts have been observed refusing the default
#: ``Python-urllib/x.y`` agent, and an unnamed client is not something an archive should
#: have to guess about anyway. Browsers set their own and ignore this.
USER_AGENT = "eegprep-lean"

#: Fetching a chunk that is not cached at the edge crosses to S3, so this is generous.
DEFAULT_TIMEOUT_S = 60.0


class TransportError(RuntimeError):
    """A request did not produce bytes.

    Carries the status when there was one. A range read that comes back ``200`` rather
    than ``206`` is a failure here even though it is a success to HTTP: it means the
    server ignored the range and is about to hand over an entire shard, which for this
    data is megabytes nobody asked for.
    """

    def __init__(self, message: str, *, url: str, status: int | None = None) -> None:
        super().__init__(f"{message} ({url})" if status is None else f"{message} [{status}] ({url})")
        self.url = url
        self.status = status


@dataclass(frozen=True)
class Response:
    """What a read returns. ``status`` is kept so a caller can tell 200 from 206."""

    status: int
    body: bytes


class Transport(Protocol):
    """Fetch bytes, optionally a byte range, and never block the event loop."""

    async def get(self, url: str, *, start: int | None = None, end: int | None = None) -> Response: ...


class Fetch(Protocol):
    """A host's HTTP client, as :class:`FetchTransport` calls it.

    One ``GET``, returning the status and the body whatever the status was. It raises
    only when no response arrived at all.
    """

    async def __call__(self, url: str, *, headers: dict[str, str]) -> tuple[int, bytes]: ...


def _range_header(start: int | None, end: int | None) -> str | None:
    """Build a Range header value, including the suffix form the sharding codec needs.

    ``start=None, end=n`` is ``bytes=-n``, the last n bytes. Zarr's sharding codec reads
    its chunk index that way, so a transport that cannot express it can read metadata but
    never data.
    """
    if start is None:
        # Suffix form, or no range at all.
        return None if end is None else f"bytes=-{int(end)}"
    if end is None:
        return f"bytes={int(start)}-"
    return f"bytes={int(start)}-{int(end)}"


def _check_ranged(response: Response, *, url: str, ranged: bool) -> Response:
    """Refuse a ranged request that was answered in full.

    A ``200`` to a ``Range`` request is a success to HTTP and a failure here: the server
    ignored the range and is sending the whole object. For a shard that is megabytes
    nobody asked for, and worse, a caller that slices the result by the offsets it
    requested reads the wrong bytes and gets plausible numbers out. Fail instead.
    """
    if ranged and response.status != 206:
        raise TransportError("range request was answered in full", url=url, status=response.status)
    return response


class UrllibTransport:
    """Native transport: ``urllib`` on a worker thread.

    ``asyncio.to_thread`` is what keeps this usable from async code without an async HTTP
    dependency. It is also exactly what cannot work in Pyodide, whose main thread cannot
    start one, which is why the browser gets its own implementation rather than a flag.
    """

    def __init__(self, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.timeout_s = timeout_s

    async def get(self, url: str, *, start: int | None = None, end: int | None = None) -> Response:
        return await asyncio.to_thread(self._get_blocking, url, start, end)

    def _get_blocking(self, url: str, start: int | None, end: int | None) -> Response:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        range_value = _range_header(start, end)
        if range_value:
            request.add_header("Range", range_value)
        try:
            # urllib follows the 302 to the bucket on its own, which is what a
            # non-browser client is expected to do: zarr.nemar.org is the contract and
            # where the bytes physically come from is its business, not ours.
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                result = Response(status=response.status, body=response.read())
        except urllib.error.HTTPError as err:
            raise TransportError("request failed", url=url, status=err.code) from err
        except urllib.error.URLError as err:
            raise TransportError(f"request failed: {err.reason}", url=url) from err
        return _check_ranged(result, url=url, ranged=range_value is not None)


class PyfetchTransport:
    """Browser transport: the host's own ``fetch``, via ``pyodide.http.pyfetch``.

    Imported lazily so this module stays importable off Pyodide, where the tests run.
    """

    def __init__(self, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.timeout_s = timeout_s

    async def get(self, url: str, *, start: int | None = None, end: int | None = None) -> Response:
        from pyodide.http import pyfetch  # ty: ignore[unresolved-import]

        headers = {"User-Agent": USER_AGENT}
        range_value = _range_header(start, end)
        if range_value:
            headers["Range"] = range_value
        try:
            response = await pyfetch(url, headers=headers)
        except Exception as err:  # pragma: no cover - browser only
            raise TransportError(f"request failed: {err}", url=url) from err
        if response.status >= 400:  # pragma: no cover - browser only
            raise TransportError("request failed", url=url, status=response.status)
        return _check_ranged(
            Response(status=response.status, body=await response.bytes()),
            url=url,
            ranged=range_value is not None,
        )


class FetchTransport:
    """Transport over a client the host supplies, for a runtime that owns the network.

    Sends ``Range`` and nothing else. In a browser any other header either triggers a
    CORS preflight or is dropped, and ``User-Agent`` is both, depending on the browser;
    the host's client is also free to refuse headers it does not recognize. Deadlines
    are the host's too, since the client it supplies is the thing that can enforce one.
    """

    def __init__(self, fetch: Fetch) -> None:
        self.fetch = fetch

    async def get(self, url: str, *, start: int | None = None, end: int | None = None) -> Response:
        range_value = _range_header(start, end)
        headers = {"Range": range_value} if range_value else {}
        try:
            status, body = await self.fetch(url, headers=headers)
        except Exception as err:
            raise TransportError(f"request failed: {err}", url=url) from err
        if status >= 400:
            raise TransportError("request failed", url=url, status=status)
        return _check_ranged(Response(status=status, body=bytes(body)), url=url, ranged=range_value is not None)


#: Set by :func:`set_default_transport`, and consulted before the platform is.
_host_default: Transport | None = None


def set_default_transport(transport: Transport | None) -> None:
    """Choose what :func:`default_transport` returns, for a host that owns the network.

    Meant to be called once, by the runtime that owns the interpreter, before any reader
    code runs: a sandbox that removes ``pyodide.http`` registers a transport over its own
    client here, so ``await open_array(url)`` works unchanged. An explicit ``transport=``
    argument still wins everywhere, and ``None`` restores platform selection.
    """
    global _host_default
    _host_default = transport


def running_in_pyodide() -> bool:
    """True when this interpreter is Pyodide's.

    Pyodide reports ``sys.platform == "emscripten"``. That is the same signal the
    dependency floors are evaluated against, so the two cannot drift apart.
    """
    return sys.platform == "emscripten"


def default_transport(*, timeout_s: float = DEFAULT_TIMEOUT_S) -> Transport:
    """The transport that can actually work here.

    The host's, when one was set with :func:`set_default_transport`, returned as it is:
    ``timeout_s`` applies only to the platform transports.
    """
    if _host_default is not None:
        return _host_default
    if running_in_pyodide():
        return PyfetchTransport(timeout_s=timeout_s)
    return UrllibTransport(timeout_s=timeout_s)
