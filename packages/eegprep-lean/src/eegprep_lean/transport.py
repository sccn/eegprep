"""HTTP transport for the browser and for everywhere else.

Three implementations of one small interface. The browser and a workstation disagree
about how a request is made and there is no portable answer, and a sandbox may disagree
with both.

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

#: Sent on every request from :class:`UrllibTransport`. The NEMAR hosts have been
#: observed refusing the default ``Python-urllib/x.y`` agent, and an unnamed client is not
#: something an archive should have to guess about anyway. That is true only of Chrome,
#: which drops a script-set ``User-Agent`` and sends its own instead; Safari and Firefox
#: send the script-set value, which is why the browser transports never set this header at
#: all (see :class:`FetchTransport`).
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
    when no response arrived at all, and a request the host refuses must raise too,
    never answer with a status of its own: the store reads 403, 404 and 416 as a key
    that does not exist, so a refusal dressed as a 403 would pass for absence.
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


class FetchTransport:
    """Transport over a client the host supplies, for a runtime that owns the network.

    Sends ``Range`` and nothing else. In a browser another header risks a CORS preflight
    or is dropped, and ``User-Agent`` is both, depending on the browser; the host's
    client is also free to refuse headers it does not recognize. ``Range`` itself is
    safelisted only as ``bytes=N-M``, so the suffix form the shard index is read with
    still costs a preflight, which the NEMAR hosts answer. Deadlines are the host's too,
    since the client it supplies is the thing that can enforce one.

    Nothing redirects here. A host client may follow a redirect, raise on one, or hand
    back the ``3xx`` itself; the last must not be read as data, so only a ``2xx`` is.
    """

    def __init__(self, fetch: Fetch) -> None:
        self.fetch = fetch

    async def get(self, url: str, *, start: int | None = None, end: int | None = None) -> Response:
        range_value = _range_header(start, end)
        headers = {"Range": range_value} if range_value else {}
        try:
            result = await self.fetch(url, headers=headers)
        except Exception as err:
            raise TransportError(f"request failed: {err}", url=url) from err
        # Outside the try: a client that breaks its own contract fails as itself, not
        # as a request that failed.
        status, body = result
        if not 200 <= status < 300:
            raise TransportError("request failed", url=url, status=status)
        # A host client may hand back a bytes-like buffer rather than bytes; Response
        # promises bytes.
        return _check_ranged(Response(status=status, body=bytes(body)), url=url, ranged=range_value is not None)


async def _pyfetch(url: str, *, headers: dict[str, str]) -> tuple[int, bytes]:
    """Adapts ``pyodide.http.pyfetch`` to the :class:`Fetch` shape :class:`FetchTransport` calls.

    Imported lazily so this module stays importable off Pyodide; the tests install a
    stand-in module at ``sys.modules["pyodide.http"]`` to exercise this off Pyodide too.
    """
    from pyodide.http import pyfetch  # ty: ignore[unresolved-import]

    response = await pyfetch(url, headers=headers)
    return response.status, await response.bytes()


class PyfetchTransport:
    """Browser transport: :class:`FetchTransport` over the host's own ``fetch``.

    ``pyodide.http.pyfetch`` is Pyodide's binding to the browser's ``fetch``. Routing it
    through :class:`FetchTransport` gives the browser transport one implementation of the
    header rule instead of two: ``Range`` only, and nothing at all for an unranged read. A
    script-set ``User-Agent`` here used to turn every read into a CORS-preflighted request
    in Safari and Firefox, which ``zarr.nemar.org`` refuses; Chrome only worked because it
    silently drops a script-set ``User-Agent`` and sends its own instead.
    """

    def __init__(self, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.timeout_s = timeout_s
        self._transport = FetchTransport(_pyfetch)

    async def get(self, url: str, *, start: int | None = None, end: int | None = None) -> Response:
        return await self._transport.get(url, start=start, end=end)


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
