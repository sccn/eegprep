"""Shared by the transport and read-window tests: a host's own client, and a clean default."""

from __future__ import annotations

import asyncio
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


@pytest.fixture(autouse=True)
def _platform_default_afterwards() -> Iterator[None]:
    """Restore platform selection after every test, so a registered transport cannot
    leak into a later one that reads with no ``transport=`` argument."""
    yield
    set_default_transport(None)
