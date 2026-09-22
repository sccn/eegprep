"""Shared by the transport and read-window tests: a host's own client, and a clean default."""

from __future__ import annotations

import asyncio
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from eegprep_lean.transport import set_default_transport


class HostFetch:
    """A client of the shape :class:`~eegprep_lean.FetchTransport` takes, as a host supplies one.

    Real HTTP over ``urllib``, returning the status rather than raising for an HTTP error,
    which is the contract. Records each request's headers, so a test can assert what was
    sent rather than only what came back.
    """

    def __init__(self) -> None:
        self.seen: list[tuple[str, dict[str, str]]] = []

    async def __call__(self, url: str, *, headers: dict[str, str]) -> tuple[int, bytes]:
        self.seen.append((url, dict(headers)))
        return await asyncio.to_thread(self._get, url, headers)

    @staticmethod
    def _get(url: str, headers: dict[str, str]) -> tuple[int, bytes]:
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as err:
            return err.code, err.read()


@pytest.fixture
def host_fetch() -> HostFetch:
    return HostFetch()


@pytest.fixture
def host_default() -> Iterator[None]:
    """Restore platform selection afterwards, so a registered transport cannot leak."""
    yield
    set_default_transport(None)
