"""A read-only Zarr store over HTTP, for NEMAR's served copies.

Needs the ``zarr`` extra. It is not in the base install because the base exists to be
small, and a session that only wants to know what a dataset contains never pays for it.

Two things here are easy to get wrong and fail silently.

**Zarr's byte ranges are half-open; HTTP's are closed.** ``RangeByteRequest(start, end)``
excludes ``end``, while ``Range: bytes=a-b`` includes ``b``. Mapping one to the other
without the adjustment drops or duplicates a byte at every chunk boundary, which decodes
without error and returns wrong numbers.

**A missing key is ``None``, not an exception.** Zarr asks for keys that legitimately do
not exist and expects ``None`` back. Raising instead turns an ordinary absence into a
failed read, and the sharding codec in particular probes for objects that are not there.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterable
from typing import Any

from zarr.abc.store import (  # ty: ignore[unresolved-import]
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype  # ty: ignore[unresolved-import]

from .transport import Transport, TransportError, default_transport


class ReadOnlyStoreError(RuntimeError):
    """Something tried to write through a store that only reads."""


class NemarHttpStore(Store):
    """Zarr store backed by HTTPS range requests against a NEMAR contract base.

    Construct it with a URL from :meth:`~eegprep_lean.DatasetIndex.level0_url` or
    :meth:`~eegprep_lean.DatasetIndex.view_url`, which are built on ``contract_base``.
    Nothing here consults ``data_base``, so a store cannot quietly start reading the
    bucket because a caller passed the wrong base.
    """

    def __init__(self, base_url: str, *, transport: Transport | None = None) -> None:
        super().__init__(read_only=True)
        self.base_url = base_url.rstrip("/")
        self.transport = transport or default_transport()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NemarHttpStore) and other.base_url == self.base_url

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.base_url))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.base_url!r})"

    # Reading -----------------------------------------------------------------

    def _url(self, key: str) -> str:
        return f"{self.base_url}/{key.lstrip('/')}"

    @staticmethod
    def _http_range(byte_range: ByteRequest | None) -> tuple[int | None, int | None]:
        """Translate a Zarr byte request into transport start/end.

        ``RangeByteRequest.end`` is exclusive and the transport's ``end`` is the last byte
        HTTP should return, so it loses one. The other two forms carry no such trap.
        """
        if byte_range is None:
            return None, None
        if isinstance(byte_range, RangeByteRequest):
            return byte_range.start, byte_range.end - 1
        if isinstance(byte_range, OffsetByteRequest):
            return byte_range.offset, None
        if isinstance(byte_range, SuffixByteRequest):
            return None, byte_range.suffix
        raise TypeError(f"unsupported byte request: {byte_range!r}")

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        start, end = self._http_range(byte_range)
        try:
            response = await self.transport.get(self._url(key), start=start, end=end)
        except TransportError as err:
            # Absence is ordinary here: zarr probes for keys that need not exist, and the
            # sharding codec does it routinely. Anything else is a real failure and is
            # left to propagate rather than being flattened into "not found".
            if err.status in (403, 404, 416):
                return None
            raise
        return prototype.buffer.from_bytes(response.body)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        return [await self.get(key, prototype, byte_range) for key, byte_range in key_ranges]

    async def exists(self, key: str) -> bool:
        # One byte is enough to learn whether the object is there, and cheaper than the
        # whole thing for a shard measured in megabytes.
        return await self.get(key, _prototype(), RangeByteRequest(0, 1)) is not None

    # Everything else ---------------------------------------------------------

    @property
    def supports_writes(self) -> bool:
        return False

    @property
    def supports_deletes(self) -> bool:
        return False

    @property
    def supports_partial_writes(self) -> bool:
        return False

    @property
    def supports_listing(self) -> bool:
        """False, and not merely unimplemented.

        Anonymous listing is denied on the serving bucket, including at the root, so
        there is nothing to enumerate. Declaring support and returning an empty listing
        would let a caller conclude a dataset is empty when it is only unlistable.
        """
        return False

    async def set(self, key: str, value: Buffer) -> None:
        raise ReadOnlyStoreError(f"{self.base_url} is a read-only mirror; nothing here writes")

    async def delete(self, key: str) -> None:
        raise ReadOnlyStoreError(f"{self.base_url} is a read-only mirror; nothing here deletes")

    async def list(self) -> AsyncGenerator[str, None]:
        raise NotImplementedError(self._no_listing())
        yield ""  # pragma: no cover - unreachable, makes this an async generator

    async def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
        raise NotImplementedError(self._no_listing())
        yield ""  # pragma: no cover - unreachable, makes this an async generator

    async def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        raise NotImplementedError(self._no_listing())
        yield ""  # pragma: no cover - unreachable, makes this an async generator

    def _no_listing(self) -> str:
        return (
            f"{self.base_url} cannot be listed: anonymous listing is denied on the serving "
            "bucket, so a client discovers what exists from the dataset's index.json rather "
            "than by enumeration"
        )


def _prototype() -> BufferPrototype:
    from zarr.core.buffer import default_buffer_prototype  # ty: ignore[unresolved-import]

    return default_buffer_prototype()


async def open_array(url: str, *, transport: Transport | None = None) -> Any:
    """Open one NEMAR array for reading, asynchronously.

    Deliberately the asynchronous API. ``zarr.open`` is synchronous and starts an IO
    thread, which Pyodide's main thread cannot do; the error it raises names threads
    rather than zarr or the browser, so it does not explain itself.
    """
    from zarr.api.asynchronous import open_array as zarr_open_array  # ty: ignore[unresolved-import]

    return await zarr_open_array(store=NemarHttpStore(url, transport=transport), mode="r")
