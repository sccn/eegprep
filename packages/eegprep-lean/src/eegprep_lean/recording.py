"""Read one whole recording from NEMAR's Zarr serving copy, in one synchronous call.

:func:`load_recording` is :func:`~eegprep_lean.read_index` and
:func:`~eegprep_lean.read_window` composed over the whole level-0 array of one store's
EEG group, in physical units. It adds no second reader: every byte goes through this
package's own index reader, transport, store and window code, so the User-Agent, the byte
ranges and the dequantization are the ones the rest of the package already has.

It is synchronous, by ``asyncio.run``, because its callers are scripts and recipes that
have no event loop. That is why it cannot be called from inside a running one (a
notebook cell, or Pyodide's main thread): ``await`` :func:`~eegprep_lean.read_window`
there instead.

Reading a whole recording is the one place this package can ask for gigabytes, so the
size is checked against ``max_bytes`` from the index's own counts *before* any signal is
fetched, and nothing is ever written to disk.

:func:`index_sha256` is base tier, standard library only. It records which index the
recording was read against, so a result can say what it was computed from.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING

from .extras import is_missing_extra, missing_extra_error
from .index import INDEX_URL_TEMPLATE, ChannelGroup, DatasetIndex, IndexError_, Store, read_index
from .transport import Transport, default_transport

if TYPE_CHECKING:
    from .window import Window

#: The cap on the decoded recording, as float64: channels times samples times 8 bytes.
DEFAULT_MAX_BYTES = 2 * 1024**3

#: Physical level-0 data is held as float64.
_BYTES_PER_SAMPLE = 8


def index_sha256(
    dataset_id: str,
    *,
    transport: Transport | None = None,
    index_url: str | None = None,
) -> str:
    """SHA-256 of the exact index bytes read, as lowercase hexadecimal.

    The bytes as served, not a re-serialization of the parsed document, so the digest can
    be checked against the file by anyone. This is a separate request from
    :func:`~eegprep_lean.read_index`, so a dataset reconverted between the two can make
    the digest describe a newer index than the one a read used; read both close together
    and treat a mismatch on a later run as the dataset having changed.

    ``index_url`` is as for :func:`~eegprep_lean.read_index`. Not callable from inside a
    running event loop.
    """
    client = transport or default_transport()
    url = index_url if index_url is not None else INDEX_URL_TEMPLATE.format(dataset_id=dataset_id)
    response = asyncio.run(client.get(url))
    return hashlib.sha256(response.body).hexdigest()


def _choose_group(store: Store, name: str | None) -> ChannelGroup:
    """The named group, or the store's one EEG group.

    A store can carry a group per modality, and "the recording" of an EEG pipeline is the
    EEG one. Several EEG groups are the same recording at different rates, which
    :meth:`~eegprep_lean.Store.group` refuses to choose between, and so does this.
    """
    if name is not None:
        return store.group(name)
    eeg = [group for group in store.groups if group.modality.upper() == "EEG"]
    if len(eeg) == 1:
        return eeg[0]
    if not eeg and store.groups:
        found = ", ".join(f"{group.name} ({group.modality})" for group in store.groups)
        raise IndexError_(f"{store.path} has no EEG channel group (has: {found}); pass group= to read another")
    return store.group()


async def _load(
    dataset_id: str,
    store_path: str,
    group_name: str | None,
    max_bytes: int,
    transport: Transport | None,
    index: DatasetIndex | None,
) -> Window:
    try:
        from .window import read_window
    except ImportError as err:
        if not is_missing_extra(err, "zarr"):
            raise
        raise missing_extra_error("load_recording", "zarr", err) from err

    if index is None:
        index = await read_index(dataset_id, transport=transport)
    store = index.store(store_path)
    group = _choose_group(store, group_name)

    needed = group.n_channels * group.n_samples * _BYTES_PER_SAMPLE
    if needed > max_bytes:
        raise ValueError(
            f"{dataset_id}/{store_path} ({group.name}) is {group.n_channels} channels by "
            f"{group.n_samples} samples, {needed} bytes as float64, over max_bytes={max_bytes}"
        )
    return await read_window(
        index,
        store,
        start_sample=0,
        n_samples=group.n_samples,
        group=group,
        transport=transport,
    )


def load_recording(
    dataset_id: str,
    store: str,
    *,
    group: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    transport: Transport | None = None,
    index: DatasetIndex | None = None,
) -> Window:
    """Read one whole recording at level 0, in physical units, as a :class:`~eegprep_lean.Window`.

    ``store`` is the recording's source path in the index, for example
    ``sub-01/eeg/sub-01_task-rest_eeg.set``. ``group`` names a channel group; left out, it
    is the store's one EEG group. ``index`` skips fetching the index when the caller
    already holds it. Raises :class:`ValueError` when channels times samples times 8
    exceeds ``max_bytes``, before any signal is read. Needs the ``zarr`` extra, and is not
    callable from inside a running event loop.
    """
    return asyncio.run(_load(dataset_id, store, group, max_bytes, transport, index))
