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
fetched, and nothing is ever written to disk. The check counts peak memory, not just the
decoded array: see :data:`PEAK_BYTES_PER_SAMPLE`.

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

#: Peak resident bytes per channel-sample of reading a recording and then calling
#: :func:`~eegprep_lean.to_eeg` on it, which is what ``max_bytes`` is counted in.
#:
#: Reading peaks at 10: the stored ``int16`` counts (2) are alive while their ``float64``
#: conversion (8) is built in place. ``to_eeg`` then peaks higher, at 8 for the window
#: still held by the caller, plus 8 for the dictionary's own ``data``, plus the ``times``
#: vector of 8 bytes per time point, which is 8 per channel-sample only for one channel
#: and is counted that way here as the worst case: 8 + 8 + 8 = 24. Measured with
#: ``tracemalloc`` on a 33-channel recording, reading peaked at 10.0 bytes per
#: channel-sample and ``to_eeg`` at 16.5, inside this bound.
#:
#: ``to_mne_raw`` is not inside it: MNE keeps its own time vector and bookkeeping, and
#: the volts copy and MNE's side measured about 42 bytes per channel-sample in all. A
#: caller going to MNE passes a ``max_bytes`` a third of what it can afford.
PEAK_BYTES_PER_SAMPLE = 24

#: What ``max_bytes`` is when a caller names none: peak resident memory, 2 GiB.
DEFAULT_MAX_BYTES = 2 * 1024**3


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

    ``index_url`` is as for :func:`~eegprep_lean.read_index`; pass the one given to
    :func:`load_recording` so both describe the same document. Not callable from inside a
    running event loop (``RuntimeError``).
    """
    client = transport or default_transport()
    url = index_url if index_url is not None else INDEX_URL_TEMPLATE.format(dataset_id=dataset_id)
    response = _run_sync("index_sha256", lambda: client.get(url))
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


def _run_sync(name: str, make):
    """Run ``make()``'s coroutine to completion, or say why this cannot be done here.

    Checked before the coroutine exists, so refusing inside a running loop leaves no
    coroutine that was never awaited behind it.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make())
    raise RuntimeError(
        f"{name} is synchronous and cannot be called from a running event loop (a notebook cell, "
        "or the browser's main thread); await read_index and read_window instead"
    )


async def _load(
    dataset_id: str,
    store_path: str,
    group_name: str | None,
    max_bytes: int,
    transport: Transport | None,
    index: DatasetIndex | None,
    index_url: str | None,
) -> Window:
    try:
        from .window import read_window
    except ImportError as err:
        if not is_missing_extra(err, "zarr"):
            raise
        raise missing_extra_error("load_recording", "zarr", err) from err

    if index is None:
        index = await read_index(dataset_id, transport=transport, index_url=index_url)
    store = index.store(store_path)
    group = _choose_group(store, group_name)

    needed = group.n_channels * group.n_samples * PEAK_BYTES_PER_SAMPLE
    if needed > max_bytes:
        raise ValueError(
            f"{dataset_id}/{store_path} ({group.name}) is {group.n_channels} channels by "
            f"{group.n_samples} samples, {needed} bytes at peak ({PEAK_BYTES_PER_SAMPLE} per "
            f"channel-sample), over max_bytes={max_bytes}"
        )
    window = await read_window(
        index,
        store,
        start_sample=0,
        n_samples=group.n_samples,
        group=group,
        transport=transport,
    )
    # read_window clamps to the array's real length, so a store reconverted shorter than
    # its index says comes back as a smaller window and not as an error. A whole recording
    # that is not the whole recording must not pass as one.
    expected = (group.n_channels, group.n_samples)
    if window.data.shape != expected:
        raise IndexError_(
            f"{dataset_id}/{store_path} ({group.name}) was read as {window.data.shape[0]} channels by "
            f"{window.data.shape[1]} samples, and its index says {expected[0]} by {expected[1]}. "
            "The store and its index disagree, which is a dataset reconverted since the index was read."
        )
    return window


def load_recording(
    dataset_id: str,
    store: str,
    *,
    group: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    transport: Transport | None = None,
    index: DatasetIndex | None = None,
    index_url: str | None = None,
) -> Window:
    """Read one whole recording at level 0, in physical units, as a :class:`~eegprep_lean.Window`.

    ``store`` is the recording's source path in the index, for example
    ``sub-01/eeg/sub-01_task-rest_eeg.set``. ``group`` names a channel group; left out, it
    is the store's one EEG group. ``index`` skips fetching the index when the caller
    already holds it, and then ``index_url`` is unused; otherwise ``index_url`` is as for
    :func:`~eegprep_lean.read_index`, and :func:`index_sha256` should be given the same one.
    Raises :class:`ValueError` when channels times samples times
    :data:`PEAK_BYTES_PER_SAMPLE` exceeds ``max_bytes``, before any signal is read, and
    :class:`~eegprep_lean.IndexError_` when what was read is not the size the index
    declares. Needs the ``zarr`` extra. Not callable from inside a running event loop
    (``RuntimeError``).
    """
    return _run_sync("load_recording", lambda: _load(dataset_id, store, group, max_bytes, transport, index, index_url))
