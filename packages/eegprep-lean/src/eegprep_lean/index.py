"""Read a NEMAR dataset's Zarr index, and resolve paths from it.

The index contract is published at https://docs.nemar.org/platform/zarr/index-contract/
and this module is written against it rather than against any observed response. Three of
its rules are load-bearing enough to state here, because breaking any of them fails
quietly rather than loudly:

**Read ``format_version`` first.** Version 3 is current, but an older index stays served
until its dataset is reconverted, so a reader that assumes the current shape is wrong
rather than merely unlucky.

**``contract_base`` is the only URL that may be hardcoded.** The index also publishes
``data_base``, which names the bucket the bytes physically sit in today. This reader does
not follow it. ``zarr.nemar.org`` is the contract, the redirect behind it is an
implementation detail, and a client that reads the bucket directly is holding a URL that
was never promised to keep working, while making its reads invisible to the archive that
serves them.

**``layout`` is declared ``const`` in the schema**, so once ``format_version`` is checked
the templates can be filled in rather than probed. There is nothing to discover by
guessing, and no directory listing to fall back on: anonymous listing is denied on the
bucket, so a 403 means nothing about whether a thing exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .transport import Transport, default_transport

#: The contract this reader implements. An index declaring anything else is refused
#: rather than guessed at.
SUPPORTED_FORMAT_VERSION = 3

#: Where a dataset's index lives. This is the one URL shape a client may build itself;
#: everything else comes out of the document.
INDEX_URL_TEMPLATE = "https://zarr.nemar.org/{dataset_id}/zarr/index.json"


class IndexError_(RuntimeError):
    """The index could not be read, or does not say what this reader can act on."""


class UnsupportedFormatVersion(IndexError_):
    """The index is a version this reader was not written against.

    Deliberately its own type. Meeting an older index is an expected condition with a
    real answer (implement that version, or tell the user this dataset is not ready),
    not a malformed document.
    """

    def __init__(self, found: Any, supported: int = SUPPORTED_FORMAT_VERSION) -> None:
        super().__init__(
            f"index declares format_version {found!r}, and this reader implements {supported}. "
            "Older indexes stay served until a dataset is reconverted, so this is a dataset "
            "that has not caught up rather than a broken document."
        )
        self.found = found
        self.supported = supported


@dataclass(frozen=True)
class ChannelGroup:
    """One concurrent stream of a recording, at one sampling rate."""

    name: str
    modality: str
    rate: float
    n_channels: int
    n_samples: int
    n_view_levels: int

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.rate if self.rate else 0.0

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> "ChannelGroup":
        return cls(
            name=str(entry["name"]),
            modality=str(entry.get("modality", "")),
            rate=float(entry.get("rate") or 0.0),
            n_channels=int(entry.get("n_channels") or 0),
            n_samples=int(entry.get("n_samples") or 0),
            n_view_levels=int(entry.get("n_view_levels") or 0),
        )


@dataclass(frozen=True)
class Store:
    """One recording's Zarr store, as the index describes it."""

    path: str
    zarr: str
    groups: tuple[ChannelGroup, ...]

    def group(self, name: str | None = None) -> ChannelGroup:
        """Pick a channel group by name, or the only one when there is exactly one.

        Refuses to guess between several. A store's groups are the same recording at
        different rates, so silently taking the first would return real data at the wrong
        sampling rate, which is the kind of wrong that survives a plot.
        """
        if name is not None:
            for group in self.groups:
                if group.name == name:
                    return group
            available = ", ".join(g.name for g in self.groups) or "none"
            raise IndexError_(f"no channel group {name!r} in {self.path} (has: {available})")
        if len(self.groups) == 1:
            return self.groups[0]
        if not self.groups:
            raise IndexError_(
                f"{self.path} declares no channel groups, so there is nothing to read. "
                "The index lists this store, which means conversion produced no readable "
                "stream rather than that the recording is missing."
            )
        available = ", ".join(g.name for g in self.groups)
        raise IndexError_(
            f"{self.path} has {len(self.groups)} channel groups and no name was given; "
            f"they are the same recording at different rates, so pick one: {available}"
        )

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> "Store":
        return cls(
            path=str(entry["path"]),
            zarr=str(entry["zarr"]),
            groups=tuple(ChannelGroup.from_entry(g) for g in entry.get("groups", [])),
        )


@dataclass(frozen=True)
class DatasetIndex:
    """A dataset's served Zarr copy, as declared by its own index document."""

    dataset_id: str
    format_version: int
    contract_base: str
    store_count: int
    stores: tuple[Store, ...]

    def store(self, path: str) -> Store:
        """The store for a recording's source path, for example a ``.set`` file."""
        for store in self.stores:
            if store.path == path:
                return store
        raise IndexError_(f"no store for {path!r} in {self.dataset_id}")

    def level0_url(self, store: Store, group: ChannelGroup | None = None) -> str:
        """The full-resolution array's URL, from ``layout.level0``.

        Built on ``contract_base``. The bucket named by ``data_base`` is deliberately not
        used: see this module's docstring.
        """
        chosen = group or store.group()
        return f"{self.contract_base}{store.zarr}/{chosen.name}/0"

    def view_url(self, store: Store, level: int, group: ChannelGroup | None = None) -> str:
        """A downsampled view's URL, from ``layout.view``.

        Views are numbered from 1; level 0 is the full-resolution array and has its own
        template, which is why asking for view 0 is a mistake rather than a synonym.
        """
        chosen = group or store.group()
        if not 1 <= level <= chosen.n_view_levels:
            raise IndexError_(
                f"view level {level} is outside 1..{chosen.n_view_levels} for "
                f"{chosen.name}; level 0 is the full-resolution array, use level0_url"
            )
        return f"{self.contract_base}{store.zarr}/{chosen.name}/view/{level}"


async def read_index(
    dataset_id: str,
    *,
    transport: Transport | None = None,
) -> DatasetIndex:
    """Fetch and validate a dataset's Zarr index.

    Raises :class:`UnsupportedFormatVersion` for an index this reader does not implement,
    rather than reading it as though it were the current shape.
    """
    client = transport or default_transport()
    url = INDEX_URL_TEMPLATE.format(dataset_id=dataset_id)
    response = await client.get(url)
    try:
        document = json.loads(response.body)
    except json.JSONDecodeError as err:
        raise IndexError_(f"index at {url} is not JSON: {err}") from err

    # Before anything else, per the contract.
    found = document.get("format_version")
    if found != SUPPORTED_FORMAT_VERSION:
        raise UnsupportedFormatVersion(found)

    contract_base = document.get("contract_base")
    if not contract_base:
        raise IndexError_(f"index at {url} declares no contract_base")
    if not contract_base.endswith("/"):
        contract_base = f"{contract_base}/"

    return DatasetIndex(
        dataset_id=str(document.get("dataset_id", dataset_id)),
        format_version=int(found),
        contract_base=str(contract_base),
        # store_count is authoritative; stores[] can be a page of it.
        store_count=int(document.get("store_count") or 0),
        stores=tuple(Store.from_entry(entry) for entry in document.get("stores", [])),
    )
