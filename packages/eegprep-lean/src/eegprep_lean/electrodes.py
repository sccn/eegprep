"""Where a dataset says each electrode is, as a NEMAR store carries it.

Base tier. Only the standard library and this package's transport, so a session can learn
whether a store declares electrode positions without installing numpy.

When a recording's BIDS dataset has an ``electrodes.tsv``, NEMAR's converter writes three
attributes into the store's root ``zarr.json``: ``electrode_positions``, a map of channel
label to ``[x, y, z]`` copied from the file unchanged, and ``electrode_coordinate_system``
and ``electrode_coordinate_units``, the ``EEGCoordinateSystem`` and ``EEGCoordinateUnits``
of the matching ``coordsystem.json``, each an empty string when there is none. A dataset
with no electrodes file gets none of the three, which is a store that declares no
positions and not a read that failed.

Nothing here converts a unit or an axis: :class:`ElectrodeAttributes` holds what the store
says, in the dataset's own frame, and :func:`~eegprep_lean.locations.chanlocs_from_electrodes`
is the one place that turns it into EEGLAB's.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from .index import DatasetIndex, IndexError_, Store
from .transport import Transport, default_transport


class ElectrodePositionsNotAnObject(IndexError_):
    """A store root states ``electrode_positions`` as something other than an object.

    Its own type, as :class:`~eegprep_lean.UnsupportedFormatVersion` is, so a caller records
    why a read failed without parsing a message. It is a refusal and not a store with no
    positions because NEMAR's locations digest refuses such a store too: ignoring the
    attribute here would treat as positionless a store that the digest cannot name.
    """


@dataclass(frozen=True)
class ElectrodeAttributes:
    """A store's electrode attributes, as the store root states them."""

    positions: dict[str, tuple[float, float, float]]
    """Channel label to ``(x, y, z)`` in the dataset's own unit and frame.

    Only entries of exactly three finite numbers: the converter's JSON writer lets a
    ``NaN`` cell through as a bare token, which Python reads and from which no position can
    be made. Every electrode of the file is here, including ones the recording does not
    have, because the join to a recording's channels is by label and happens later.
    """

    coordinate_system: str
    """The coordsystem file's ``EEGCoordinateSystem``, or the empty string."""

    coordinate_units: str
    """The coordsystem file's ``EEGCoordinateUnits``, or the empty string."""

    source: str | None = None
    """``electrode_positions_source``: whether the file names a subject or the dataset,
    when the converter that wrote the store says. None for a store that does not."""

    file: str | None = None
    """``electrode_positions_file``: the electrodes file the positions came from, when the
    converter that wrote the store says. None for a store that does not."""

    @classmethod
    def from_attributes(cls, attributes: dict[str, Any]) -> ElectrodeAttributes | None:
        """The electrode attributes of a store root's ``attributes``, or None when it has no
        position map, which is a dataset with no electrodes file.

        A position map that is not an object raises :class:`ElectrodePositionsNotAnObject`
        rather than being ignored: ignoring it would hand back a recording with no locations
        and no sign that the store declared some.
        """
        stored = attributes.get("electrode_positions")
        if stored is None:
            return None
        if not isinstance(stored, dict):
            raise ElectrodePositionsNotAnObject(
                f"electrode_positions in the store root is a {type(stored).__name__}, "
                "not an object of channel label to [x, y, z]"
            )
        positions = {}
        for label, value in stored.items():
            position = _position(value)
            if position is not None:
                positions[str(label)] = position
        return cls(
            positions=positions,
            coordinate_system=_text(attributes.get("electrode_coordinate_system")),
            coordinate_units=_text(attributes.get("electrode_coordinate_units")),
            source=_text(attributes.get("electrode_positions_source")) or None,
            file=_text(attributes.get("electrode_positions_file")) or None,
        )


def _position(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    # bool is an int in Python, and True is not a coordinate.
    if not all(isinstance(axis, (int, float)) and not isinstance(axis, bool) for axis in value):
        return None
    try:
        # Adding 0.0 turns -0.0 into 0.0. NEMAR's locations digest hashes every zero as +0, and left
        # alone the sign of a zero would set the sign of Y and theta further on.
        x, y, z = (float(axis) + 0.0 for axis in value)
    except OverflowError:
        # An integer too large for a float is an infinity to a reader of binary64, so no position.
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    return x, y, z


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


async def read_electrode_attributes(
    index: DatasetIndex,
    store: Store,
    *,
    transport: Transport | None = None,
) -> ElectrodeAttributes | None:
    """Fetch a store's electrode attributes, or None when it declares no positions.

    One small JSON document, the store's root ``zarr.json``, a few KiB. Built on
    ``contract_base`` like every other URL here, never on ``data_base``.
    :func:`~eegprep_lean.load_recording` makes this read and puts the result in
    :attr:`~eegprep_lean.Window.electrodes`; a caller that reads windows itself makes it
    once per store and passes the result to ``dataclasses.replace(window, electrodes=...)``.
    """
    client = transport or default_transport()
    url = f"{index.store_url(store)}/zarr.json"
    response = await client.get(url)
    try:
        # Integers are read as binary64, as JSON.parse reads them and as NEMAR's locations digest
        # is defined: an integer of any length is a float, where json's default raises ValueError
        # from 4300 digits, before any entry could be judged.
        document = json.loads(response.body, parse_int=float)
    except json.JSONDecodeError as err:
        raise IndexError_(f"store root at {url} is not JSON: {err}") from err
    return ElectrodeAttributes.from_attributes(document.get("attributes") or {})
