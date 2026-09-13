"""Import EEGLAB-compatible event tables without an EEG dataset wrapper."""

from __future__ import annotations

from typing import Any

from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.popfunc.pop_importevent import _import_event_records


def importevent(
    event: Any,
    oldevent: Any = None,
    srate: float = 1.0,
    *args: Any,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Import event records from a text table or in-memory rows.

    Latencies use seconds by default and are converted to EEGLAB's 1-based
    sample positions. Pass ``timeunit=float("nan")`` when input latencies are
    already sample positions. Explicit event ``indices`` are 1-based.
    """
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    return _import_event_records(options.get("event", event), oldevent, float(srate), options)


__all__ = ["importevent"]
