"""Import Presentation LOG events into an EEGPrep EEG dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args
from eegprep.functions.popfunc.pop_importevent import pop_importevent


def pop_importpres(
    EEG: dict[str, Any],
    filename: str | None = None,
    *args: Any,
    return_com: bool = False,
    **kwargs: Any,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import a Presentation LOG file using its named event columns."""
    if filename is None:
        filename = kwargs.pop("filename", None)
    if filename is None:
        raise ValueError("pop_importpres requires a Presentation LOG filename")
    typefield, latfield, durfield, align, remaining, legacy_count = _legacy_arguments(args)
    options = parse_key_value_args(remaining, kwargs, lowercase_kwargs=True)
    typefield = str(options.pop("typefield", typefield or "code"))
    latfield = str(options.pop("latfield", latfield or "time"))
    durfield = str(options.pop("durfield", durfield or ""))
    if "align" not in options:
        options["align"] = align

    skipline = int(options.pop("skipline", 0) or 0)
    explicit_fields = "fields" in options
    records = None
    if not explicit_fields:
        records = _presentation_records(filename, typefield, latfield, durfield, skipline)
    if records is None:
        if legacy_count and not explicit_fields:
            raise ValueError(f"Could not detect Presentation fields {typefield!r} and {latfield!r}")
        event_source: Any = filename
        options.setdefault("fields", ["type", "latency"])
        options.setdefault("timeunit", float("nan"))
        if skipline:
            options["skipline"] = skipline
    else:
        event_source = records
        options.setdefault("timeunit", 1e-4)

    eeg, _command = pop_importevent(EEG, "event", event_source, return_com=True, **options)
    history_arguments = [format_history_value(filename)]
    if legacy_count:
        history_arguments.extend(
            [
                format_history_value(typefield),
                format_history_value(latfield),
                format_history_value(durfield),
                format_history_value(align),
            ]
        )
    command = f"EEG = pop_importpres(EEG, {', '.join(history_arguments)});"
    eeg["history"] = command if not EEG.get("history") else f"{EEG['history'].rstrip()}\n{command}"
    return (eeg, command) if return_com else eeg


def _legacy_arguments(args: tuple[Any, ...]) -> tuple[Any, Any, Any, Any, tuple[Any, ...], int]:
    values = list(args[:4])
    legacy_count = len(values)
    values.extend([None] * (4 - len(values)))
    typefield, latfield, durfield, align = values
    if durfield is not None and not isinstance(durfield, str):
        align = durfield
        durfield = None
        legacy_count = min(legacy_count, 3)
    if align is None:
        align = 0
    return typefield, latfield, durfield, align, args[legacy_count:], legacy_count


def _presentation_records(
    filename: str | Path,
    typefield: str,
    latfield: str,
    durfield: str,
    skipline: int,
) -> list[dict[str, Any]] | None:
    with Path(filename).open(encoding="utf-8-sig", errors="replace", newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))
    expected = {typefield.casefold(), latfield.casefold()}
    start = max(skipline, 0)
    header_index = next(
        (
            index
            for index, row in enumerate(rows[start:], start=start)
            if expected.issubset({value.strip().casefold() for value in row})
        ),
        None,
    )
    if header_index is None:
        return None
    header = [value.strip() for value in rows[header_index]]
    renamed = [_presentation_field_name(value, typefield, latfield, durfield) for value in header]
    records = []
    for row in rows[header_index + 1 :]:
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) != len(header):
            raise ValueError("Presentation LOG rows must have the same number of columns as the header")
        records.append(dict(zip(renamed, (_coerce_presentation_value(value) for value in row))))
    return records


def _presentation_field_name(name: str, typefield: str, latfield: str, durfield: str) -> str:
    lowered = name.casefold()
    if lowered == typefield.casefold():
        return "type"
    if lowered == latfield.casefold():
        return "latency"
    if durfield and durfield.casefold() != "none" and lowered == durfield.casefold():
        return "duration"
    normalized = name.replace(" ", "_")
    if normalized.endswith(")") and "(" in normalized and not normalized.startswith("("):
        normalized = normalized[: normalized.rfind("(")]
    return normalized


def _coerce_presentation_value(value: str) -> Any:
    stripped = value.strip()
    try:
        number = float(stripped)
    except ValueError:
        return stripped
    return int(number) if number.is_integer() else number
