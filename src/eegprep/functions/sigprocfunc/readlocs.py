"""Read EEGLAB-style channel-location files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc._pop_utils import parse_key_value_args, parse_numeric_sequence
from eegprep.functions.sigprocfunc.convertlocs import convertlocs


CHANNEL_FORMATS: dict[str, dict[str, Any]] = {
    "loc": {"format": ("channum", "theta", "radius", "labels"), "skipline": 0},
    "sph": {"format": ("channum", "sph_theta", "sph_phi", "labels"), "skipline": 0},
    "sfp": {"format": ("labels", "-Y", "X", "Z"), "skipline": 0},
    "xyz": {"format": ("channum", "-Y", "X", "Z", "labels"), "skipline": 0},
    "besa": {"format": ("type", "labels", "sph_theta_besa", "sph_phi_besa", "sph_radius"), "skipline": 0},
    "chanedit": {
        "format": (
            "channum",
            "labels",
            "theta",
            "radius",
            "X",
            "Y",
            "Z",
            "sph_theta",
            "sph_phi",
            "sph_radius",
            "type",
        ),
        "skipline": 1,
    },
    "tsv": {"format": ("labels", "X", "Y", "Z"), "skipline": 1},
    "txt": {"format": ("labels", "sph_theta_besa", "sph_phi_besa"), "skipline": 0},
}

LIST_COLUMN_FORMATS = (
    "labels",
    "channum",
    "theta",
    "radius",
    "sph_theta",
    "sph_phi",
    "sph_radius",
    "sph_theta_besa",
    "sph_phi_besa",
    "gain",
    "calib",
    "type",
    "X",
    "Y",
    "Z",
    "-X",
    "-Y",
    "-Z",
    "custom1",
    "custom2",
    "custom3",
    "custom4",
    "ignore",
)

FIDUCIAL_LABELS = {
    "nz",
    "nasion",
    "nazion",
    "fidnz",
    "fidt9",
    "fidt10",
    "lpa",
    "rpa",
    "left",
    "right",
    "nas",
    "lht",
    "rht",
    "lhj",
    "rhj",
    "fiducial",
}


def readlocs(filename: Any, *args: Any, return_outputs: bool = False, **kwargs: Any) -> Any:
    """Read a channel-location file or structure into EEGPrep dictionaries."""
    if isinstance(filename, str) and filename.lower() == "getinfos":
        return _format_infos(), list(LIST_COLUMN_FORMATS)

    options = parse_key_value_args(args, kwargs, lowercase_keys=True, lowercase_kwargs=True)
    locs = _read_locs(filename, options)
    if return_outputs:
        labels = [loc.get("labels", "") for loc in locs]
        theta = np.asarray([loc.get("theta", np.nan) for loc in locs], dtype=float)
        radius = np.asarray([loc.get("radius", np.nan) for loc in locs], dtype=float)
        indices = [
            index + 1 for index, loc in enumerate(locs) if _has_field(loc, "theta") and _has_field(loc, "radius")
        ]
        return locs, labels, theta, radius, indices
    return locs


def readelp(filename: str | Path) -> list[dict[str, Any]]:
    """Read a Polhemus ``.elp`` electrode-position file."""
    locs: list[dict[str, Any]] = []
    pending_label: str | None = None
    fid_labels = ["Nz", "LPA", "RPA"]
    fid_index = 0
    for raw_line in Path(filename).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if pending_label is not None:
            values = _numbers(line)
            if len(values) >= 3:
                locs.append({"labels": pending_label, "X": values[0], "Y": values[1], "Z": values[2], "type": "EEG"})
                pending_label = None
            continue
        if line.startswith("%F"):
            values = _numbers(line[2:])
            if len(values) >= 3:
                label = fid_labels[fid_index] if fid_index < len(fid_labels) else f"FID{fid_index + 1}"
                locs.append({"labels": label, "X": values[0], "Y": values[1], "Z": values[2], "type": "FID"})
                fid_index += 1
        elif line.startswith("%N"):
            label = line[2:].strip().split()
            if label and label[0] not in {"Name", "Status"}:
                pending_label = label[0]
    return convertlocs(locs, "cart2all")


def readeetraklocs(filename: str | Path) -> list[dict[str, Any]]:
    """Read a simple EETrak/ASA ``.elc`` location file."""
    rows = _read_text_rows(Path(filename), 0)
    labels_at = _first_row(rows, "Labels")
    positions_at = _first_row(rows, "Positions")
    if labels_at is None or positions_at is None or labels_at <= positions_at:
        raise ValueError("Could not find 'Labels' and 'Positions' sections")
    position_rows = rows[positions_at + 1 : labels_at]
    label_rows = rows[labels_at + 1 :]
    labels = [item for row in label_rows for item in row]
    locs = []
    for index, row in enumerate(position_rows):
        values = [_coerce_value(item) for item in row if item != ":"]
        numeric = [float(item) for item in values if isinstance(item, (int, float))]
        if len(numeric) >= 3:
            locs.append(
                {
                    "labels": labels[index] if index < len(labels) else f"E{index + 1}",
                    "X": numeric[0],
                    "Y": numeric[1],
                    "Z": numeric[2],
                }
            )
    return convertlocs(locs, "cart2all")


def _read_locs(source: Any, options: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(source, dict) and "chanlocs" in source:
        return chanlocs_as_list(source["chanlocs"])
    if isinstance(source, (dict, list, tuple, np.ndarray)) and not isinstance(source, (str, bytes, Path)):
        return chanlocs_as_list(source)

    path = Path(source)
    filetype = _resolve_filetype(path, options)
    if filetype == "mat" or _looks_like_mat_file(path):
        locs = _read_mat_locations(path)
    elif filetype in {"polhemus", "polhemusx", "polhemusy"}:
        locs = readelp(path)
        if filetype == "polhemusy":
            for loc in locs:
                loc["X"], loc["Y"] = loc.get("Y", ""), loc.get("X", "")
            locs = convertlocs(locs, "cart2all")
    elif filetype == "elc":
        locs = readeetraklocs(path)
    else:
        locs = _read_text_locations(path, filetype, options)

    readchans = options.get("readchans", options.get("elecind"))
    if readchans not in (None, "", []):
        indices = [int(index) - 1 for index in parse_numeric_sequence(readchans, dtype=int)]
        locs = [locs[index] for index in indices]
    return locs


def _read_text_locations(path: Path, filetype: str, options: dict[str, Any]) -> list[dict[str, Any]]:
    format_option = options.get("format")
    if format_option:
        fields = tuple(str(item) for item in format_option)
        skipline = int(options.get("skiplines") or 0)
    else:
        if filetype not in CHANNEL_FORMATS:
            raise ValueError("readlocs could not detect file type; pass filetype='custom' and format=[...]")
        info = CHANNEL_FORMATS[filetype]
        fields = tuple(info["format"])
        skipline = int(options.get("skiplines") if options.get("skiplines") is not None else info["skipline"])

    rows = _read_text_rows(path, skipline)
    if not rows:
        return []
    rows = [row for row in rows if not _is_dash_row(row)]
    if rows and _is_header_row(rows[0]):
        fields = tuple(_canonical_field(item) for item in rows.pop(0))
        rows = [row for row in rows if not _is_dash_row(row)]

    locs: list[dict[str, Any]] = []
    for row in rows:
        if len(row) < len(fields):
            if filetype != "chanedit":
                raise ValueError("Channel-location row has fewer columns than the selected format")
            row = [*row, *([""] * (len(fields) - len(row)))]
        loc: dict[str, Any] = {}
        for field, value in zip(fields, row):
            canonical, multiplier = _field_and_multiplier(field)
            if canonical == "ignore":
                continue
            parsed = _parse_channel_value(canonical, value)
            loc[canonical] = parsed * multiplier if isinstance(parsed, (int, float)) else parsed
        locs.append(loc)

    locs = _postprocess_locations(locs)
    return locs


def _postprocess_locations(locs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not locs:
        return []
    if any("sph_theta_besa" in loc for loc in locs):
        locs = convertlocs(locs, "sphbesa2all")
    elif any("sph_theta" in loc for loc in locs):
        locs = convertlocs(locs, "sph2all")
    elif any("X" in loc for loc in locs):
        locs = convertlocs(locs, "cart2all")
    elif any("theta" in loc for loc in locs):
        locs = convertlocs(locs, "topo2all")

    if all("labels" not in loc or loc.get("labels") in ("", None) for loc in locs):
        for index, loc in enumerate(locs, start=1):
            loc["labels"] = f"E{index}"
    for loc in locs:
        loc["labels"] = str(loc.get("labels", "")).replace(".", "")
        if loc["labels"].lower() in FIDUCIAL_LABELS:
            loc["type"] = "FID"

    if all("channum" in loc for loc in locs):
        if not all(isinstance(loc["channum"], (int, float)) for loc in locs):
            raise ValueError("Channel numbers must be numeric")
        locs = sorted(locs, key=lambda loc: float(loc["channum"]))
        for loc in locs:
            loc.pop("channum", None)
    for loc in locs:
        loc.pop("ignore", None)
    return locs


def _read_mat_locations(path: Path) -> list[dict[str, Any]]:
    data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    if "chanlocs" in data:
        return _postprocess_locations(_mat_chanlocs_to_dicts(data["chanlocs"]))
    if "coordinates" in data and "labels" in data:
        coordinates = np.asarray(data["coordinates"], dtype=float)
        labels = np.asarray(data["labels"], dtype=object).reshape(-1)
        locs = [
            {"labels": str(labels[index]), "X": float(row[0]), "Y": float(row[1]), "Z": float(row[2])}
            for index, row in enumerate(coordinates)
        ]
        return _postprocess_locations(locs)
    raise ValueError(f"No channel-location variables found in {path}")


def _mat_chanlocs_to_dicts(value: Any) -> list[dict[str, Any]]:
    values = np.asarray(value, dtype=object).reshape(-1)
    locs = []
    for item in values:
        if hasattr(item, "_fieldnames"):
            locs.append({field: _mat_value(getattr(item, field)) for field in item._fieldnames})
        elif isinstance(item, dict):
            locs.append({str(key): _mat_value(val) for key, val in item.items()})
    return locs


def _mat_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.size == 1:
            return _mat_value(value.item())
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_text_rows(path: Path, skipline: int) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()[int(skipline) :]
    rows = []
    for line in lines:
        raw_text = line.split("%", 1)[0].split("#", 1)[0]
        if not raw_text.strip():
            continue
        text = raw_text.strip(" \r\n")
        if "," in text:
            row = [item.strip() for item in next(csv.reader([text], skipinitialspace=True))]
        elif "\t" in text:
            row = [item.strip() for item in next(csv.reader([text], delimiter="\t"))]
        else:
            row = text.split()
        if any(item != "" for item in row):
            rows.append(row)
    return rows


def _resolve_filetype(path: Path, options: dict[str, Any]) -> str:
    if options.get("format"):
        return "custom"
    requested = str(options.get("filetype") or "").strip().lower()
    if requested and requested != "autodetect":
        return {"locs": "loc", "eloc": "loc"}.get(requested, requested)
    suffix = path.suffix.lower().lstrip(".")
    return {
        "locs": "loc",
        "eloc": "loc",
        "ced": "chanedit",
        "elp": str(options.get("defaultelp") or "polhemus").lower(),
        "eps": "besa",
    }.get(suffix, suffix)


def _looks_like_mat_file(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(6) == b"MATLAB"


def _field_and_multiplier(field: str) -> tuple[str, int]:
    canonical = _canonical_field(field)
    if canonical.startswith("-"):
        return canonical[1:], -1
    return canonical, 1


def _canonical_field(field: str) -> str:
    field = str(field).strip()
    lower = field.lower()
    if lower in {"x", "y", "z"}:
        return field.upper()
    if lower in {"-x", "-y", "-z"}:
        return "-" + field[-1].upper()
    return lower


def _parse_channel_value(field: str, value: str) -> Any:
    if field == "labels" or field == "type":
        return str(value)
    return _coerce_value(value)


def _coerce_value(value: Any) -> Any:
    text = str(value).strip()
    if text == "":
        return ""
    try:
        numeric = float(text)
    except ValueError:
        return text
    return int(numeric) if numeric.is_integer() else numeric


def _is_header_row(row: list[str]) -> bool:
    canonical = {_canonical_field(item).lstrip("-") for item in row}
    known = {_canonical_field(item).lstrip("-") for item in LIST_COLUMN_FORMATS}
    return bool(canonical) and canonical <= known and any(item in known for item in canonical)


def _is_dash_row(row: list[str]) -> bool:
    return all(set(item) <= {"-"} for item in row)


def _has_field(loc: dict[str, Any], field: str) -> bool:
    try:
        return field in loc and np.isfinite(float(loc[field]))
    except (TypeError, ValueError):
        return False


def _numbers(text: str) -> list[float]:
    values = []
    for token in text.replace(",", " ").split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _first_row(rows: list[list[str]], label: str) -> int | None:
    for index, row in enumerate(rows):
        if row and row[0].lower() == label.lower():
            return index
    return None


def _format_infos() -> list[dict[str, Any]]:
    return [
        {"type": key, "importformat": list(value["format"]), "skipline": value["skipline"]}
        for key, value in CHANNEL_FORMATS.items()
    ]


__all__ = ["CHANNEL_FORMATS", "LIST_COLUMN_FORMATS", "readelp", "readeetraklocs", "readlocs"]
