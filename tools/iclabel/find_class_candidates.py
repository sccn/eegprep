"""Find ICLabel components in the classes the int8 parity gate never measured.

The frozen evaluation set (``evaluation_manifest.json``) labels 193 components
Other, 18 Brain, 5 Eye, and 1 Muscle. Heart, Line Noise, and Channel Noise do
not occur. This tool runs the float32 network, ``iclabel.onnx``, over other
public CC0 or PDDL NEMAR/OpenNeuro recordings and lists components whose
teacher top-1 is Heart, Line Noise, Channel Noise, or Muscle.

Provenance uses the same facts as a recording entry in that manifest: dataset
id (``dataset.openneuro_source_id`` or ``dataset.nemar_dataset_id``), version
or DOI, path (``source_path``), annex key (``git_annex_key``), the recording
SHA-256 (the manifest's feature archive records SHA-256; its recordings record
MD5), and the ICA method and seed (``preprocessing.ica``).

Recordings from ds002680 / on002680 subjects sub-002 through sub-015 are
refused so a later evaluation set stays disjoint from the frozen one. Hits from
this tool are a measurement, not evaluation evidence, until a committed script
pins the dataset versions and the ICA seed.

Example::

    uv run python -m tools.iclabel.find_class_candidates \\
        --spec recordings.json --data-dir /tmp/iclabel-candidates \\
        --output /tmp/iclabel-candidates/candidates.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

from eegprep import ICL_feature_extractor, eeg_picard, pop_biosig, pop_loadset
from eegprep.functions.popfunc.pop_chanedit import pop_chanedit
from eegprep.functions.popfunc.pop_select import pop_select
from eegprep.functions.sigprocfunc.readlocs import readlocs
from eegprep.plugins.ICLabel.pop_icflag import ICLABEL_CLASSES
from tools.iclabel.quantize_iclabel_onnx import PACKAGED_FLOAT32_MODEL, predict_features


TARGET_CLASSES = ("Heart", "Line Noise", "Channel Noise", "Muscle")
_PROVENANCE_FIELDS = (
    "dataset_id",
    "version_or_doi",
    "path",
    "annex_key",
    "sha256",
    "ica_method",
    "ica_seed",
    "subject",
)
_FROZEN_DATASETS = frozenset({"ds002680", "on002680"})
_FROZEN_SUBJECT_MIN = 2
_FROZEN_SUBJECT_MAX = 15
_ANNEX_KEY = re.compile(r"^(SHA256E|MD5E)-s(\d+)--([0-9a-fA-F]+)(?:\.|$)")
_HASH_CHUNK_SIZE = 1024 * 1024
_STANDARD_LOCS = str(files("eegprep").joinpath("resources/montages/standard-10-5-342ch.locs"))
_SET_SUFFIXES = {".set"}
_BIOSIG_SUFFIXES = {".edf", ".bdf", ".gdf"}


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_annex_key(path: Path, annex_key: str) -> None:
    """Raise unless ``annex_key`` is the git-annex key of the bytes at ``path``."""
    match = _ANNEX_KEY.match(str(annex_key))
    if match is None:
        raise ValueError(
            f"find_class_candidates: annex key {annex_key!r} is not a SHA256E or MD5E git-annex key. "
            "Use the key from the NEMAR dataset repository."
        )
    kind, size_text, expected = match.groups()
    actual_size = Path(path).stat().st_size
    if actual_size != int(size_text):
        raise ValueError(
            f"find_class_candidates: annex key {annex_key} records {size_text} bytes but {path} is "
            f"{actual_size} bytes. Re-download the recording named by that key."
        )
    if kind == "SHA256E":
        actual = file_sha256(path)
        if actual.lower() != expected.lower():
            raise ValueError(
                f"find_class_candidates: annex key {annex_key} does not match {path} "
                f"(sha256 {actual}, size {actual_size}). "
                "Pass the git-annex key of this file, or re-download it."
            )
        return
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.lower() != expected.lower():
        raise ValueError(
            f"find_class_candidates: annex key {annex_key} does not match {path} "
            f"(md5 {actual}, sha256 {file_sha256(path)}, size {actual_size}). "
            "Pass the git-annex key of this file, or re-download it."
        )


def find_class_candidates(
    teacher_probabilities: np.ndarray,
    provenance: Mapping[str, Any],
    *,
    target_classes: Sequence[str] = TARGET_CLASSES,
) -> list[dict[str, Any]]:
    """Return components whose teacher top-1 is in ``target_classes``.

    Each record copies the provenance fields and adds ``component_index``
    (0-based), ``teacher_class``, and ``teacher_probability``.
    """
    _require_provenance(provenance)
    _refuse_frozen_evaluation(str(provenance["dataset_id"]), provenance["subject"])
    probabilities = np.asarray(teacher_probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(ICLABEL_CLASSES):
        shape = tuple(probabilities.shape)
        raise ValueError(
            f"find_class_candidates: teacher probabilities must have shape (components, 7), got {shape}. "
            "Pass one row per component and one column per ICLabel class, in ICLABEL_CLASSES order."
        )
    if probabilities.shape[0] and not np.isfinite(probabilities).all():
        raise ValueError(
            "find_class_candidates: teacher probabilities must be finite. "
            "Recompute them with the float32 ICLabel network."
        )
    if probabilities.shape[0] == 0:
        return []
    unknown = [name for name in target_classes if name not in ICLABEL_CLASSES]
    if unknown:
        raise ValueError(
            f"find_class_candidates: unknown ICLabel classes {unknown}. Use names from {list(ICLABEL_CLASSES)}."
        )
    labels = np.argmax(probabilities, axis=1)
    wanted = {ICLABEL_CLASSES.index(name) for name in target_classes}
    records = []
    for index, label in enumerate(labels.tolist()):
        if label not in wanted:
            continue
        record = {field: provenance[field] for field in _PROVENANCE_FIELDS}
        record["component_index"] = index
        record["teacher_class"] = ICLABEL_CLASSES[label]
        record["teacher_probability"] = float(probabilities[index, label])
        records.append(record)
    return records


def scan_recordings(
    specs: Sequence[Mapping[str, Any]],
    *,
    data_dir: Path,
    model_path: Path = PACKAGED_FLOAT32_MODEL,
    ica_method: str = "picard",
    ica_seed: int = 404,
) -> dict[str, Any]:
    """Download each recording, classify it with float32 ICLabel, and list target components."""
    data_dir = Path(data_dir)
    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for spec in specs:
        path = _materialize_recording(spec, data_dir)
        provenance = _provenance_from_spec(spec, path, ica_method=ica_method, ica_seed=ica_seed)
        _refuse_frozen_evaluation(str(provenance["dataset_id"]), provenance["subject"])
        verify_annex_key(path, str(provenance["annex_key"]))
        try:
            probabilities, features = classify_recording(
                path,
                model_path=model_path,
                ica_method=ica_method,
                ica_seed=ica_seed,
                channel_labels=spec.get("channel_labels"),
                location_from_label=spec.get("location_from_label"),
            )
        except (RuntimeError, ValueError, OSError) as exc:
            failures.append({"path": str(provenance["path"]), "error": str(exc)})
            print(f"SKIP {provenance['path']}: {exc}", flush=True)
            continue
        _write_feature_checkpoint(data_dir, provenance, features, probabilities)
        found = find_class_candidates(probabilities, provenance)
        if spec.get("channel_labels") or spec.get("location_from_label"):
            for record in found:
                if spec.get("channel_labels"):
                    record["channel_labels"] = list(spec["channel_labels"])
                if spec.get("location_from_label"):
                    record["location_from_label"] = dict(spec["location_from_label"])
                if spec.get("channel_label_source"):
                    record["channel_label_source"] = spec["channel_label_source"]
        print(
            f"CLASSIFIED {provenance['dataset_id']} {provenance['subject']} "
            f"components={probabilities.shape[0]} candidates={len(found)}",
            flush=True,
        )
        candidates.extend(found)
    return {
        "model": {
            "path": "src/eegprep/plugins/ICLabel/iclabel.onnx",
            "sha256": file_sha256(model_path),
        },
        "ica_method": ica_method,
        "ica_seed": ica_seed,
        "target_classes": list(TARGET_CLASSES),
        "candidates": candidates,
        "failures": failures,
        "summary": summarize_candidates(candidates),
    }


def classify_recording(
    path: Path,
    *,
    model_path: Path,
    ica_method: str,
    ica_seed: int,
    channel_labels: Sequence[str] | None = None,
    location_from_label: Mapping[str, str] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Run ICA when needed and return float32 class probabilities and ICLabel features."""
    eeg = _load_eeg(path)
    if channel_labels is not None:
        eeg = _rename_channels(eeg, channel_labels, path)
    eeg = _ensure_channel_locations(eeg, path, location_from_label=location_from_label)
    eeg = _decompose(eeg, path, ica_method=ica_method, ica_seed=ica_seed)
    features = tuple(np.asarray(array, dtype=np.float32) for array in ICL_feature_extractor(eeg, True))
    if len(features) != 3:
        raise RuntimeError(
            f"find_class_candidates: ICL_feature_extractor returned {len(features)} arrays for {path}. "
            "Call it with flag_autocorr=True so topo, psd, and autocorr are all present."
        )
    return predict_features(model_path, features), features


def summarize_candidates(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count candidates, subjects, and datasets per target class."""
    summary = {}
    for name in TARGET_CLASSES:
        selected = [record for record in candidates if record["teacher_class"] == name]
        summary[name] = {
            "count": len(selected),
            "subjects": sorted({str(record["subject"]) for record in selected}),
            "datasets": sorted({str(record["dataset_id"]) for record in selected}),
            "subject_count": len({(record["dataset_id"], record["subject"]) for record in selected}),
            "dataset_count": len({record["dataset_id"] for record in selected}),
        }
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, required=True, help="JSON list of recording objects to classify")
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="where downloads and feature checkpoints are stored"
    )
    parser.add_argument("--output", type=Path, required=True, help="where to write the candidate JSON")
    parser.add_argument("--model", type=Path, default=PACKAGED_FLOAT32_MODEL, help="float32 ICLabel ONNX network")
    parser.add_argument("--ica-method", default="picard", choices=("picard", "precomputed"))
    parser.add_argument("--ica-seed", type=int, default=404)
    args = parser.parse_args(argv)

    specs = json.loads(args.spec.read_text(encoding="utf-8"))
    if not isinstance(specs, list):
        raise SystemExit(
            "find_class_candidates: --spec must be a JSON list of recording objects. "
            "Write one object per recording, with dataset_id, version_or_doi, path, annex_key, url, and subject."
        )
    report = scan_recordings(
        specs,
        data_dir=args.data_dir,
        model_path=args.model,
        ica_method=args.ica_method,
        ica_seed=args.ica_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


def _require_provenance(provenance: Mapping[str, Any]) -> None:
    for field in _PROVENANCE_FIELDS:
        if field not in provenance:
            raise ValueError(
                f"find_class_candidates: recording provenance is missing {field!r}. "
                f"Add {field} to the recording record before selecting candidates."
            )


def _refuse_frozen_evaluation(dataset_id: str, subject: Any) -> None:
    if dataset_id.casefold() not in _FROZEN_DATASETS:
        return
    number = _subject_number(subject)
    if number is None or not _FROZEN_SUBJECT_MIN <= number <= _FROZEN_SUBJECT_MAX:
        return
    raise ValueError(
        "find_class_candidates: "
        f"{dataset_id} subject {subject} is one of frozen evaluation subjects sub-002 through sub-015. "
        "Pass a different subject or dataset."
    )


def _subject_number(subject: Any) -> int | None:
    text = str(subject).strip().lower()
    if text.startswith("sub-"):
        text = text[4:]
    if text.isdigit():
        return int(text)
    return None


def _provenance_from_spec(spec: Mapping[str, Any], path: Path, *, ica_method: str, ica_seed: int) -> dict[str, Any]:
    missing = [field for field in ("dataset_id", "version_or_doi", "path", "annex_key", "subject") if field not in spec]
    if missing:
        raise ValueError(
            f"find_class_candidates: recording spec is missing {missing}. "
            "Add dataset_id, version_or_doi, path, annex_key, and subject to each spec entry."
        )
    return {
        "dataset_id": spec["dataset_id"],
        "version_or_doi": spec["version_or_doi"],
        "path": spec["path"],
        "annex_key": spec["annex_key"],
        "sha256": file_sha256(path),
        "ica_method": spec.get("ica_method", ica_method),
        "ica_seed": spec.get("ica_seed", ica_seed),
        "subject": spec["subject"],
    }


def _materialize_recording(spec: Mapping[str, Any], data_dir: Path) -> Path:
    relative = Path(str(spec["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            f"find_class_candidates: recording path {spec['path']!r} must be relative to the dataset root. "
            "Use the BIDS path, such as sub-01/eeg/sub-01_task-rest_eeg.set."
        )
    destination = data_dir / relative
    if destination.exists():
        return destination
    url = spec.get("url")
    if not url:
        raise ValueError(
            f"find_class_candidates: {destination} is not on disk and the spec has no url. "
            "Add the OpenNeuro or NEMAR object URL, or place the file at that path."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    request = Request(str(url), headers={"User-Agent": "EEGPrep-ICLabel-class-candidates/1"})
    print(f"DOWNLOAD {relative} <- {url}", flush=True)
    with urlopen(request, timeout=180) as response, partial.open("wb") as handle:
        for chunk in iter(lambda: response.read(_HASH_CHUNK_SIZE), b""):
            handle.write(chunk)
    os.replace(partial, destination)
    return destination


def _load_eeg(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in _SET_SUFFIXES:
        return pop_loadset(str(path))
    if suffix in _BIOSIG_SUFFIXES:
        return pop_biosig(path)
    raise ValueError(
        f"find_class_candidates: {path} has unsupported suffix {suffix!r}. Pass a .set, .edf, .bdf, or .gdf recording."
    )


def _rename_channels(eeg: dict[str, Any], labels: Sequence[str], path: Path) -> dict[str, Any]:
    channels = _channel_list(eeg)
    if len(labels) != int(eeg["nbchan"]) or len(channels) < int(eeg["nbchan"]):
        raise ValueError(
            f"find_class_candidates: channel_labels for {path} has {len(labels)} names for {eeg['nbchan']} channels. "
            "Pass one label per channel, in file order."
        )
    for channel, label in zip(channels, labels):
        if not isinstance(channel, dict):
            raise ValueError(
                f"find_class_candidates: {path} channel locations are not records, so they cannot be renamed. "
                "Use a recording pop_loadset or pop_biosig loads with a chanlocs list."
            )
        channel["labels"] = str(label)
    return eeg


def _ensure_channel_locations(
    eeg: dict[str, Any],
    path: Path,
    location_from_label: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if not _locations_present(eeg):
        eeg = pop_chanedit(eeg, "lookup", _STANDARD_LOCS)
    if location_from_label:
        _apply_location_donors(eeg, location_from_label, path)
    channels = _channel_list(eeg)
    missing = [index for index, channel in enumerate(channels[: int(eeg["nbchan"])]) if not _finite_coord(channel)]
    if not missing:
        return eeg
    keep = [index for index in range(int(eeg["nbchan"])) if index not in set(missing)]
    dropped = [str(channels[index].get("labels") or "?") for index in missing]
    if len(keep) < 4:
        raise ValueError(
            f"find_class_candidates: {path} channels {dropped} have no coordinates after lookup in {_STANDARD_LOCS}, "
            f"leaving {len(keep)} channels. Use a recording whose labels match the standard 10-5 montage, "
            "or a .set file that already stores coordinates."
        )
    print(f"DROP {path} channels without standard coordinates: {dropped}", flush=True)
    return pop_select(eeg, channel=keep)


def _apply_location_donors(eeg: dict[str, Any], mapping: Mapping[str, str], path: Path) -> None:
    template = readlocs(_STANDARD_LOCS)
    donors = {}
    for channel in template:
        if isinstance(channel, Mapping):
            label = str(channel.get("labels") or "")
            if label:
                donors.setdefault(label.casefold(), channel)
    for channel in _channel_list(eeg):
        if not isinstance(channel, dict) or _finite_coord(channel):
            continue
        label = str(channel.get("labels") or "")
        donor_name = mapping.get(label)
        if donor_name is None:
            continue
        donor = donors.get(str(donor_name).casefold())
        if donor is None:
            raise ValueError(
                f"find_class_candidates: location donor {donor_name!r} for channel {label!r} in {path} "
                f"is not in {_STANDARD_LOCS}. Use a label from that montage."
            )
        for key in ("theta", "radius", "X", "Y", "Z", "sph_theta", "sph_phi", "sph_radius"):
            if key in donor:
                channel[key] = donor[key]


def _locations_present(eeg: dict[str, Any]) -> bool:
    channels = _channel_list(eeg)
    if len(channels) < int(eeg["nbchan"]):
        return False
    return all(_finite_coord(channel) for channel in channels[: int(eeg["nbchan"])])


def _channel_list(eeg: dict[str, Any]) -> list[Any]:
    chanlocs = eeg.get("chanlocs")
    if chanlocs is None:
        return []
    if isinstance(chanlocs, np.ndarray):
        return chanlocs.reshape(-1).tolist()
    return list(chanlocs)


def _finite_coord(channel: Any) -> bool:
    if not isinstance(channel, Mapping):
        return False
    for key in ("X", "x"):
        if key not in channel:
            continue
        value = channel[key]
        if isinstance(value, np.ndarray):
            if value.size != 1:
                return False
            value = value.reshape(-1)[0]
        if value is None or value == "":
            return False
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return False
    return False


def _decompose(eeg: dict[str, Any], path: Path, *, ica_method: str, ica_seed: int) -> dict[str, Any]:
    if ica_method == "precomputed":
        weights = eeg.get("icaweights")
        if weights is None or np.size(weights) == 0:
            raise ValueError(
                f"find_class_candidates: {path} has no ICA weights. "
                "Run it with --ica-method picard, or pass a dataset that already stores icaweights."
            )
        return eeg
    if ica_method != "picard":
        raise ValueError(
            f"find_class_candidates: ICA method {ica_method!r} is not supported. Use 'picard' or 'precomputed'."
        )
    try:
        return eeg_picard(eeg, random_state=ica_seed, verbose=False, max_iter=500)
    except Exception as exc:
        raise RuntimeError(
            f"find_class_candidates: picard did not finish on {path} ({exc}). "
            "Use a recording picard can decompose, or pass --ica-method precomputed when the file already stores ICA."
        ) from exc


def _write_feature_checkpoint(
    data_dir: Path,
    provenance: Mapping[str, Any],
    features: tuple[np.ndarray, np.ndarray, np.ndarray],
    probabilities: np.ndarray,
) -> None:
    directory = data_dir / "features"
    directory.mkdir(parents=True, exist_ok=True)
    stem = Path(str(provenance["path"])).stem
    name = f"{provenance['dataset_id']}_{provenance['subject']}_{stem}.npz"
    temporary = directory / f".{name}.part"
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            topo=features[0],
            psd=features[1],
            autocorr=features[2],
            teacher=np.asarray(probabilities, dtype=np.float32),
        )
    os.replace(temporary, directory / name)


if __name__ == "__main__":
    main()
