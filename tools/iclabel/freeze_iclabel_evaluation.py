"""Download the frozen ICLabel recordings and build float32 feature archives.

The manifest is the authority for recording selection and checksums. This tool
only downloads the public NEMAR objects named by that manifest, runs the fixed
EEGPrep ICA and ICLabel feature path, and writes resumable derived archives.
It does not run a quantized model or select a shipped artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np

from eegprep import ICL_feature_extractor, eeg_runica, pop_loadset

from tools.iclabel.quantize_iclabel_onnx import load_frozen_manifest


COMPONENT_COUNT = 31
DEFAULT_DATA_DIR = Path("/private/tmp/eegprep-phase6-iclabel-data")
DEFAULT_OUTPUT_DIR = Path(__file__).parent
_CHUNK_SIZE = 1024 * 1024


def _verify_recording(path: Path, recording: Mapping[str, object]) -> None:
    expected_size = int(recording["size_bytes"])
    expected_md5 = str(recording["md5"])
    if path.stat().st_size != expected_size:
        raise ValueError(f"Size mismatch for {path}: expected {expected_size} bytes")
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_md5:
        raise ValueError(f"MD5 mismatch for {path}: expected {expected_md5}")


def _download_recording(manifest: Mapping[str, object], recording: Mapping[str, object], path: Path) -> None:
    if path.exists():
        _verify_recording(path, recording)
        return

    dataset = manifest["dataset"]
    url_template = str(dataset["s3_object_url_template"])
    url = url_template.format(git_annex_key=quote(str(recording["git_annex_key"]), safe=""))
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(path.name + ".part")
    request = Request(url, headers={"User-Agent": "EEGPrep-ICLabel-evaluation/1"})
    print(f"DOWNLOAD {recording['source_path']} <- {url}", flush=True)
    with urlopen(request, timeout=180) as response, partial_path.open("wb") as handle:
        while chunk := response.read(_CHUNK_SIZE):
            handle.write(chunk)
    try:
        _verify_recording(partial_path, recording)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    os.replace(partial_path, path)


def _feature_archive_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = [np.asarray(archive[name], dtype=np.float32) for name in ("topo", "psd", "autocorr")]
    except (KeyError, OSError, ValueError):
        return False
    return (
        arrays[0].shape == (32, 32, 1, COMPONENT_COUNT)
        and arrays[1].shape == (1, 100, 1, COMPONENT_COUNT)
        and arrays[2].shape == (1, 100, 1, COMPONENT_COUNT)
        and all(np.isfinite(array).all() for array in arrays)
    )


def _extract_recording_features(raw_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eeg = pop_loadset(str(raw_path))
    print(f"ICA {raw_path}", flush=True)
    eeg = eeg_runica(
        eeg,
        extended=1,
        seed=379,
        rndreset="off",
        maxsteps=512,
        verbose=False,
    )
    print(f"FEATURES {raw_path}", flush=True)
    features = tuple(np.asarray(feature, dtype=np.float32) for feature in ICL_feature_extractor(eeg, True))
    if len(features) != 3:
        raise ValueError(f"ICLabel feature extraction returned {len(features)} arrays for {raw_path}")
    if any(not np.isfinite(feature).all() for feature in features):
        raise ValueError(f"ICLabel feature extraction returned non-finite values for {raw_path}")
    expected_shapes = ((32, 32, 1, COMPONENT_COUNT), (1, 100, 1, COMPONENT_COUNT), (1, 100, 1, COMPONENT_COUNT))
    if tuple(feature.shape for feature in features) != expected_shapes:
        raise ValueError(
            f"Unexpected ICLabel feature shapes for {raw_path}: {tuple(feature.shape for feature in features)}"
        )
    return features


def _recording_features(
    manifest: Mapping[str, object],
    split_name: str,
    recording: Mapping[str, object],
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_path = data_dir / split_name / str(recording["source_path"])
    _download_recording(manifest, recording, raw_path)
    checkpoint = data_dir / "features" / split_name / f"{recording['subject']}.npz"
    if _feature_archive_is_valid(checkpoint):
        print(f"CHECKPOINT {split_name}/{recording['subject']}", flush=True)
        with np.load(checkpoint, allow_pickle=False) as archive:
            return tuple(np.asarray(archive[name], dtype=np.float32) for name in ("topo", "psd", "autocorr"))

    features = _extract_recording_features(raw_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(checkpoint.name + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, topo=features[0], psd=features[1], autocorr=features[2])
    os.replace(temporary, checkpoint)
    return features


def _write_archive(path: Path, features: Sequence[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}-", suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, topo=features[0], psd=features[1], autocorr=features[2])
    os.replace(temporary, path)


def build_split(
    manifest: Mapping[str, object],
    split_name: str,
    data_dir: Path,
    output_path: Path,
    limit: int | None = None,
) -> None:
    split = manifest[split_name]
    recordings = split["recordings"]
    if limit is not None:
        recordings = recordings[:limit]
    extracted = []
    for index, recording in enumerate(recordings, start=1):
        print(f"RECORDING {split_name} {index}/{len(recordings)} {recording['subject']}", flush=True)
        extracted.append(_recording_features(manifest, split_name, recording, data_dir))
    if limit is None and len(extracted) != int(split["component_count"]) // COMPONENT_COUNT:
        raise ValueError(f"{split_name} archive did not process all manifest recordings")
    features = tuple(np.concatenate([recording[index] for recording in extracted], axis=3) for index in range(3))
    expected_count = len(extracted) * COMPONENT_COUNT
    if any(feature.shape[3] != expected_count for feature in features):
        raise ValueError(f"{split_name} archive has the wrong component count")
    _write_archive(output_path, features)
    print(f"ARCHIVE {split_name} {output_path} components={expected_count}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("evaluation_manifest.json"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=("all", "evaluation", "calibration"), default="all")
    parser.add_argument("--limit", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    manifest = load_frozen_manifest(args.manifest)
    splits = ("evaluation", "calibration") if args.split == "all" else (args.split,)
    for split_name in splits:
        output_name = f"{split_name}_features.npz"
        build_split(manifest, split_name, args.data_dir, args.output_dir / output_name, args.limit)
    print(json.dumps({"status": "complete", "splits": list(splits)}), flush=True)


if __name__ == "__main__":
    main()
