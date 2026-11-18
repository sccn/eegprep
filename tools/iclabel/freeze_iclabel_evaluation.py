"""Download the frozen ICLabel evaluation recordings and build their float32 feature archive.

The manifest is the authority for recording selection and checksums. This tool
only downloads the public NEMAR objects named by that manifest, runs the fixed
EEGPrep ICA and ICLabel feature path, and writes resumable derived archives.
It does not run any ICLabel network.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import quote
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from eegprep import ICL_feature_extractor, eeg_runica, pop_loadset

from tools.iclabel.quantize_iclabel_onnx import (
    COMPONENTS_PER_RECORDING,
    DEFAULT_EVALUATION_FEATURES,
    load_frozen_manifest,
)


DEFAULT_OUTPUT_DIR = Path(__file__).parent
_CHUNK_SIZE = 1024 * 1024
_FEATURE_SHAPES = (
    (32, 32, 1, COMPONENTS_PER_RECORDING),
    (1, 100, 1, COMPONENTS_PER_RECORDING),
    (1, 100, 1, COMPONENTS_PER_RECORDING),
)


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
    urls = [
        str(dataset["s3_object_url_template"]).format(git_annex_key=quote(str(recording["git_annex_key"]), safe="")),
        str(dataset["openneuro_s3_url_template"]).format(source_path=quote(str(recording["source_path"]), safe="/")),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(path.name + ".part")
    for url_index, url in enumerate(urls):
        request = Request(url, headers={"User-Agent": "EEGPrep-ICLabel-evaluation/1"})
        for attempt in range(3):
            print(f"DOWNLOAD {recording['source_path']} <- {url}", flush=True)
            try:
                with urlopen(request, timeout=180) as response, partial_path.open("wb") as handle:
                    while chunk := response.read(_CHUNK_SIZE):
                        handle.write(chunk)
            except HTTPError as error:
                partial_path.unlink(missing_ok=True)
                if url_index == 0 and error.code in (403, 404):
                    print(f"FALLBACK public OpenNeuro S3 after NEMAR HTTP {error.code}", flush=True)
                    break
                raise
            try:
                _verify_recording(partial_path, recording)
            except ValueError:
                partial_path.unlink(missing_ok=True)
                if attempt < 2:
                    print("RETRY after checksum mismatch", flush=True)
                    continue
                if url_index == 0:
                    print("FALLBACK public OpenNeuro S3 after NEMAR checksum mismatch", flush=True)
                    break
                raise
            os.replace(partial_path, path)
            return
    raise RuntimeError(f"Unable to download a checksum-valid object for {recording['source_path']}")


def _feature_archive_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    problem = _checkpoint_problem(path)
    if problem is not None:
        print(f"RECOMPUTE feature checkpoint {path} {problem}", file=sys.stderr, flush=True)
    return problem is None


def _checkpoint_problem(path: Path) -> str | None:
    """Return why a feature checkpoint cannot be reused, or None when it can."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = [np.asarray(archive[name], dtype=np.float32) for name in ("topo", "psd", "autocorr")]
    # Zero-byte checkpoints raise EOFError and truncated ones zipfile.BadZipFile.
    except (EOFError, KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
        return f"is unreadable ({exc!r})"
    shapes = tuple(array.shape for array in arrays)
    if shapes != _FEATURE_SHAPES:
        return f"has shapes {shapes}; expected {_FEATURE_SHAPES}"
    if not all(np.isfinite(array).all() for array in arrays):
        return "contains non-finite values"
    return None


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
    if tuple(feature.shape for feature in features) != _FEATURE_SHAPES:
        raise ValueError(
            f"Unexpected ICLabel feature shapes for {raw_path}: {tuple(feature.shape for feature in features)}"
        )
    return features


def _recording_features(
    manifest: Mapping[str, object],
    recording: Mapping[str, object],
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_path = data_dir / str(recording["source_path"])
    _download_recording(manifest, recording, raw_path)
    checkpoint = data_dir / "features" / f"{recording['subject']}.npz"
    if _feature_archive_is_valid(checkpoint):
        print(f"CHECKPOINT {recording['subject']}", flush=True)
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


def build_evaluation_archive(
    manifest: Mapping[str, object],
    data_dir: Path,
    output_path: Path,
    limit: int | None = None,
) -> None:
    evaluation = manifest["evaluation"]
    recordings = evaluation["recordings"]
    if limit is not None:
        recordings = recordings[:limit]
    extracted = []
    for index, recording in enumerate(recordings, start=1):
        print(f"RECORDING {index}/{len(recordings)} {recording['subject']}", flush=True)
        extracted.append(_recording_features(manifest, recording, data_dir))
    if limit is None and len(extracted) != int(evaluation["component_count"]) // COMPONENTS_PER_RECORDING:
        raise ValueError("Evaluation archive did not process all manifest recordings")
    features = tuple(np.concatenate([recording[index] for recording in extracted], axis=3) for index in range(3))
    expected_count = len(extracted) * COMPONENTS_PER_RECORDING
    if any(feature.shape[3] != expected_count for feature in features):
        raise ValueError("Evaluation archive has the wrong component count")
    _write_archive(output_path, features)
    print(f"ARCHIVE {output_path} components={expected_count}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("evaluation_manifest.json"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    manifest = load_frozen_manifest(args.manifest)
    build_evaluation_archive(manifest, args.data_dir, args.output_dir / DEFAULT_EVALUATION_FEATURES.name, args.limit)


if __name__ == "__main__":
    main()
