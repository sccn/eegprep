"""Shared CLI result, error, and manifest helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np

import eegprep


COMMAND_RESULT_SCHEMA_VERSION = "eegprep.cli.result.v1"
MANIFEST_SCHEMA_VERSION = "eegprep.manifest.v2"


class EEGPrepCLIError(Exception):
    """Structured error raised by CLI commands."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str | Path | None = None,
        suggestion: str | None = None,
        details: Any | None = None,
        exit_code: int = 1,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.path = None if path is None else str(path)
        self.suggestion = suggestion
        self.details = details
        self.exit_code = exit_code

    def to_response(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "error",
            "schema_version": "eegprep.error.v1",
            "code": self.code,
            "message": self.message,
        }
        if self.path is not None:
            payload["path"] = self.path
        if self.suggestion is not None:
            payload["suggestion"] = self.suggestion
        if self.details is not None:
            payload["details"] = self.details
        return payload


@dataclass(frozen=True)
class RuntimeStamp:
    """Start/finish timestamps for manifests."""

    started_at: str
    finished_at: str


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def runtime_stamp(started_at: str) -> RuntimeStamp:
    return RuntimeStamp(started_at=started_at, finished_at=now_iso())


def ok(schema_version: str, **payload: Any) -> dict[str, Any]:
    return {"status": "ok", "schema_version": schema_version, **payload}


def warning(schema_version: str, **payload: Any) -> dict[str, Any]:
    return {"status": "warning", "schema_version": schema_version, **payload}


def command_ok(command: str, **payload: Any) -> dict[str, Any]:
    return {
        "status": "ok",
        "schema_version": COMMAND_RESULT_SCHEMA_VERSION,
        "command": command,
        **payload,
    }


def command_error(command: str, error: EEGPrepCLIError) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "code": error.code,
        "message": error.message,
    }
    if error.path is not None:
        payload["path"] = error.path
    if error.suggestion is not None:
        payload["suggestion"] = error.suggestion
    if error.details is not None:
        payload["details"] = error.details
    return {
        "status": "error",
        "schema_version": COMMAND_RESULT_SCHEMA_VERSION,
        "command": command,
        "code": error.code,
        "message": error.message,
        "exit_code": error.exit_code,
        "error": payload,
    }


def emit_command_result(result: dict[str, Any], *, json_output: bool = True) -> int:
    if json_output:
        print(json.dumps(json_safe(result), sort_keys=True))
    else:
        print(result.get("status", "ok"))
        if result.get("status") == "error":
            print(result.get("error", {}).get("message", ""), file=sys.stderr)
    if result.get("status") == "ok":
        return 0
    return int(result.get("exit_code", 1) or 1)


def print_result(result: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(json_safe(result), sort_keys=True, separators=(",", ":")))
    else:
        print(humanize_result(result))


def humanize_result(result: dict[str, Any]) -> str:
    if result.get("status") == "error":
        message = f"{result.get('code', 'ERROR')}: {result.get('message', '')}"
        if result.get("suggestion"):
            message += f"\nSuggestion: {result['suggestion']}"
        return message
    lines = [f"status: {result.get('status', 'ok')}"]
    for key, value in result.items():
        if key in {"status", "schema_version"}:
            continue
        if isinstance(value, (dict, list)):
            lines.append(f"{key}: {json.dumps(json_safe(value), sort_keys=True)}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def existing_input(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise EEGPrepCLIError(
            "INPUT_FILE_NOT_FOUND",
            f"Input file not found: {resolved}",
            path=resolved,
            suggestion="Check the path or run eegprep inspect dataset <path> on an existing EEGLAB .set file.",
        )
    if not resolved.is_file():
        raise EEGPrepCLIError("UNSUPPORTED_FORMAT", f"Input path is not a file: {resolved}", path=resolved)
    return resolved


def output_path(path: str | Path | None, *, input_path: Path | None = None, overwrite: bool = False) -> Path:
    if path is None:
        if input_path is None or not overwrite:
            raise EEGPrepCLIError(
                "OUTPUT_REQUIRED",
                "An output path is required unless --overwrite is explicitly used.",
                suggestion="Pass --output <path> for non-destructive CLI workflows.",
            )
        return input_path
    resolved = Path(path).expanduser()
    if resolved.exists() and not overwrite:
        raise EEGPrepCLIError(
            "OUTPUT_EXISTS",
            f"Output already exists: {resolved}",
            path=resolved,
            suggestion="Use --overwrite or choose a new output path.",
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    return sha256_file(path)


def software_info() -> dict[str, Any]:
    return {
        "eegprep_version": eegprep.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def build_manifest(
    *,
    command: str,
    input_files: list[Path | dict[str, Any]],
    output_files: list[dict[str, Any]],
    parameters: dict[str, Any],
    history: str = "",
    started_at: str,
    finished_at: str | None = None,
    deterministic: bool | None = None,
    warnings: list[Any] | None = None,
) -> dict[str, Any]:
    stamp = runtime_stamp(started_at) if finished_at is None else RuntimeStamp(started_at, finished_at)
    input_records = [_input_file_record(path) for path in input_files]
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "command": command,
        "input_files": _make_manifest_paths_absolute(input_records),
        "output_files": _make_manifest_paths_absolute(output_files),
        "parameters": json_safe(parameters),
        "history": history,
        "software": software_info(),
        "runtime": {"started_at": stamp.started_at, "finished_at": stamp.finished_at},
        "warnings": warnings or [],
    }
    if deterministic is not None:
        manifest["deterministic"] = deterministic
    return manifest


def _make_manifest_relative(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    """Return the on-disk representation of a v2 manifest."""
    prepared = copy.deepcopy(json_safe(manifest))
    if prepared.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        return prepared

    base_dir = manifest_path.resolve().parent
    for item in _manifest_file_records(prepared):
        value = item.get("path")
        if not isinstance(value, str) or not value:
            continue
        native_path = Path(value).expanduser()
        if not native_path.is_absolute():
            continue
        try:
            item["path"] = Path(os.path.relpath(native_path, start=base_dir)).as_posix()
        except ValueError:
            # Windows cannot express a path on another drive as a relative path.
            continue
    return prepared


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Read a manifest, resolving v2 artifact paths against its directory."""
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        return manifest

    for item in _manifest_file_records(manifest):
        value = item.get("path")
        if not isinstance(value, str) or not value or _is_absolute_on_any_platform(value):
            continue
        relative_path = Path(*PurePosixPath(value).parts)
        item["path"] = str((resolved.parent / relative_path).resolve())
    return manifest


def write_manifest(path: str | Path | None, manifest: dict[str, Any]) -> Path | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    rel_manifest = _make_manifest_relative(manifest, resolved)
    resolved.write_text(json.dumps(json_safe(rel_manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def write_json_file(
    path: str | Path,
    payload: dict[str, Any],
    *,
    overwrite: bool = False,
    output_type: str = "json",
) -> dict[str, Any]:
    target = Path(path)
    if target.exists() and not overwrite:
        raise EEGPrepCLIError(
            "OUTPUT_EXISTS",
            f"Output already exists: {target}",
            path=target,
            suggestion="Use --overwrite or choose a different output path.",
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(target), "type": output_type, "sha256": sha256_file(target)}


def write_manifest_file(path: str | Path, manifest: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    target = Path(path)
    rel_manifest = _make_manifest_relative(manifest, target)
    return write_json_file(target, rel_manifest, overwrite=overwrite, output_type="manifest")


def _make_manifest_paths_absolute(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize runtime artifact paths before they acquire manifest-relative semantics."""
    prepared = copy.deepcopy(json_safe(records))
    for item in prepared:
        value = item.get("path")
        if not isinstance(value, str) or not value or _is_foreign_absolute_path(value):
            continue
        item["path"] = str(Path(value).expanduser().resolve())
    return prepared


def _manifest_file_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in ("input_files", "output_files"):
        value = manifest.get(key)
        if isinstance(value, list):
            records.extend(item for item in value if isinstance(item, dict))
    return records


def _is_absolute_on_any_platform(value: str) -> bool:
    return Path(value).is_absolute() or PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _is_foreign_absolute_path(value: str) -> bool:
    return not Path(value).is_absolute() and (
        PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()
    )


def _input_file_record(path: Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path, dict):
        return json_safe(path)
    return {"path": str(path), "sha256": sha256_file(path)}
