"""Validate the EEGLAB core parity matrix."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MATRIX_RELATIVE_PATH = Path("docs/parity/eeglab_core_parity_matrix.json")

IN_SCOPE_FUNCTION_FOLDERS = (
    "adminfunc",
    "guifunc",
    "miscfunc",
    "popfunc",
    "sigprocfunc",
    "statistics",
    "studyfunc",
    "timefreqfunc",
)
EXPLICIT_IN_SCOPE_EEGLAB_FILES = ("plugins/clean_rawdata/clean_asr.m",)

VALID_GAP_CATEGORIES = (
    "1_long_tail_helper_coverage",
    "2_missing_or_legacy_pop_entry_points",
    "3_unsupported_options",
    "4_study_group_depth",
    "5_statistics_package",
    "6_time_frequency_internals",
    "7_file_format_channel_location",
)
VALID_STATUSES = (
    "implemented",
    "partial",
    "port",
    "consolidated",
    "stale_skip",
    "matlab_runtime_skip",
    "external_dependency_skip",
)
VALID_PHASES = ("none", "phase_1", "phase_2", "phase_3", "phase_4", "phase_5", "phase_6", "phase_7")
ACTIONABLE_STATUSES = {"partial", "port"}
EQUIVALENT_REQUIRED_STATUSES = {"implemented", "partial", "consolidated"}
FOLLOW_UP_ISSUE_RE = re.compile(r"#[1-9][0-9]*")
REQUIRED_ROW_FIELDS = (
    "eeglab_path",
    "eeglab_name",
    "eegprep_equivalent",
    "gap_category",
    "status",
    "rationale",
    "responsible_phase",
    "user_facing_surface",
    "test_notes",
)
STALE_POLICY_FIELDS = (
    "menu_reachable",
    "documented_user_api",
    "called_by_in_scope_workflow",
    "required_by_parity_tests",
    "needed_as_phase_helper",
    "likely_user_alias",
)


@dataclass(frozen=True)
class MatrixValidationError:
    location: str
    message: str

    def as_text(self) -> str:
        return f"{self.location}: {self.message}"


@dataclass(frozen=True)
class MatrixValidationReport:
    errors: tuple[MatrixValidationError, ...]
    row_count: int
    expected_eeglab_count: int
    status_counts: dict[str, int]
    category_counts: dict[str, int]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [error.as_text() for error in self.errors],
            "row_count": self.row_count,
            "expected_eeglab_count": self.expected_eeglab_count,
            "status_counts": self.status_counts,
            "category_counts": self.category_counts,
        }


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discover_in_scope_eeglab_paths(repo_root: Path) -> set[str]:
    paths: set[str] = set()
    function_root = repo_root / "src/eegprep/eeglab/functions"
    if function_root.is_dir():
        for folder in IN_SCOPE_FUNCTION_FOLDERS:
            for path in sorted((function_root / folder).glob("*.m")):
                paths.add(f"functions/{folder}/{path.name}")
    for path in EXPLICIT_IN_SCOPE_EEGLAB_FILES:
        if (repo_root / "src/eegprep/eeglab" / path).is_file():
            paths.add(path)
    return paths


def load_matrix(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("Parity matrix root must be a JSON object")
    return payload


def validate_matrix_file(matrix_path: Path | None = None, repo_root: Path | None = None) -> MatrixValidationReport:
    root = repo_root or default_repo_root()
    path = matrix_path or root / MATRIX_RELATIVE_PATH
    return validate_matrix_payload(load_matrix(path), root)


def validate_matrix_payload(payload: dict[str, Any], repo_root: Path) -> MatrixValidationReport:
    errors: list[MatrixValidationError] = []
    rows = payload.get("rows")
    if not isinstance(rows, list):
        errors.append(MatrixValidationError("rows", "must be a list"))
        rows = []

    _validate_metadata(payload.get("metadata"), errors)

    expected_paths = discover_in_scope_eeglab_paths(repo_root)
    seen_paths: set[str] = set()
    status_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()

    for index, row in enumerate(rows):
        location = f"rows[{index}]"
        if not isinstance(row, dict):
            errors.append(MatrixValidationError(location, "must be an object"))
            continue
        _validate_row(row, location, errors)
        eeglab_path = row.get("eeglab_path")
        if isinstance(eeglab_path, str):
            if eeglab_path in seen_paths:
                errors.append(MatrixValidationError(location, f"duplicates eeglab_path {eeglab_path!r}"))
            seen_paths.add(eeglab_path)
        status = row.get("status")
        if isinstance(status, str):
            status_counts[status] += 1
        category = row.get("gap_category")
        if isinstance(category, str):
            category_counts[category] += 1

    if not expected_paths:
        errors.append(
            MatrixValidationError(
                "coverage",
                "EEGLAB reference tree is missing or empty; initialize src/eegprep/eeglab before validating coverage",
            )
        )
    else:
        for missing_path in sorted(expected_paths - seen_paths):
            errors.append(MatrixValidationError("coverage", f"missing classification for {missing_path}"))
        for extra_path in sorted(seen_paths - expected_paths):
            errors.append(MatrixValidationError("coverage", f"classifies out-of-scope EEGLAB path {extra_path}"))

    return MatrixValidationReport(
        errors=tuple(errors),
        row_count=len(rows),
        expected_eeglab_count=len(expected_paths),
        status_counts=dict(sorted(status_counts.items())),
        category_counts=dict(sorted(category_counts.items())),
    )


def _validate_metadata(metadata: Any, errors: list[MatrixValidationError]) -> None:
    if not isinstance(metadata, dict):
        errors.append(MatrixValidationError("metadata", "must be an object"))
        return
    if metadata.get("schema_version") != 1:
        errors.append(MatrixValidationError("metadata.schema_version", "must be 1"))
    if tuple(metadata.get("status_taxonomy", ())) != VALID_STATUSES:
        errors.append(MatrixValidationError("metadata.status_taxonomy", "must match the validator status taxonomy"))
    policy = metadata.get("stale_skip_policy")
    if not isinstance(policy, dict):
        errors.append(MatrixValidationError("metadata.stale_skip_policy", "must be an object"))
        return
    for field in STALE_POLICY_FIELDS:
        value = policy.get(field)
        if not isinstance(value, str) or not value:
            errors.append(MatrixValidationError(f"metadata.stale_skip_policy.{field}", "must define the policy"))


def _validate_row(row: dict[str, Any], location: str, errors: list[MatrixValidationError]) -> None:
    for field in REQUIRED_ROW_FIELDS:
        if field not in row:
            errors.append(MatrixValidationError(location, f"missing required field {field!r}"))

    eeglab_path = row.get("eeglab_path")
    eeglab_name = row.get("eeglab_name")
    if not isinstance(eeglab_path, str) or not eeglab_path.endswith(".m"):
        errors.append(MatrixValidationError(f"{location}.eeglab_path", "must be an EEGLAB .m path string"))
    elif not isinstance(eeglab_name, str) or Path(eeglab_path).stem != eeglab_name:
        errors.append(MatrixValidationError(f"{location}.eeglab_name", "must match the EEGLAB file stem"))

    equivalent = row.get("eegprep_equivalent")
    if equivalent is not None and not isinstance(equivalent, str):
        errors.append(MatrixValidationError(f"{location}.eegprep_equivalent", "must be a string or null"))

    category = row.get("gap_category")
    if category not in VALID_GAP_CATEGORIES:
        errors.append(MatrixValidationError(f"{location}.gap_category", "must be a valid gap category"))

    status = row.get("status")
    if status not in VALID_STATUSES:
        errors.append(MatrixValidationError(f"{location}.status", "must be a valid status"))
    elif status in EQUIVALENT_REQUIRED_STATUSES and not equivalent:
        errors.append(MatrixValidationError(f"{location}.eegprep_equivalent", f"is required for status {status!r}"))

    for field in ("rationale", "responsible_phase", "test_notes"):
        value = row.get(field)
        if not isinstance(value, str) or not value:
            errors.append(MatrixValidationError(f"{location}.{field}", "must be a non-empty string"))

    phase = row.get("responsible_phase")
    if phase not in VALID_PHASES:
        errors.append(MatrixValidationError(f"{location}.responsible_phase", "must be a valid phase id"))
    if status in ACTIONABLE_STATUSES and phase == "none":
        errors.append(MatrixValidationError(f"{location}.responsible_phase", f"is required for status {status!r}"))
    if status in ACTIONABLE_STATUSES and not FOLLOW_UP_ISSUE_RE.search(
        f"{row.get('rationale') or ''} {row.get('test_notes') or ''}"
    ):
        errors.append(
            MatrixValidationError(
                location,
                f"status {status!r} must cite a concrete follow-up issue in rationale or test_notes",
            )
        )

    surfaces = row.get("user_facing_surface")
    if not isinstance(surfaces, list) or not all(isinstance(surface, str) and surface for surface in surfaces):
        errors.append(MatrixValidationError(f"{location}.user_facing_surface", "must be a list of strings"))

    stale_policy = row.get("stale_policy")
    if status == "stale_skip":
        if not isinstance(stale_policy, dict):
            errors.append(MatrixValidationError(f"{location}.stale_policy", "is required for stale_skip rows"))
            return
        for field in STALE_POLICY_FIELDS:
            if stale_policy.get(field) is not False:
                errors.append(MatrixValidationError(f"{location}.stale_policy.{field}", "must be false"))
    elif "stale_policy" in row:
        errors.append(MatrixValidationError(f"{location}.stale_policy", "is only allowed for stale_skip rows"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "matrix",
        nargs="?",
        type=Path,
        default=None,
        help=f"matrix path, default {MATRIX_RELATIVE_PATH}",
    )
    parser.add_argument("--json", action="store_true", help="emit a JSON validation report")
    args = parser.parse_args(argv)

    report = validate_matrix_file(args.matrix)
    if args.json:
        print(json.dumps(report.to_jsonable(), indent=2, sort_keys=True))
    elif report.ok:
        print(
            f"Parity matrix OK: {report.row_count} rows cover {report.expected_eeglab_count} in-scope EEGLAB functions."
        )
    else:
        for error in report.errors:
            print(error.as_text())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
