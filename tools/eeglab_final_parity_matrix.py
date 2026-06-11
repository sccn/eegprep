"""Validate the final EEGPrep standalone parity matrix."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MATRIX_RELATIVE_PATH = Path("docs/parity/eeglab_final_parity_matrix.json")
REFERENCE_PATHS_RELATIVE_PATH = Path("docs/parity/eeglab_final_reference_paths.txt")

PLUGIN_ROOTS = ("clean_rawdata", "firfilt", "ICLabel", "dipfit")
OBJECT_STORAGE_FOLDERS = ("@eegobj", "@memmapdata", "@mmo")
TUTORIAL_EXTENSIONS = {".m", ".mlx"}
EXCLUDED_REFERENCE_PREFIXES = (
    "plugins/clean_rawdata/manopt/manopt/",
    "plugins/clean_rawdata/manopt/examples/",
    "plugins/clean_rawdata/manopt/tests/",
    "plugins/ICLabel/matconvnet/",
    "plugins/ICLabel/tests/",
)

VALID_AREAS = (
    "bundled_plugin_clean_rawdata",
    "bundled_plugin_firfilt",
    "bundled_plugin_iclabel_viewprops",
    "bundled_plugin_dipfit",
    "object_storage",
    "optional_toolbox_workflow",
    "documentation",
)
VALID_STATUSES = (
    "implemented",
    "partial",
    "port",
    "consolidated",
    "stale_skip",
    "matlab_runtime_skip",
    "optional_dependency",
    "external_plugin",
    "docs_gap",
)
VALID_PHASES = ("none", "phase_2", "phase_3", "phase_4", "phase_5", "phase_6", "phase_7", "phase_8")
SKIP_STATUSES = {"stale_skip", "matlab_runtime_skip", "external_plugin"}
EQUIVALENT_REQUIRED_STATUSES = {"implemented", "partial", "consolidated"}
OPTIONAL_DEPENDENCY_FIELDS = ("name", "dependency_type", "fallback_behavior", "user_message", "phase_contract")
OPTIONAL_DEPENDENCY_POLICY_FIELDS = (
    "standalone_first",
    "optional_dependency",
    "external_plugin",
    "user_facing_failure",
)
REQUIRED_DOC_SECTION_IDS = (
    "installation",
    "concepts",
    "gui_tutorials",
    "console_and_history",
    "preprocessing",
    "ica_and_rejection",
    "study_and_statistics",
    "plugins_and_extensions",
    "api_reference",
    "eeglab_migration",
    "developer_parity",
)
REQUIRED_ROW_FIELDS = (
    "row_id",
    "title",
    "area",
    "source_paths",
    "eegprep_equivalent",
    "status",
    "responsible_phase",
    "phase_issue",
    "rationale",
    "user_facing_surface",
    "docs_targets",
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
ISSUE_RE = re.compile(r"^#[1-9][0-9]*")


@dataclass(frozen=True)
class FinalMatrixValidationError:
    location: str
    message: str

    def as_text(self) -> str:
        return f"{self.location}: {self.message}"


@dataclass(frozen=True)
class FinalMatrixValidationReport:
    errors: tuple[FinalMatrixValidationError, ...]
    row_count: int
    expected_eeglab_count: int
    status_counts: dict[str, int]
    area_counts: dict[str, int]

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
            "area_counts": self.area_counts,
        }


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discover_final_eeglab_paths(repo_root: Path) -> set[str]:
    """Return EEGLAB reference paths covered by the final parity epic."""

    eeglab_root = repo_root / "src/eegprep/eeglab"
    if not eeglab_root.is_dir():
        return set()

    snapshot_paths = _load_reference_path_snapshot(repo_root)
    live_paths = _discover_live_final_eeglab_paths(eeglab_root)
    if snapshot_paths and not snapshot_paths <= live_paths:
        return snapshot_paths
    return live_paths


def _discover_live_final_eeglab_paths(eeglab_root: Path) -> set[str]:
    """Discover final-epic source paths from an initialized EEGLAB checkout."""

    paths: set[str] = set()
    plugin_root = eeglab_root / "plugins"
    for plugin in PLUGIN_ROOTS:
        for path in sorted((plugin_root / plugin).rglob("*.m")):
            relative = path.relative_to(eeglab_root).as_posix()
            if not _is_excluded_reference_path(relative):
                paths.add(relative)

    functions_root = eeglab_root / "functions"
    for folder in OBJECT_STORAGE_FOLDERS:
        for path in sorted((functions_root / folder).glob("*.m")):
            paths.add(path.relative_to(eeglab_root).as_posix())

    tutorial_root = eeglab_root / "tutorial_scripts"
    for path in sorted(tutorial_root.iterdir() if tutorial_root.is_dir() else ()):
        if path.is_file() and path.suffix in TUTORIAL_EXTENSIONS:
            paths.add(path.relative_to(eeglab_root).as_posix())

    return paths


def _load_reference_path_snapshot(repo_root: Path) -> set[str]:
    path = repo_root / REFERENCE_PATHS_RELATIVE_PATH
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def load_matrix(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("Final parity matrix root must be a JSON object")
    return payload


def validate_matrix_file(matrix_path: Path | None = None, repo_root: Path | None = None) -> FinalMatrixValidationReport:
    root = repo_root or default_repo_root()
    path = matrix_path or root / MATRIX_RELATIVE_PATH
    return validate_matrix_payload(load_matrix(path), root)


def validate_matrix_payload(payload: dict[str, Any], repo_root: Path) -> FinalMatrixValidationReport:
    errors: list[FinalMatrixValidationError] = []
    rows = payload.get("rows")
    if not isinstance(rows, list):
        errors.append(FinalMatrixValidationError("rows", "must be a list"))
        rows = []

    phase_issues = _validate_metadata(payload.get("metadata"), errors)
    expected_paths = discover_final_eeglab_paths(repo_root)
    seen_expected_paths: dict[str, str] = {}
    seen_row_ids: set[str] = set()
    status_counts: Counter[str] = Counter()
    area_counts: Counter[str] = Counter()

    for index, row in enumerate(rows):
        location = f"rows[{index}]"
        if not isinstance(row, dict):
            errors.append(FinalMatrixValidationError(location, "must be an object"))
            continue
        _validate_row(row, location, phase_issues, errors)

        row_id = row.get("row_id")
        if isinstance(row_id, str):
            if row_id in seen_row_ids:
                errors.append(FinalMatrixValidationError(location, f"duplicates row_id {row_id!r}"))
            seen_row_ids.add(row_id)

        for source_path in _source_paths(row):
            if not _source_path_exists(repo_root, source_path):
                errors.append(FinalMatrixValidationError(location, f"source_path does not exist: {source_path}"))
                continue
            if source_path not in expected_paths:
                continue
            previous = seen_expected_paths.get(source_path)
            if previous is not None:
                errors.append(
                    FinalMatrixValidationError(
                        location,
                        f"duplicates expected EEGLAB source_path {source_path!r} already covered by {previous}",
                    )
                )
            else:
                seen_expected_paths[source_path] = str(row_id)

        status = row.get("status")
        if isinstance(status, str):
            status_counts[status] += 1
        area = row.get("area")
        if isinstance(area, str):
            area_counts[area] += 1

    if not expected_paths:
        errors.append(
            FinalMatrixValidationError(
                "coverage",
                "EEGLAB reference tree is missing or empty; initialize src/eegprep/eeglab before validating coverage",
            )
        )
    else:
        for missing_path in sorted(expected_paths - set(seen_expected_paths)):
            errors.append(FinalMatrixValidationError("coverage", f"missing classification for {missing_path}"))

    return FinalMatrixValidationReport(
        errors=tuple(errors),
        row_count=len(rows),
        expected_eeglab_count=len(expected_paths),
        status_counts=dict(sorted(status_counts.items())),
        area_counts=dict(sorted(area_counts.items())),
    )


def _validate_metadata(metadata: Any, errors: list[FinalMatrixValidationError]) -> dict[str, str]:
    if not isinstance(metadata, dict):
        errors.append(FinalMatrixValidationError("metadata", "must be an object"))
        return {}
    if metadata.get("schema_version") != 1:
        errors.append(FinalMatrixValidationError("metadata.schema_version", "must be 1"))
    if tuple(metadata.get("status_taxonomy", ())) != VALID_STATUSES:
        errors.append(
            FinalMatrixValidationError("metadata.status_taxonomy", "must match the validator status taxonomy")
        )

    phase_issues = metadata.get("phase_issues")
    if not isinstance(phase_issues, dict):
        errors.append(FinalMatrixValidationError("metadata.phase_issues", "must be an object"))
        phase_issues = {}
    for phase in VALID_PHASES:
        if phase == "none":
            continue
        value = phase_issues.get(phase)
        if not isinstance(value, str) or not ISSUE_RE.match(value):
            errors.append(FinalMatrixValidationError(f"metadata.phase_issues.{phase}", "must cite the phase issue"))

    policy = metadata.get("optional_dependency_policy")
    if not isinstance(policy, dict):
        errors.append(FinalMatrixValidationError("metadata.optional_dependency_policy", "must be an object"))
    else:
        for field in OPTIONAL_DEPENDENCY_POLICY_FIELDS:
            if not isinstance(policy.get(field), str) or not policy[field]:
                errors.append(
                    FinalMatrixValidationError(f"metadata.optional_dependency_policy.{field}", "must be defined")
                )

    docs_architecture = metadata.get("docs_architecture")
    if not isinstance(docs_architecture, dict):
        errors.append(FinalMatrixValidationError("metadata.docs_architecture", "must be an object"))
    else:
        section_ids = {
            section.get("id") for section in docs_architecture.get("sections", []) if isinstance(section, dict)
        }
        for section_id in REQUIRED_DOC_SECTION_IDS:
            if section_id not in section_ids:
                errors.append(
                    FinalMatrixValidationError(
                        "metadata.docs_architecture.sections",
                        f"must define section {section_id!r}",
                    )
                )

    exclusions = metadata.get("source_exclusion_policy")
    if not isinstance(exclusions, list):
        errors.append(FinalMatrixValidationError("metadata.source_exclusion_policy", "must be a list"))
    elif not exclusions:
        errors.append(FinalMatrixValidationError("metadata.source_exclusion_policy", "must not be empty"))

    return {str(key): str(value) for key, value in phase_issues.items()} if isinstance(phase_issues, dict) else {}


def _validate_row(
    row: dict[str, Any],
    location: str,
    phase_issues: dict[str, str],
    errors: list[FinalMatrixValidationError],
) -> None:
    for field in REQUIRED_ROW_FIELDS:
        if field not in row:
            errors.append(FinalMatrixValidationError(location, f"missing required field {field!r}"))

    row_id = row.get("row_id")
    if not isinstance(row_id, str) or not row_id:
        errors.append(FinalMatrixValidationError(f"{location}.row_id", "must be a non-empty string"))

    title = row.get("title")
    if not isinstance(title, str) or not title:
        errors.append(FinalMatrixValidationError(f"{location}.title", "must be a non-empty string"))

    area = row.get("area")
    if area not in VALID_AREAS:
        errors.append(FinalMatrixValidationError(f"{location}.area", "must be a valid final parity area"))

    paths = row.get("source_paths")
    if not isinstance(paths, list) or not paths or not all(isinstance(path, str) and path for path in paths):
        errors.append(FinalMatrixValidationError(f"{location}.source_paths", "must be a non-empty list of strings"))
    elif len(set(paths)) != len(paths):
        errors.append(FinalMatrixValidationError(f"{location}.source_paths", "must not contain duplicates"))

    status = row.get("status")
    if status not in VALID_STATUSES:
        errors.append(FinalMatrixValidationError(f"{location}.status", "must be a valid status"))
    elif status in EQUIVALENT_REQUIRED_STATUSES and not _has_equivalent(row.get("eegprep_equivalent")):
        errors.append(FinalMatrixValidationError(f"{location}.eegprep_equivalent", f"is required for {status!r}"))

    phase = row.get("responsible_phase")
    if phase not in VALID_PHASES:
        errors.append(FinalMatrixValidationError(f"{location}.responsible_phase", "must be a valid phase id"))
    phase_issue = row.get("phase_issue")
    if status in SKIP_STATUSES:
        if phase != "none":
            errors.append(FinalMatrixValidationError(f"{location}.responsible_phase", "must be none for skip rows"))
        if phase_issue not in (None, ""):
            errors.append(FinalMatrixValidationError(f"{location}.phase_issue", "must be null for skip rows"))
    elif status in VALID_STATUSES:
        if phase == "none":
            errors.append(FinalMatrixValidationError(f"{location}.responsible_phase", f"is required for {status!r}"))
        elif phase_issue != phase_issues.get(str(phase)):
            errors.append(FinalMatrixValidationError(f"{location}.phase_issue", "must match metadata.phase_issues"))

    for field in ("rationale", "test_notes"):
        value = row.get(field)
        if not isinstance(value, str) or not value:
            errors.append(FinalMatrixValidationError(f"{location}.{field}", "must be a non-empty string"))

    for field in ("user_facing_surface", "docs_targets"):
        value = row.get(field)
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            errors.append(FinalMatrixValidationError(f"{location}.{field}", "must be a list of strings"))

    if status == "optional_dependency":
        _validate_optional_dependency(row.get("optional_dependency"), f"{location}.optional_dependency", errors)
    elif "optional_dependency" in row:
        errors.append(
            FinalMatrixValidationError(
                f"{location}.optional_dependency", "is only allowed for optional_dependency rows"
            )
        )

    if status == "docs_gap" and phase != "phase_7":
        errors.append(FinalMatrixValidationError(f"{location}.responsible_phase", "docs_gap rows belong to phase_7"))

    stale_policy = row.get("stale_policy")
    if status == "stale_skip":
        if not isinstance(stale_policy, dict):
            errors.append(FinalMatrixValidationError(f"{location}.stale_policy", "is required for stale_skip rows"))
        else:
            for field in STALE_POLICY_FIELDS:
                if stale_policy.get(field) is not False:
                    errors.append(FinalMatrixValidationError(f"{location}.stale_policy.{field}", "must be false"))
    elif "stale_policy" in row:
        errors.append(FinalMatrixValidationError(f"{location}.stale_policy", "is only allowed for stale_skip rows"))


def _validate_optional_dependency(
    value: Any,
    location: str,
    errors: list[FinalMatrixValidationError],
) -> None:
    if not isinstance(value, dict):
        errors.append(FinalMatrixValidationError(location, "is required for optional_dependency rows"))
        return
    for field in OPTIONAL_DEPENDENCY_FIELDS:
        if not isinstance(value.get(field), str) or not value[field]:
            errors.append(FinalMatrixValidationError(f"{location}.{field}", "must be a non-empty string"))


def _source_paths(row: dict[str, Any]) -> tuple[str, ...]:
    paths = row.get("source_paths")
    if not isinstance(paths, list):
        return ()
    return tuple(path for path in paths if isinstance(path, str))


def _source_path_exists(repo_root: Path, source_path: str) -> bool:
    if source_path.startswith(("functions/", "plugins/", "tutorial_scripts/")):
        source = repo_root / "src/eegprep/eeglab" / source_path
        if source.exists():
            return True
        return source_path in _load_reference_path_snapshot(repo_root) and _source_root_exists(repo_root, source_path)
    if source_path.startswith("docs/"):
        return (repo_root / source_path).exists()
    return False


def _has_equivalent(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, list):
        return bool(value) and all(isinstance(item, str) and item for item in value)
    return False


def _source_root_exists(repo_root: Path, source_path: str) -> bool:
    parts = Path(source_path).parts
    if source_path.startswith("plugins/") and len(parts) >= 2:
        source_root = f"{parts[0]}/{parts[1]}"
        return _eeglab_reference_root_exists(repo_root, source_root)
    if source_path.startswith("tutorial_scripts/"):
        return _eeglab_reference_root_exists(repo_root, "tutorial_scripts")
    if source_path.startswith("functions/") and len(parts) >= 2:
        source_root = f"{parts[0]}/{parts[1]}"
        return _eeglab_reference_root_exists(repo_root, source_root)
    return False


def _eeglab_reference_root_exists(repo_root: Path, relative_root: str) -> bool:
    eeglab_root = repo_root / "src/eegprep/eeglab"
    if (eeglab_root / relative_root).exists():
        return True
    if not eeglab_root.is_dir():
        return False
    try:
        subprocess.run(
            ["git", "-C", str(eeglab_root), "ls-tree", "--exit-code", "HEAD", relative_root],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def _is_excluded_reference_path(relative_path: str) -> bool:
    return any(relative_path.startswith(prefix) for prefix in EXCLUDED_REFERENCE_PREFIXES)


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
            "Final parity matrix OK: "
            f"{report.row_count} rows cover {report.expected_eeglab_count} final-epic EEGLAB paths."
        )
    else:
        for error in report.errors:
            print(error.as_text())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
