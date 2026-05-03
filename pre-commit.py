#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyyaml",
#     "tomli; python_version < '3.11'",
# ]
# ///
#
# Common usage:
#   ./pre-commit.py
#       Check staged files before committing.
#   ./pre-commit.py --fix
#       Check staged files and apply safe whitespace/notebook cleanup fixes.
#   ./pre-commit.py --changed-from origin/develop
#       Check files changed by a PR branch against develop.
#   ./pre-commit.py --all-files
#       Check all tracked files. This is useful after repository-wide cleanups.
#   ./pre-commit.py path/to/file.py path/to/notebook.ipynb
#       Check explicit files.
#
# Checks currently run:
#   - Python syntax parsing for *.py files.
#   - JSON, TOML, YAML syntax parsing when dependencies are available.
#   - Jupyter notebooks have no code-cell outputs or execution counts.
#   - Text files contain no merge conflict markers.
#   - Text files contain no trailing whitespace.
#   - Text files end with a newline.
#   - Non-binary files are no larger than 5 MB.
#
"""Run lightweight pre-commit checks for EEGPrep.

The default mode checks staged files, which makes this suitable as a local
pre-commit command. PR jobs can use ``--changed-from origin/develop`` to check
only files touched by a branch without turning existing baseline issues into
new failures.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import io
import json
import pathlib
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only used on Python 3.10
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

try:
    import yaml
except ModuleNotFoundError:
    yaml = None  # type: ignore[assignment]


ROOT_DIR = pathlib.Path(__file__).resolve().parent
MAX_FILE_SIZE = 5 * 1024 * 1024

EXCLUDE_PATTERNS = [
    ".git/**",
    "**/__pycache__/**",
    "**/*.pyc",
    ".pytest_cache/**",
    ".ruff_cache/**",
    ".mypy_cache/**",
    "build/**",
    "dist/**",
    "*.egg-info/**",
    "htmlcov/**",
    "docs/_build/**",
    "docs/source/_build/**",
    "docs/source/auto_examples/images/**",
    "src/eegprep/eeglab/**",
    "src/eegprep/bin/**",
]

BINARY_PATTERNS = [
    "**/*.3DD",
    "**/*.bdf",
    "**/*.bin",
    "**/*.cnt",
    "**/*.dat",
    "**/*.edf",
    "**/*.exe",
    "**/*.fdt",
    "**/*.flt",
    "**/*.gif",
    "**/*.gz",
    "**/*.ico",
    "**/*.jpg",
    "**/*.mat",
    "**/*.mff/**",
    "**/*.mnc",
    "**/*.mov",
    "**/*.mp4",
    "**/*.npy",
    "**/*.pdf",
    "**/*.png",
    "**/*.set",
    "**/*.xdf",
]

PYTHON_PATTERNS = [
    "**/*.py",
]

PYTHON_EXCLUDE_PATTERNS = [
    # Scratch file kept in the repository but not valid importable Python.
    "scripts/tmp_pipeline_for_ppt.py",
]

CONFIG_PATTERNS = ["**/*.json", "**/*.toml", "**/*.yaml", "**/*.yml"]
NOTEBOOK_PATTERNS = ["**/*.ipynb"]


@dataclass
class CheckResult:
    name: str
    exit_code: int
    output: str = ""


@dataclass
class PrecommitConfig:
    patterns: list[str]
    checks: list[Callable[[list[pathlib.Path], bool], int]]
    exclude_patterns: list[str] = field(default_factory=list)


_check_results: list[CheckResult] = []


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True, check=check)


def relpath(path: pathlib.Path) -> str:
    return path.resolve().relative_to(ROOT_DIR).as_posix()


def echo(message: str = "") -> None:
    print(message, flush=True)


def record(name: str, exit_code: int, output: str = "") -> int:
    status = "ok" if exit_code == 0 else "FAIL"
    echo(f"  {name:.<40s} {status}")
    _check_results.append(CheckResult(name=name, exit_code=exit_code, output=output.rstrip()))
    return exit_code


def matches_pattern(file_path: pathlib.Path, patterns: list[str]) -> bool:
    relative_path = relpath(file_path)
    for pattern in patterns:
        if fnmatch.fnmatch(relative_path, pattern):
            return True
        if pattern.startswith("**/") and fnmatch.fnmatch(relative_path, pattern[3:]):
            return True
    return False


def is_excluded(file_path: pathlib.Path) -> bool:
    return matches_pattern(file_path, EXCLUDE_PATTERNS)


def is_binary_path(file_path: pathlib.Path) -> bool:
    return matches_pattern(file_path, BINARY_PATTERNS)


def git_files(args: list[str]) -> list[pathlib.Path]:
    result = run_cmd(args)
    return [ROOT_DIR / line for line in result.stdout.splitlines() if line]


def get_files(all_files: bool, changed_from: str | None, file_args: list[str]) -> list[pathlib.Path]:
    if file_args:
        files = []
        for raw_path in file_args:
            path = pathlib.Path(raw_path)
            if not path.is_absolute():
                path = ROOT_DIR / path
            if not path.exists():
                echo(f"Warning: skipping non-existent file: {raw_path}")
                continue
            files.append(path)
    elif all_files:
        files = git_files(["git", "ls-files"])
    elif changed_from:
        files = git_files(["git", "diff", "--name-only", "--diff-filter=ACM", f"{changed_from}...HEAD"])
    else:
        files = git_files(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"])

    seen: set[pathlib.Path] = set()
    filtered = []
    for file_path in files:
        resolved = file_path.resolve()
        if resolved in seen or not resolved.exists() or not resolved.is_file():
            continue
        seen.add(resolved)
        filtered.append(resolved)
    return filtered


def get_matching_files(
    patterns: list[str], files: list[pathlib.Path], exclude_patterns: list[str] | None = None
) -> list[pathlib.Path]:
    exclude_patterns = exclude_patterns or []
    matched = []
    for file_path in files:
        if is_excluded(file_path):
            continue
        if matches_pattern(file_path, exclude_patterns):
            continue
        if matches_pattern(file_path, patterns):
            matched.append(file_path)
    return matched


def is_probably_text(file_path: pathlib.Path) -> bool:
    if is_binary_path(file_path):
        return False
    try:
        chunk = file_path.read_bytes()[:4096]
    except OSError:
        return False
    return b"\0" not in chunk


def check_large_files(files: list[pathlib.Path], fix: bool) -> int:
    del fix
    large_files = []
    for file_path in files:
        if is_binary_path(file_path):
            continue
        size = file_path.stat().st_size
        if size > MAX_FILE_SIZE:
            large_files.append((file_path, size))

    if not large_files:
        return record("Large files", 0)

    buf = io.StringIO()
    buf.write(f"Files must be {MAX_FILE_SIZE / 1024 / 1024:.1f} MB or smaller:\n")
    for path, size in large_files:
        buf.write(f"  - {relpath(path)} ({size / 1024 / 1024:.1f} MB)\n")
    return record("Large files", 1, buf.getvalue())


def check_python_ast(files: list[pathlib.Path], fix: bool) -> int:
    del fix
    invalid_files = []
    for file_path in files:
        try:
            ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        except SyntaxError as error:
            invalid_files.append((file_path, error))
        except UnicodeDecodeError as error:
            invalid_files.append((file_path, error))

    if not invalid_files:
        return record("Python syntax", 0)

    buf = io.StringIO()
    for path, error in invalid_files:
        buf.write(f"  - {relpath(path)}: {error}\n")
    return record("Python syntax", 1, buf.getvalue())


def check_merge_conflicts(files: list[pathlib.Path], fix: bool) -> int:
    del fix
    files_with_conflicts = []
    for file_path in files:
        if not is_probably_text(file_path):
            continue
        try:
            content = file_path.read_bytes()
        except OSError:
            continue
        has_marker = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith(b"<<<<<<<") or stripped.startswith(b">>>>>>>"):
                has_marker = True
                break
        if has_marker:
            files_with_conflicts.append(file_path)

    if not files_with_conflicts:
        return record("Merge conflicts", 0)

    buf = io.StringIO()
    for path in files_with_conflicts:
        buf.write(f"  - {relpath(path)}\n")
    return record("Merge conflicts", 1, buf.getvalue())


def check_config_syntax(files: list[pathlib.Path], fix: bool) -> int:
    del fix
    errors = []
    warned_missing_toml = False
    warned_missing_yaml = False
    for file_path in files:
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".json":
                json.loads(file_path.read_text(encoding="utf-8"))
            elif suffix == ".toml":
                if tomllib is None:
                    if not warned_missing_toml:
                        echo("  Warning: tomli is not installed; skipping TOML syntax checks")
                        warned_missing_toml = True
                    continue
                with file_path.open("rb") as handle:
                    tomllib.load(handle)
            elif suffix in {".yaml", ".yml"}:
                if yaml is None:
                    if not warned_missing_yaml:
                        echo("  Warning: PyYAML is not installed; skipping YAML syntax checks")
                        warned_missing_yaml = True
                    continue
                yaml.safe_load(file_path.read_text(encoding="utf-8"))
        except Exception as error:
            errors.append((file_path, error))

    if not errors:
        return record("Config syntax", 0)

    buf = io.StringIO()
    for path, error in errors:
        buf.write(f"  - {relpath(path)}: {error}\n")
    return record("Config syntax", 1, buf.getvalue())


def check_trailing_whitespace(files: list[pathlib.Path], fix: bool) -> int:
    files_with_whitespace = []
    for file_path in files:
        if not is_probably_text(file_path):
            continue
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
        except UnicodeDecodeError:
            continue

        has_trailing = any(line.rstrip("\n\r").endswith((" ", "\t")) for line in lines)
        if not has_trailing:
            continue

        files_with_whitespace.append(file_path)
        if fix:
            file_ended_with_newline = lines[-1].endswith(("\n", "\r")) if lines else True
            cleaned_lines = []
            for index, line in enumerate(lines):
                is_last_line = index == len(lines) - 1
                cleaned = line.rstrip()
                if is_last_line and not file_ended_with_newline:
                    cleaned_lines.append(cleaned)
                else:
                    cleaned_lines.append(cleaned + "\n")
            file_path.write_text("".join(cleaned_lines), encoding="utf-8")

    if not files_with_whitespace:
        return record("Trailing whitespace", 0)

    buf = io.StringIO()
    for path in files_with_whitespace:
        buf.write(f"  - {relpath(path)}\n")
    return record("Trailing whitespace", 1, buf.getvalue())


def check_eof_newline(files: list[pathlib.Path], fix: bool) -> int:
    files_missing_newline = []
    for file_path in files:
        if not is_probably_text(file_path) or file_path.stat().st_size == 0:
            continue
        try:
            content = file_path.read_bytes()
        except OSError:
            continue
        if content.endswith(b"\n"):
            continue

        files_missing_newline.append(file_path)
        if fix:
            with file_path.open("ab") as handle:
                handle.write(b"\n")

    if not files_missing_newline:
        return record("End-of-file newline", 0)

    buf = io.StringIO()
    for path in files_missing_newline:
        buf.write(f"  - {relpath(path)}\n")
    return record("End-of-file newline", 1, buf.getvalue())


def check_notebooks(files: list[pathlib.Path], fix: bool) -> int:
    dirty_notebooks = []
    invalid_notebooks = []

    for notebook_path in files:
        try:
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        except Exception as error:
            invalid_notebooks.append((notebook_path, error))
            continue

        needs_cleaning = False
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("outputs") or cell.get("execution_count") is not None:
                needs_cleaning = True
                if fix:
                    cell["outputs"] = []
                    cell["execution_count"] = None

        if needs_cleaning:
            dirty_notebooks.append(notebook_path)
            if fix:
                notebook_path.write_text(
                    json.dumps(notebook, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

    if not dirty_notebooks and not invalid_notebooks:
        return record("Jupyter notebooks", 0)

    buf = io.StringIO()
    for path, error in invalid_notebooks:
        buf.write(f"  - {relpath(path)}: {error}\n")
    for path in dirty_notebooks:
        buf.write(f"  - {relpath(path)} has outputs or execution counts\n")
    return record("Jupyter notebooks", 1, buf.getvalue())


PRECOMMIT_CONFIGS = [
    PrecommitConfig(
        patterns=PYTHON_PATTERNS,
        exclude_patterns=PYTHON_EXCLUDE_PATTERNS,
        checks=[check_python_ast],
    ),
    PrecommitConfig(
        patterns=CONFIG_PATTERNS,
        checks=[check_config_syntax],
    ),
    PrecommitConfig(
        patterns=NOTEBOOK_PATTERNS,
        checks=[check_notebooks],
    ),
    PrecommitConfig(
        patterns=["**/*"],
        checks=[
            check_large_files,
            check_merge_conflicts,
            check_trailing_whitespace,
            check_eof_newline,
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="Specific files to check")
    parser.add_argument("--fix", action="store_true", help="Automatically fix safe file hygiene issues")
    parser.add_argument("--all-files", action="store_true", help="Check all tracked files")
    parser.add_argument(
        "--changed-from",
        metavar="REF",
        help="Check files changed against a base ref, e.g. origin/develop",
    )
    args = parser.parse_args()

    selected_modes = sum(bool(value) for value in [args.all_files, args.changed_from, args.files])
    if selected_modes > 1:
        parser.error("choose only one of --all-files, --changed-from, or explicit files")
    return args


def main() -> int:
    args = parse_args()
    try:
        files = get_files(args.all_files, args.changed_from, args.files)
    except subprocess.CalledProcessError as error:
        echo((error.stdout or "") + (error.stderr or ""))
        return 1

    if not files:
        echo("No files selected.")
        echo("=" * 60)
        echo("OK")
        echo("=" * 60)
        return 0

    echo(f"Checking {len(files)} file(s)")
    exit_codes = []
    for config in PRECOMMIT_CONFIGS:
        matched_files = get_matching_files(config.patterns, files, config.exclude_patterns)
        if not matched_files:
            continue
        for check in config.checks:
            try:
                exit_codes.append(check(matched_files, args.fix))
            except Exception as error:  # pragma: no cover - defensive summary path
                exit_codes.append(record(check.__name__, 1, str(error)))

    failures = [result for result in _check_results if result.exit_code != 0 and result.output]
    if failures:
        echo()
        echo("=" * 60)
        echo("Failure details:")
        echo()
        for result in failures:
            echo(f"--- {result.name} ---")
            echo(result.output)
            echo()

    echo("=" * 60)
    if any(exit_codes):
        echo("FAILED")
        echo("=" * 60)
        return 1

    echo("OK")
    echo("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
