#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path


ENGINES = ("codex", "claude", "droid", "copilot")

SAFE_INITIAL = """import numpy as np


def trim_eeg(eeg, start_sample, stop_sample):
    data = np.asarray(eeg["data"])
    start = int(start_sample) - 1
    stop = int(stop_sample)
    trimmed = data[:, start:stop]
    out = dict(eeg)
    out["data"] = trimmed
    out["pnts"] = trimmed.shape[1]
    out["xmin"] = start / float(eeg["srate"])
    out["xmax"] = (stop - 1) / float(eeg["srate"])
    return out
"""

BUGGY_CHANGED = """import numpy as np


def trim_eeg(eeg, start_sample, stop_sample):
    data = np.asarray(eeg["data"])
    trimmed = data[start_sample:stop_sample, :]
    out = dict(eeg)
    out["data"] = trimmed
    out["pnts"] = stop_sample - start_sample
    return out
"""

BENIGN_CHANGED = """import numpy as np


def trim_eeg(eeg, start_sample, stop_sample):
    data = np.asarray(eeg["data"])
    start = int(start_sample) - 1
    stop = int(stop_sample)
    if start < 0 or stop <= start or stop > data.shape[1]:
        raise ValueError("sample range is outside EEG data")
    trimmed = data[:, start:stop]
    out = dict(eeg)
    out["data"] = trimmed
    out["pnts"] = trimmed.shape[1]
    out["xmin"] = start / float(eeg["srate"])
    out["xmax"] = (stop - 1) / float(eeg["srate"])
    return out
"""

BUGGY_PROMPT = (
    "Acceptance fixture: this EEG change contains a real EEGPrep-style bug. "
    "Review normally and report only concrete defects introduced by the patch."
)
BENIGN_PROMPT = (
    "Calibration fixture: this EEG change validates 1-based sample bounds and "
    "preserves channel-major data. Do not flag it unless there is a concrete bug."
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="test-review-harness",
        description="Create a temporary EEG-style repo and run the adapted autoreview helper.",
    )
    parser.add_argument("--fixture", choices=("buggy", "benign"), default="buggy")
    parser.add_argument("--engine", action="append", choices=ENGINES, dest="engines")
    parser.add_argument(
        "--dry-run", action="store_true", help="Verify target/bundle setup without spending a model call."
    )
    return parser.parse_args(argv)


def run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def write_fixture_file(repo: Path, content: str) -> None:
    (repo / "eeg_ops.py").write_text(content, encoding="utf-8", newline="\n")


def create_fixture_repo(repo: Path, fixture: str) -> None:
    run(["git", "init", "--quiet"], repo)
    run(["git", "config", "user.name", "Review Fixture"], repo)
    run(["git", "config", "user.email", "review-fixture@example.com"], repo)
    write_fixture_file(repo, SAFE_INITIAL)
    run(["git", "add", "eeg_ops.py"], repo)
    run(["git", "commit", "--quiet", "-m", "initial safe EEG trim"], repo)
    write_fixture_file(repo, BUGGY_CHANGED if fixture == "buggy" else BENIGN_CHANGED)


def run_reviews(repo: Path, script_dir: Path, fixture: str, engines: list[str], *, dry_run: bool) -> None:
    autoreview = script_dir / "autoreview"
    for engine in engines:
        print(f"== {engine} ==", flush=True)
        command = [
            sys.executable,
            str(autoreview),
            "--mode",
            "local",
            "--engine",
            engine,
            "--prompt",
            BUGGY_PROMPT if fixture == "buggy" else BENIGN_PROMPT,
        ]
        if fixture == "buggy":
            command.extend(["--require-finding", "channel", "--expect-findings"])
        if dry_run:
            command.append("--dry-run")
        run(command, repo)


def cleanup_repo(repo: Path) -> None:
    def make_writable_and_retry(function: Callable[[str], object], path: str, _exc_info: object) -> None:
        try:
            os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
            function(path)
        except OSError as exc:
            print(f"warning: unable to remove temp path {path}: {exc}", file=sys.stderr)

    if repo.exists():
        shutil.rmtree(repo, onerror=make_writable_and_retry)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    script_dir = Path(__file__).resolve().parent
    repo = Path(tempfile.mkdtemp(prefix="eegprep-autoreview-fixture."))
    try:
        create_fixture_repo(repo, args.fixture)
        run_reviews(repo, script_dir, args.fixture, args.engines or ["codex"], dry_run=args.dry_run)
    except subprocess.CalledProcessError as exc:
        return int(exc.returncode or 1)
    finally:
        cleanup_repo(repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
