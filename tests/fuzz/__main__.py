"""Command line for nightly differential fuzzing.

``run`` compares EEGPrep with an oracle. ``reproduce`` replays one saved case.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tests.fuzz.harness import (
    FakeOracle,
    MatlabBackend,
    PythonBackend,
    default_sanity,
    default_strategies,
    nightly_seed,
    run_repro,
    run_session,
)
from tests.fuzz.issues import default_run_url, tracker_from_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Differential fuzz EEGPrep against EEGLAB")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Survey seeded cases and shrink divergences")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--examples", type=int, default=6)
    run.add_argument("--shrink-budget", type=int, default=12)
    run.add_argument("--max-shrinks", type=int, default=4)
    run.add_argument("--issue-cap", type=int, default=3)
    run.add_argument("--oracle", choices=("matlab", "fake"), default="matlab")
    run.add_argument("--fake-mode", choices=("match", "survey-diverge", "sanity-fail"), default="match")
    run.add_argument("--functions", default="", help="Comma-separated subset of the four targets")
    run.add_argument("--file-issues", action="store_true", help="File confirmed divergences with gh")

    replay = sub.add_parser("reproduce", help="Replay a saved repro JSON")
    replay.add_argument("path", type=Path)
    replay.add_argument("--oracle", choices=("matlab", "fake"), default="matlab")
    replay.add_argument("--fake-mode", choices=("match", "survey-diverge", "sanity-fail"), default="match")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("eegprep").setLevel(logging.WARNING)

    python = PythonBackend()
    if args.oracle == "fake":
        oracle = FakeOracle(args.fake_mode, python)
    else:
        oracle = MatlabBackend()

    if args.command == "reproduce":
        outcome = run_repro(args.path, python_backend=python, oracle_backend=oracle)
        print(
            f"{outcome.report} diverged={outcome.diverged} class={outcome.error_class} "
            f"abs={outcome.max_abs} rel={outcome.max_rel}"
        )
        print(outcome.detail)
        return 0 if outcome.both_ok and not outcome.diverged else 1

    seed_value = nightly_seed() if args.seed is None else args.seed
    strategies = default_strategies()
    sanity = default_sanity()
    if args.functions:
        selected = tuple(part.strip() for part in args.functions.split(",") if part.strip())
        strategies = {name: strategies[name] for name in selected}
        sanity = [case for case in sanity if case.function in selected]
    tracker = None
    if args.file_issues:
        tracker = tracker_from_env(default_run_url())
        if tracker is None:
            logging.error("refusing to file issues without GH_TOKEN or GITHUB_TOKEN")
            return 2
    result = run_session(
        seed_value=seed_value,
        output_dir=args.output,
        examples=args.examples,
        shrink_budget=args.shrink_budget,
        max_shrinks=args.max_shrinks,
        issue_cap=args.issue_cap,
        python_backend=python,
        oracle_backend=oracle,
        strategies=strategies,
        sanity=sanity,
        tracker=tracker,
        run_url=default_run_url(),
    )
    print(f"fuzz seed {result.seed} harness_ok={result.harness_ok} divergences={len(result.divergences)}")
    return 0 if result.harness_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
