# Review instructions

## What Important means here

Reserve Important for correctness. A finding is Important when the change
would produce wrong EEG results, break EEGLAB parity, mishandle an EEG
structure, hide a failure, or weaken a numerical or test bar. Style, naming,
formatting, and refactoring suggestions are not findings.

Flag these as Important:

- Any loosening of a numerical tolerance, `rtol`, `atol`, ceiling, or
  threshold, whoever the author is. `tests/fuzz/tolerances.toml` and a
  recorded summary under `tests/parity_expected/` (`ds003061_pipeline.json`,
  written by `--record-parity-summary`) are changed only by Suraj or Arno
  Delorme. Flag any other edit to those files, and flag a re-recorded parity
  summary.
- A new `skip` or `xfail`, or a weakened assertion, including a new entry in
  `tests/eeglab_python_backend_xfails.py`. Do not accept a skip or a weaker
  tolerance in place of a missing feature.
- A workaround used to make a test pass: casting or coercing a value, a broad
  `try`/`except`, a special case for a test input, or a fallback that hides
  the error. The fix has to remove the cause.
- A change to a ported function that does not match the EEGLAB source
  vendored at `src/eegprep/eeglab`, at the commit named by
  `EEGLAB_TESTS_EEGLAB_COMMIT` in `tests/eeglab_tests/__init__.py`, or a pull
  request that changes a port without citing the EEGLAB file and the lines it
  follows. The counterpart of `src/eegprep/functions/<area>/` is
  `src/eegprep/eeglab/functions/<area>/`, and `src/eegprep/plugins/<name>/`
  matches `src/eegprep/eeglab/plugins/<name>/`.
- A 0-based/1-based index mistake. Helpers that keep EEGLAB names use 1-based
  indices, including event latencies, dataset indices, and frame numbers.
  Python storage is 0-based. Check the first sample, the last sample, and
  boundary events.
- MATLAB column-major layout versus NumPy C order; a change of single versus
  double precision or of output class; a `squeeze` or shape change (MATLAB
  drops singleton dimensions but stays 2-D); a changed default.
- Copying an EEGLAB quirk or bug with no note. When EEGLAB is wrong, EEGPrep
  does the right thing, documents the difference, and locks it in with a
  test. Divergence without that note and that test is Important.
- A comment that restates the code, or a comment that justifies a workaround,
  a skipped test, or a loosened tolerance. Comments may explain non-obvious
  intent or an EEGLAB indexing, rounding, or line reference.
- A tolerance in a new test that was fitted to observed output. Derive it
  from the computation: machine epsilon, problem size, and conditioning.
  Fuzz comparisons use `abs(python - oracle) <= abs + rel * abs(oracle)`, the
  same rule as `numpy.allclose`.
- A core-function change (EEGLAB-ported processing code) whose description
  does not cite one MATLAB reference-suite run. The run is
  `.github/workflows/eeglab-reference.yml` (`workflow_dispatch`) on the pull
  request branch, or locally `uv run --no-sync pytest tests
  --eeglab-backend=matlab --eeglab-root=<pinned EEGLAB checkout>
  --eeglab-suite-root=<pinned eeglab_tests checkout>`.
  `--eeglab-backend=python` does not count. Point both roots at
  `EEGLAB_TESTS_EEGLAB_COMMIT` and `EEGLAB_TESTS_COMMIT` in
  `tests/eeglab_tests/__init__.py`. The description needs the exact command or
  workflow run, both commits, pass/fail/skip counts, runtime, and why any
  failure or skip remains. Pull requests that change only tests, docs, CI, or
  tooling do not need this run. A review that finds nothing is not approval.
  A human still approves a core-function change, and this run is part of that
  approval.
- Wrong results on a realistic input; a failure to import or run on a common
  path; unsafe file, path, or input handling; a missing test for a changed
  behavior that can regress. Name the exact missing case. A numerical test
  that mocks the transform is the same kind of gap.
- A clear `CLAUDE.md` or `AGENTS.md` violation that changes behavior. Quote
  the rule. `CLAUDE.md` imports `AGENTS.md`.

EEG data is channel-major: continuous `(nbchan, pnts)`, epoched
`(nbchan, pnts, trials)`. Core fields are `data`, `nbchan`, `pnts`, `trials`,
`srate`, `xmin`, `xmax`, `times`, `chanlocs`, `event`, `urevent`, `epoch`,
`history`, `icaact`, `icawinv`, `icasphere`, `icaweights`, and `icachansind`.

EEGLAB source is https://github.com/sccn/eeglab. Data structures are
https://eeglab.org/tutorials/ConceptsGuide/Data_Structures.html. Keep naming,
code layout, data structures, GUI behavior, and workflow familiar to an
EEGLAB user. Where parity and idiomatic Python conflict, prefer the smaller
Python form and require the pull request to state the tradeoff. Do not ask
for MATLAB style when the existing EEGPrep form is clearer.

Each Important finding names the file or function, why the result is wrong,
and a concrete fix. Cite `file:line` in the changed code or in the vendored
EEGLAB file. Do not post a finding you cannot confirm.

## Cap the nits

Report at most five Nits per review. If you found more, say "plus N similar
items" in the summary instead of posting them inline. If everything you found
is a Nit, lead the summary with "No blocking issues."

Style, formatting, naming, and changelog wording are not Nits. A missing
`docs/source/` update for a user-facing behavior change is at most one Nit.
Prefer fewer comments, each tied to a changed line. Do not comment on
unchanged code unless that code causes the bug in the diff. Do not give
generic compliments or generic warnings. When the change is sound, say so and
name what you checked: correctness, EEGLAB parity, EEG structure layout, and
tests for the changed behavior.

## Do not report

- Anything `./pre-commit.py` already enforces, including formatting and import
  order. CI also runs `uv run --no-sync ruff check .`,
  `uv run --no-sync ruff format --check .`, and `uv run --no-sync ty check`.
- The vendored MATLAB tree under `src/eegprep/eeglab`. It is a reference
  checkout, not EEGPrep code. Installed package code must not read, import,
  or shell out to it.
- Wording in `docs/source/changelog.rst`.
- Pre-existing bugs this pull request did not introduce.
- Pedantic nits, general code quality, generic security or performance
  warnings, and issues that need unlikely inputs.
- Speculative abstractions, broad refactors, or unrelated cleanup. Unrelated
  dead code is out of scope; do not ask for it in this pull request.
- A request to add tests for coverage. Ask only for a specific case the
  changed behavior can break. Do not ask to replace an `unittest` test with
  pytest unless that test is what the change broke.
- A line an existing lint ignore already silences.

## Always check

- In this order: correctness and EEGLAB behavioral parity; EEG structure
  compatibility (fields, dimensions, channels, events, epochs, metadata); edge
  cases, error handling, and invalid inputs; a test for the changed behavior,
  including an EEGLAB parity check where one is feasible; unsafe file, path,
  or input handling; performance only when the cost shows up at EEG sizes;
  then whether the code is the smallest clear Python that still matches
  EEGLAB.
- A changed `pop_*` function, signal-processing port, or plugin port cites the
  EEGLAB file and lines it follows and matches them, except for a documented
  and tested intentional difference.
- GUI and `eegprep-console` changes keep one session in sync: `EEG`,
  `ALLEEG`, `CURRENTSET` (1-based: `0`, `n`, or `[n, ...]`), `LASTCOM`,
  `ALLCOM`, `STUDY`, and `CURRENTSTUDY`. A `return_com=True` history command
  still round-trips.
- Tests are `uv run pytest` on the closest existing file, then a wider run
  when the risk is wider (`uv run pytest tests`, or
  `EEGPREP_SKIP_MATLAB=1 uv run pytest tests` when MATLAB is absent).
  Reference tests that use `eeglab_backend` or `eeglab_matlab_engine` stay
  deselected unless `--eeglab-backend` is passed. A deselected reference test
  is not a pass. Do not replace a numeric assertion with a type, shape, or
  `is not None` check.
