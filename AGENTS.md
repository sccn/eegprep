# Agent Guidelines for EEGPREP

EEGPREP is a Python port of core EEGLAB preprocessing concepts, workflows, file names, data structures, GUI patterns, and user experience. Build features so EEGLAB users can predict where code lives and how APIs behave, while still using simple idiomatic Python when MATLAB style would make the code worse.

Primary references:
- EEGLAB source: https://github.com/sccn/eeglab
- EEGLAB data structures: https://eeglab.org/tutorials/ConceptsGuide/Data_Structures.html

## Repo Map

- `src/eegprep/functions/popfunc/`: EEGLAB-style `pop_*` user-facing wrappers. Keep each pop function in a `pop_<name>.py` module that mirrors `functions/popfunc/` in EEGLAB.
- `src/eegprep/functions/guifunc/`: EEGLAB-style GUI helpers such as `inputgui`, dialog specs, and Qt rendering. Keep GUI infrastructure parallel to `functions/guifunc/` in EEGLAB.
- `src/eegprep/functions/adminfunc/`: EEGLAB-style administrative helpers such as `eeg_checkset.py` and `eeg_options.py`.
- `src/eegprep/functions/sigprocfunc/`: EEGLAB-style signal processing functions such as `runica.py`, `topoplot.py`, `epoch.py`, and ICA wrappers.
- `src/eegprep/plugins/clean_rawdata/`: Python ports of the EEGLAB clean_rawdata plugin, including `clean_*` and ASR modules.
- `src/eegprep/plugins/ICLabel/`: Python ports of the EEGLAB ICLabel plugin and bundled `netICL.mat`.
- `src/eegprep/*.py`: package infrastructure, BIDS pipeline entry points, format conversion, MATLAB compatibility, and Python-specific helpers that do not have a closer EEGLAB function/plugin folder.
- `src/eegprep/utils/`: shared concrete helpers. Search here before adding utility code.
- `src/eegprep/resources/`: MATLAB option files, montages, package data.
- `src/eegprep/eeglab/`: vendored EEGLAB reference code and sample data. Treat as reference input; do not edit unless explicitly updating the bundled reference.
- `src/eegprep/matlab_local_tests/` and `scripts/*.m`: MATLAB parity helpers.
- `tests/`: `unittest` tests. Test files generally mirror source module names.
- `docs/source/`: Sphinx docs, examples, API pages.
- `.github/workflows/test.yml`: CI test and pre-commit entry points.
- `pre-commit.py`: required lint/check script for this repo.

## Before Coding

- Check whether a matching skill exists. Skills are task-focused playbooks in `.agents/skills/` and are also accessible as `.claude/skills/`. Before starting any non-trivial task, scan the skill descriptions in your system prompt; if one matches, invoke it via the Skill tool instead of using ad-hoc commands.
- Use `.agents/skills/eeglab-gui-visual-parity/SKILL.md` when building or iterating on EEGPrep GUI features, especially `pop_*` dialogs that should match EEGLAB screenshots through the visual parity capture loop.
- State assumptions before implementing. If the request has multiple plausible interpretations, present them.
- If something is unclear, stop and ask. Do not hide confusion in code.
- If a simpler approach exists, say so. Push back on speculative features, compatibility shims, or unnecessary abstractions.
- For multi-step work, state a short plan with verification for each step. Include code snippets when they clarify the intended change.
- Define verifiable success criteria. For example: bug fix means reproduce with a failing test, implement, then pass the test; feature means update behavior, tests, docs, and pre-commit.

## EEGLAB Parity

- Keep naming and directory structure as close to EEGLAB as practical. Put `pop_*` wrappers in `functions/popfunc`, GUI helpers in `functions/guifunc`, administrative functions in `functions/adminfunc`, signal-processing functions in `functions/sigprocfunc`, clean_rawdata ports in `plugins/clean_rawdata`, and ICLabel ports in `plugins/ICLabel`.
- Before porting or changing behavior, inspect the matching MATLAB file under `src/eegprep/eeglab/functions/` or `src/eegprep/eeglab/plugins/`.
- Preserve EEG dict semantics unless the user asks for a new abstraction. Core fields include `data`, `nbchan`, `pnts`, `trials`, `srate`, `xmin`, `xmax`, `times`, `chanlocs`, `event`, `urevent`, `epoch`, `history`, `icaact`, `icawinv`, `icasphere`, `icaweights`, and `icachansind`.
- Data is channel-major: continuous data is usually `(nbchan, pnts)`, epoched data is usually `(nbchan, pnts, trials)`.
- Be explicit about MATLAB/Python indexing boundaries. EEGLAB event latencies and many user-facing indices are 1-based; Python arrays are 0-based internally. Test first/last sample and boundary-event behavior.
- Prefer explicit inputs and return values. Do not introduce hidden global state; EEGLAB avoids globals for processing functions.
- Do not force MATLAB style blindly. If Python style and EEGLAB parity conflict, choose the simpler maintainable implementation and document the tradeoff in code, tests, or PR notes.

## Code Style

- Make the smallest change that solves the request. Every changed line should trace to the task.
- Do not add features beyond what was asked. No speculative configurability, future-proofing, or error handling for impossible states.
- If a solution is 200 lines and could be 50, rewrite it.
- Touch only files you must. Do not refactor adjacent code, reformat unrelated blocks, or delete pre-existing dead code unless asked.
- Remove imports, variables, functions, and files that your change made unused. Do not clean up pre-existing dead code unless asked.
- Match existing local style, even where it is imperfect.
- All imports go at the top. No local imports except for circular dependency breaks or optional dependency guards. Do not use `TYPE_CHECKING` guards; fix cycles structurally, often with protocols or smaller modules.
- Prefer top-level functions over classes when code does not mutate shared state. Avoid deep inheritance.
- Use early returns to reduce nesting.
- Prefer logging over `print`, except in scripts and temporary debugging.
- Resolve environment-dependent defaults once and fail fast on unknown inputs.
- No ad-hoc compatibility hacks such as `hasattr(obj, "old_attr")`; update call sites consistently.
- No backward compatibility by default. Replace old APIs and update all call sites. Add compatibility shims only when explicitly requested.
- Prefer small concrete helpers over indirection-heavy abstractions. Do not add abstractions for single-use code.
- Put magic strings and magic numbers in top-level constants when they are reused or semantically important.
- Before writing any utility, helper, or data structure, search for existing implementations and check the standard library, current dependencies, and appropriate third-party packages before adding new code or dependencies. If a suitable implementation exists, use it.

## Public APIs and Docs

- Public APIs need concise docstrings. Use the style already present in nearby code; Google-style is preferred for new public functions.
- Skip docstrings for trivial private helpers with clear names.
- When adding a feature or changing user-facing behavior, update Sphinx docs under `docs/source/`. Update examples/API pages when relevant.
- Keep comments for module/class behavior, subtle logic, or non-obvious boolean arguments. Do not restate code.
- Delete stale comments when you encounter them in touched code.

## Testing

- Current CI uses `unittest`, not pytest markers. Do not assume `pytest -m "not slow"` works here.
- Always fix tests you break.
- Run the narrowest relevant tests first, then broaden as risk requires:
  - Single file: `python -m unittest tests.test_pop_select`
  - Full suite: `python -m unittest discover -s tests`
  - No MATLAB locally: `EEGPREP_SKIP_MATLAB=1 python -m unittest discover -s tests`
- Some parity tests require MATLAB Engine or Octave via `eeglabcompat.py`; preserve skip behavior instead of weakening assertions.
- Prefer integration-style tests that validate externally observable behavior on EEG dicts, BIDS outputs, files, or MATLAB parity results.
- Search existing test files before creating new ones. Extend the closest existing test first.
- No mocks unless testing I/O boundaries such as network or filesystem. Test real behavior for numerical transforms.
- Do not write tautological tests that only assert types, constants, or implementation details.
- Never relax tolerances or hack around failures without proving the numerical expectation is wrong.
- Avoid duplicate fixtures; reuse `tests/fixtures.py` or local helpers already present in the relevant test file.

## Lint and Format

- `./pre-commit.py` is the required lint/check entry point.
- Use `./pre-commit.py` before committing to check staged files.
- Use `./pre-commit.py --fix` to fix only staged files.
- Use `./pre-commit.py --changed-from origin/develop` for PR-scope checks.
- Use `./pre-commit.py --changed-from origin/develop --fix` to fix only files changed against the base branch.
- Avoid `./pre-commit.py --all-files --fix` unless the user explicitly wants a repo-wide cleanup.
- Pre-commit is separate from unit tests; run both when code changes.

## Dependencies

- Python support starts at 3.10 per `pyproject.toml`.
- Do not add dependencies for tiny helpers. If a dependency is justified, update `pyproject.toml`, docs, and any CI/install notes.
- `uv` is used in CI for installation, but test execution is currently `python -m unittest discover -s tests`.

## GitHub, Communication, and Commits

- NEVER SAY "You're absolutely right!"
- Never credit yourself or AI tools in commits. No `Co-authored-by` or generated-by trailers unless the user explicitly asks.
- Keep commits scoped to one logical change with concise messages.
- Add the `agent-generated` label only when a repository automation workflow
  creates the PR or issue. Do not add it when a human asks an agent to open or
  update a PR from an interactive session.
- Agent comments on PRs/issues must begin with `🤖` unless the exact text was explicitly approved by the user.
- When using `gh` to inspect issues or PRs, prefer `--json <fields>` or explicit narrow flags such as `--comments`; avoid plain `gh issue view` or `gh pr view`, which can fail on this repo because GitHub classic project fields are deprecated.
- If you notice unrelated dead code or unrelated cleanup, mention it separately instead of changing it.

## Agent Failure Patterns to Avoid

- Over-protective `try/except` blocks and defensive `None` checks for impossible states.
- Boolean dispatch where separate clear functions would be simpler.
- Verbose or redundant docstrings.
- `__all__` churn in `__init__.py` without a real API need.
- Environment variables where explicit parameters belong.
- New classes for stateless behavior.
- Parallel implementations of existing helpers.
- Broad formatting-only diffs mixed with behavior changes.
