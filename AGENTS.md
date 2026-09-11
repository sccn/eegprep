# Agent Guidelines for EEGPREP

EEGPREP is a Python port of core EEGLAB preprocessing concepts, workflows, file names, data structures, GUI patterns, and user experience. Build features so EEGLAB users can predict where code lives and how APIs behave, while still using simple idiomatic Python when MATLAB style would make the code worse. Remember, while we are doing a porting project, EEGPrep must work well standalone and must be a delight to use for EEG Researchers.

Primary references:
- EEGLAB source: https://github.com/sccn/eeglab
- EEGLAB data structures: https://eeglab.org/tutorials/ConceptsGuide/Data_Structures.html

## Repo Map

- `src/eegprep/functions/popfunc/`: EEGLAB-style `pop_*` user-facing wrappers and `eeg_*` functions that operate on EEG structures, such as ICA wrappers. Keep each pop function in a `pop_<name>.py` module that mirrors `functions/popfunc/` in EEGLAB.
- `src/eegprep/functions/guifunc/`: EEGLAB-style GUI helpers such as `inputgui`, dialog specs, and Qt rendering. Keep GUI infrastructure parallel to `functions/guifunc/` in EEGLAB.
- `src/eegprep/functions/adminfunc/`: EEGLAB-style administrative helpers such as `eeg_checkset.py`, `eeg_options.py`, the main GUI launcher, and the synchronized `eegprep-console` workspace.
- `src/eegprep/functions/sigprocfunc/`: EEGLAB-style low-level signal processing functions such as `runica.py`, `runamica.py`, `topoplot.py`, `epoch.py`, and `eegrej.py`.
- `src/eegprep/plugins/clean_rawdata/`: Python ports of the EEGLAB clean_rawdata plugin, including `clean_*` and ASR modules.
- `src/eegprep/plugins/clean_rawdata/private/`: ports of clean_rawdata private helpers such as `fit_eeg_distribution`, `geometric_median`, FIR helpers, covariance helpers, and spherical-spline interpolation.
- `src/eegprep/plugins/ICLabel/`: Python ports of the EEGLAB ICLabel plugin and bundled `netICL.mat`.
- `src/eegprep/plugins/firfilt/`: Python ports of the EEGLAB firfilt plugin helpers.
- `src/eegprep/functions/miscfunc/`: EEGLAB-style miscellaneous helpers, including format conversion and numerical utilities.
- `src/eegprep/functions/eegobj/`: Python counterpart to EEGLAB's `functions/@eegobj/`.
- `src/eegprep/plugins/EEG_BIDS/`: Python ports and workflow helpers for the EEGLAB EEG-BIDS plugin.
- `src/eegprep/utils/`: Python-only test/development support. Do not put EEGLAB-equivalent processing code here.
- `src/eegprep/resources/`: MATLAB option files, montages, help text, and package data.
- `src/eegprep/resources/help/`: EEGPrep-owned Markdown help resources for GUI Help buttons and EEGLAB-style `pophelp` text. Add `<function_name>.md` here whenever new user-facing functionality needs GUI help.
- `src/eegprep/eeglab/`: vendored EEGLAB reference code for local development and parity work. Treat as reference input; do not edit unless explicitly updating the bundled reference. Do not make package runtime behavior depend on this directory existing.
- `tests/matlab/`: MATLAB parity scripts and MATLAB helper fixtures used by Python tests.
- `scripts/*.m`: MATLAB/Octave helper scripts that are not part of the normal unit-test tree.
- `sample_data/`: small checked-in EEG sample datasets, named to match EEGLAB's `sample_data` convention.
- `sample_notebooks/`: exploratory/sample notebooks. Keep runnable examples in docs when they are user-facing.
- `tools/`: developer and parity tooling that is not installed as part of the `eegprep` package.
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

- Use EEGLAB as a development and parity oracle, not as an EEGPrep runtime dependency. During development, compare against EEGLAB so EEGPrep features look, feel, and behave like EEGLAB for EEGLAB users; at runtime, EEGPrep must work standalone without an EEGLAB checkout.
- EEGLAB users must feel at home while using EEGPrep. For GUI-based features, match EEGLAB's alignment, arrangement, labels, buttons, text fields, control order, default values, enabled/disabled states, and workflow before introducing Python-only improvements.
- Keep naming and directory structure as close to EEGLAB as practical. Put `pop_*` wrappers in `functions/popfunc`, GUI helpers in `functions/guifunc`, administrative functions in `functions/adminfunc`, signal-processing functions in `functions/sigprocfunc`, clean_rawdata ports in `plugins/clean_rawdata`, and ICLabel ports in `plugins/ICLabel`.
- Before porting or changing behavior, inspect the matching MATLAB file under `src/eegprep/eeglab/functions/` or `src/eegprep/eeglab/plugins/`.
- Runtime code in the installed `eegprep` package must work without `src/eegprep/eeglab/` present. Do not read from, import from, shell out to, or otherwise depend on the EEGLAB checkout in package runtime paths.
- If EEGLAB-like text, examples, or Help content are needed at runtime, make them EEGPrep-owned packaged resources. For GUI Help and `pophelp`, add Markdown files under `src/eegprep/resources/help/<function_name>.md`; missing help resources should fail clearly instead of falling back to the EEGLAB checkout or Python docstrings.
- Preserve EEG dict semantics unless the user asks for a new abstraction. Core fields include `data`, `nbchan`, `pnts`, `trials`, `srate`, `xmin`, `xmax`, `times`, `chanlocs`, `event`, `urevent`, `epoch`, `history`, `icaact`, `icawinv`, `icasphere`, `icaweights`, and `icachansind`.
- Data is channel-major: continuous data is usually `(nbchan, pnts)`, epoched data is usually `(nbchan, pnts, trials)`.
- Be explicit about MATLAB/Python indexing boundaries. EEGLAB event latencies and many user-facing indices are 1-based; Python arrays are 0-based internally. Test first/last sample and boundary-event behavior.
- Prefer explicit inputs and return values. Do not introduce hidden global state; EEGLAB avoids globals for processing functions.
- Do not force MATLAB style blindly. If Python style and EEGLAB parity conflict, choose the simpler maintainable implementation and document the tradeoff in code, tests, or PR notes.

## GUI And Console Workspace

- EEGPrep's primary interactive workflow is mixed GUI plus `eegprep-console`. Both share one `EEGPrepSession`; `EEG`, `ALLEEG`, `CURRENTSET`, `LASTCOM`, `ALLCOM`, `STUDY`, and `CURRENTSTUDY` must stay synchronized.
- Treat GUI plus console flow as one user experience, not two separate clients. EEGLAB users commonly switch between menus/dialogs and command history/workspace inspection, so GUI actions, console commands, dataset selection, history, and visible state must stay seamless in both directions.
- `EEGPrepSession.CURRENTSET` is a list of EEGLAB-facing 1-based dataset indices; expose it as `0`, scalar `n`, or list `[n, ...]` in the console. Use `selected_dataset_indices()` for read-only multi-dataset state.
- GUI/menu actions must update datasets and history through `EEGPrepSession` helpers such as `store_current`, `add_history`, and `notify_changed`; do not mutate GUI-only state that the console cannot see.
- User-facing `pop_*` functions should support `return_com=True` and return EEGLAB-style history commands. The console wrappers rely on `(EEG, com)` results to auto-store bare calls like `pop_reref(EEG, [])`.
- When adding or changing a GUI-reachable function, test both interaction directions when relevant: GUI action then console inspection, and console command then GUI refresh/history.
- When GUI/console synchronization, history behavior, dataset selection, STUDY state, or registered menu actions change, update the Sphinx workflow docs so users understand how to move between the GUI and `eegprep-console`, and update developer notes when `EEGPrepSession` contracts change.
- Menu placeholders must carry phase/exclusion metadata in `menu_placeholders.py`; mark EEGBrowser/eegplot-style scrolling workflows with an explicit exclusion instead of treating them as ordinary TODOs.

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
- Any user-facing EEGPrep change must include matching Sphinx updates under `docs/source/` in the same branch unless the change has no user-visible behavior. Update workflow/tutorial pages, examples, generated or hand-written API pages, troubleshooting notes, and migration notes where relevant.
- Write docs from an EEGPrep standalone user point of view. Use EEGLAB comparisons when they clarify migration, parity, file compatibility, or indexing boundaries, but do not frame normal EEGPrep behavior as merely "EEGLAB-style."
- When adding a user-facing function or GUI dialog with a Help button, add or update the corresponding Markdown help resource in `src/eegprep/resources/help/`, for example `src/eegprep/resources/help/pop_reref.md`.
- Keep comments for module/class behavior, subtle logic, or non-obvious boolean arguments. Do not restate code.
- Delete stale comments when you encounter them in touched code.

## Testing

- Pytest is the default test runner. Existing tests may still use `unittest.TestCase`; do not rewrite them unless the touched test benefits from pytest fixtures or parametrization.
- Registered markers include `slow`, `matlab`, `octave`, `gui`, `visual`, and `parity`.
- Legacy `unittest` tests are categorized by path/name in `tests/conftest.py`; update that map when adding obvious slow, MATLAB, GUI, visual, or parity coverage.
- Always fix tests you break.
- Run the narrowest relevant tests first, then broaden as risk requires:
  - Single file: `uv run pytest tests/test_pop_select.py`
  - Marker subset: `uv run pytest -m "not slow"`
  - Full suite: `uv run pytest tests`
  - No MATLAB locally: `EEGPREP_SKIP_MATLAB=1 uv run pytest tests`
- Some parity tests require MATLAB Engine or Octave via `src/eegprep/functions/adminfunc/eeglabcompat.py`; preserve skip behavior instead of weakening assertions.
- Prefer integration-style tests that validate externally observable behavior on EEG dicts, BIDS outputs, files, or MATLAB parity results.
- For features reachable from `eegprep-console`, add or update `tests/test_console_workspace.py` coverage when namespace sync, history, import aliases, or bare `pop_*` auto-store behavior can be affected.
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
- CI also runs `uv run --no-sync ruff check .`, `uv run --no-sync ruff format --check .`, and `uv run --no-sync ty check`; run these locally after broad formatting, import, or typing-sensitive changes.
- Pre-commit is separate from unit tests; run both when code changes.

## Dependencies

- Python support starts at 3.10 per `pyproject.toml`.
- `uv` is the default package and environment manager for development, CI, and agent workflows.
- `.python-version` sets the default development interpreter to Python 3.11.
- Use `uv sync --group dev` after cloning or when dependencies change.
- Use `uv run python ...` for Python commands so tests run inside the project environment.
- Do not add dependencies for tiny helpers. If a dependency is justified, update `pyproject.toml`, docs, and any CI/install notes.

## GitHub, Communication, and Commits

- NEVER SAY "You're absolutely right!"
- Never credit yourself or AI tools in commits. No `Co-authored-by` or generated-by trailers unless the user explicitly asks.
- Keep commits scoped to one logical change with concise messages. Commit often.
- Add the `agent-generated` label only when a repository automation workflow
  creates the PR or issue. Do not add it when a human asks an agent to open or
  update a PR from an interactive session.
- Agent comments on PRs/issues must begin with `🤖` unless the exact text was explicitly approved by the user.
- When creating a PR that includes GUI changes, attach side-by-side GUI comparison images as GitHub user attachments using `gh attach --comment`; do not commit those images or visual parity artifacts to the repo.
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

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:970c3bf2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   bd dolt push
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->

<!-- BEGIN BEADS CODEX SETUP: generated by bd setup codex -->
## Beads Issue Tracker

Use Beads (`bd`) for durable task tracking in repositories that include it. Use the `beads` skill at `.agents/skills/beads/SKILL.md` (project install) or `~/.agents/skills/beads/SKILL.md` (global install) for Beads workflow guidance, then use the `bd` CLI for issue operations.

### Quick Reference

```bash
bd ready                # Find available work
bd show <id>            # View issue details
bd update <id> --claim  # Claim work
bd close <id>           # Complete work
bd prime                # Refresh Beads context
```

### Rules

- Use `bd` for all task tracking; do not create markdown TODO lists.
- Run `bd prime` when Beads context is missing or stale. Codex 0.129.0+ can load Beads context automatically through native hooks; use `/hooks` to inspect or toggle them.
- Keep persistent project memory in Beads via `bd remember`; do not create ad hoc memory files.

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.
<!-- END BEADS CODEX SETUP -->
