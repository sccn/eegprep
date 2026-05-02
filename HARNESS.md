# EEGPrep Testing Harness

This document tracks the testing infrastructure that exists to enforce Python parity with EEGLAB and the changes made to that harness over time.

## Harness Objectives

- Treat EEGLAB as the reference implementation for numerical behavior, saved dataset structure, workflow behavior, and plot output.
- Support multiple oracle modes so parity checks can run in different environments:
  - live MATLAB engine
  - live MATLAB batch CLI
  - artifact-based references
- Keep tolerances, case coverage, and temporary deviations explicit and reviewable.
- Produce machine-readable and human-readable reports for CI and debugging.

## Current Core Components

- `src/eegprep/parity/config.py`
  - Loads the parity manifest and known-deviation registry.
  - Defines typed models for tolerance profiles, parity cases, and approved deviations.
- `src/eegprep/parity/compare.py`
  - Implements structured comparison results.
  - Provides comparators for numeric outputs, EEG structures, event/epoch payloads, stage outputs, visual outputs, and workflow traces.
- `src/eegprep/parity/oracle.py`
  - Detects and resolves MATLAB engine, MATLAB batch, and artifact oracles.
- `src/eegprep/parity/report.py`
  - Emits Markdown and JSON reports for parity runs.
- `src/eegprep/resources/parity_manifest.toml`
  - Declares parity coverage, tiers, oracle choices, and tolerance profiles.
- `src/eegprep/resources/parity_known_deviations.toml`
  - Records approved temporary exceptions with scope and expiry.

## Changes Added In This Pass

### 1. Introduced a dedicated parity package

Added a new `eegprep.parity` package with:

- manifest loading
- known-deviation loading
- structured comparison results
- reusable comparator APIs
- oracle backend detection and resolution
- JSON and Markdown reporting
- a small CLI for manifest inspection

This creates a single place for parity logic instead of scattering the rules across individual tests.

### 2. Added manifest-driven governance

Added packaged TOML resources for:

- tolerance profiles
- parity cases
- surface and tier classification
- oracle backend expectations
- approved temporary deviations

This makes parity coverage explicit and reviewable instead of encoding all policy inside test code.

### 3. Unified stage comparison with the new comparator layer

Updated `src/eegprep/utils/stage_comparison.py` so stage-level `.set` comparisons now reuse `compare_stage_outputs()` from the new parity layer rather than doing a standalone data-only diff.

That gives stage comparison a path toward:

- consistent numeric metrics
- structure comparison
- event and epoch comparison
- shared reporting semantics

### 4. Added harness smoke tests

Added `tests/test_parity_harness.py` to verify:

- default manifest loading
- default deviation loading
- comparator behavior
- workflow-trace normalization
- oracle backend detection and selection
- report generation

These tests are intentionally lightweight so the harness itself can be validated quickly.

### 5. Added pytest entrypoint and CI wiring

Added:

- `pytest.ini`
- a `tests` optional dependency in `pyproject.toml`
- packaged `.toml` parity resources

Updated CI to:

- install `.[all,tests]`
- inspect the parity manifest before running tests
- execute tests with `pytest`

This keeps the project compatible with the existing suite while establishing `pytest` as the top-level runner for the harness going forward.

### 6. Kept parity APIs scoped to `eegprep.parity`

The parity harness is intentionally exposed through `eegprep.parity`, not the top-level `eegprep` namespace. This keeps testing infrastructure separate from the signal-processing public API and avoids eager parity imports during ordinary `import eegprep`.

### 7. Added MATLAB MCP support for parity work

Downloaded and configured `neuromechanist/matlab-mcp-tools` as an optional developer workbench for EEGPREP parity debugging. Shared project docs should use placeholders rather than machine-specific paths:

- `MATLAB_MCP_TOOLS_ROOT`: checkout of `neuromechanist/matlab-mcp-tools`
- `MATLAB_MCP_BOOTSTRAP`: bootstrap script that initializes the EEGPREP/EEGLAB MATLAB path
- `EEGPREP_ROOT`: local EEGPREP checkout
- `EEGLAB_ROOT`: local EEGLAB checkout
- `MATLAB_PATH`: MATLAB installation root or executable location
- `EEGPREP_PYTHON`: Python environment used for EEGPREP development

Example project-pinned MCP environment:

- `EEGPREP_ROOT=/path/to/eegprep`
- `EEGLAB_ROOT=/path/to/eeglab`
- `MATLAB_PATH=/path/to/MATLAB/R20XXx`
- `EEGPREP_PYTHON=/path/to/python`

Some hosts may need to preload the Python environment `libstdc++.so.6` for MATLAB engine compatibility. If needed, put that in the local wrapper script rather than hardcoding it into shared harness code. MCP lifecycle logs must go to stderr so stdout remains valid stdio MCP protocol traffic.

The bootstrap script should be the first MATLAB script executed in an MCP parity session. It changes to the EEGPREP root, adds EEGLAB to the MATLAB path, runs `eeglab('nogui')`, and asserts that core EEGLAB functions such as `pop_loadset` and `eeg_checkset` are available.

Recommended MCP workflow for parity debugging:

- Run `eegprep_bootstrap.m` once at session start with `execute_script`.
- Use `execute_script` or `execute_section_by_title` to run focused EEGLAB oracle snippets.
- Use `get_variable`, `get_struct_info`, and `list_workspace_variables` to inspect `EEG`, `ALLEEG`, `STUDY`, stage outputs, and intermediate values without adding throwaway MATLAB files.
- Use `get_figure_metadata`, `get_plot_data`, and `analyze_figure` for visual parity investigations such as `topoplot`, ERP plots, channel layouts, and GUI-adjacent plot behavior.
- Use `matlab_lint` on generated oracle scripts and MATLAB bridge scripts before promoting them into the artifact or CI harness.

The MCP server is intended as an agent/developer workbench for creating, debugging, and explaining parity oracles. The automated harness should continue to use manifest-driven pytest checks, artifact-backed references, and MATLAB batch/engine runners for repeatable CI gates. Once an MCP investigation produces a stable oracle, promote the script and expected outputs into the manifest-controlled harness rather than depending on MCP-only state.

## Notes

- The current harness foundation is focused on parity plumbing and governance, not yet on full coverage of all EEGLAB functionality.
- Existing ad hoc parity tests still exist and can be migrated incrementally onto the new parity package over time.
- GUI-level parity is still a future layer; the current harness covers APIs, workflows, saved datasets, and plot outputs.
- Newly registered Codex MCP servers are normally available to fresh Codex sessions; the current session may need a restart before the `matlab-eegprep` tools appear in the tool list.
- EEGLAB may print plugin-version/update notices during `eeglab('nogui')` startup depending on the active user `eeg_options.m`; these notices are startup noise and should not be treated as parity output.
