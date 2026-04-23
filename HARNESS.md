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

### 6. Exported parity APIs from `eegprep`

Updated `src/eegprep/__init__.py` so the new parity utilities are available from the top-level package namespace where that is convenient for tests and tooling.

## Notes

- The current harness foundation is focused on parity plumbing and governance, not yet on full coverage of all EEGLAB functionality.
- Existing ad hoc parity tests still exist and can be migrated incrementally onto the new parity package over time.
- GUI-level parity is still a future layer; the current harness covers APIs, workflows, saved datasets, and plot outputs.
