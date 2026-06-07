# EEGPrep Final Standalone Parity Audit

Audit date: 2026-06-07
Epic: #157
Phase issue: #158
Machine-readable contract: `docs/parity/eeglab_final_parity_matrix.json`
Validator: `uv run --no-sync python -m tools.eeglab_final_parity_matrix --json`

## Purpose

This audit expands the completed core parity matrix into the final standalone
product-completion surface for EEGPrep. It covers the remaining non-stale
workflows that are not fully represented by
`docs/parity/eeglab_core_parity_matrix.json`:

- bundled plugin depth for `clean_rawdata`, `firfilt`, `ICLabel`/viewprops, and
  `dipfit`;
- MATLAB object/storage folders `@eegobj`, `@memmapdata`, and `@mmo`;
- optional-toolbox workflows such as LIMO, PAC, FieldTrip-style STUDY helpers,
  Riemannian ASR, and DIPFIT fitting;
- EEGLAB tutorial/doc surfaces that should become EEGPrep-owned Sphinx docs.

This phase does not port feature behavior. It defines the rows, status values,
phase ownership, optional-dependency rules, documentation architecture, and
validator behavior that later phase agents must use.

## Status Taxonomy

Rows in the final matrix use these statuses:

- `implemented`: EEGPrep has the standalone behavior; the responsible phase
  owns verification, docs, and final evidence.
- `partial`: EEGPrep has part of the behavior, but important options, GUI
  paths, diagnostics, numerical parity, or docs still need phase work.
- `port`: a useful workflow is not implemented yet and should be ported or
  redesigned as native Python/Qt behavior.
- `consolidated`: EEGPrep intentionally covers the behavior through a different
  Python helper or public contract instead of a same-name file.
- `stale_skip`: a MATLAB demo, test, packaging helper, or obsolete alias is not
  a user workflow. These rows require the full stale-policy object with every
  field set to `false`.
- `matlab_runtime_skip`: MATLAB path, command-window, GUI shim, or toolbox
  activation behavior that must not exist in installed EEGPrep runtime code.
- `optional_dependency`: a real scientific workflow needs a substantial backend
  decision. The row must name the dependency, fallback behavior, user-facing
  message, and phase contract.
- `external_plugin`: behavior belongs to an external plugin ecosystem and
  should use EEGPrep extension contracts rather than core package code.
- `docs_gap`: a documentation/tutorial surface that Phase 7 must write after
  feature phases define final behavior.

All non-skip rows must name a responsible phase and the matching phase issue.
Skip rows must have `responsible_phase: "none"` and `phase_issue: null`.

## Phase Ownership

- Phase 2 / #159 owns `clean_rawdata` and FIRFilt rows, including standard ASR,
  Riemannian ASR optional-backend decisions, `vis_artifacts`, FIR order
  calculators, boundary helpers, reports, and frequency-response plotting.
- Phase 3 / #160 owns DIPFIT settings, fitting, FieldTrip/source-localization
  boundaries, atlas and coordinate transforms, leadfield/LORETA workflows, and
  dipole plotting evidence.
- Phase 4 / #161 owns optional LIMO/PAC/STUDY statistics behavior and
  FieldTrip-style STUDY neighbor/interpolation/source workflows. It must
  coordinate with Phase 3 for source-localization assumptions.
- Phase 5 / #162 owns Python-native large-dataset storage, `storedisk`,
  `option_memmapdata`, and the product decision for `@memmapdata`/`@mmo`
  semantics. It should not port MATLAB overloads one-for-one.
- Phase 6 / #163 owns ICLabel, label statistics, viewprops, component-property
  diagnostics, alternate runtime/network decisions, and visual evidence.
- Phase 7 / #164 owns the EEGLAB-style Sphinx docs architecture and all
  `docs_gap` tutorial rows. It should merge after feature phases describe
  actual completed behavior rather than intentions.
- Phase 8 / #165 owns final integration, release hardening, docs build,
  non-slow tests, MATLAB parity where available, visual parity suite, GUI Agent
  mixed-flow QA, and evidence rollup.

## Optional-Dependency Rules

EEGPrep should prefer standalone Python behavior for core preprocessing, data
structures, GUI/console state, saved-file behavior, and bundled plugin workflows
that can be tested without external scientific toolboxes.

Use `optional_dependency` only when the workflow requires a substantial backend
that can be installed, versioned, tested, and documented. These rows must not
produce fake outputs. Until a backend is selected, user-facing functions should
raise clear limitations that name the missing backend and point to EEGPrep docs.

Use `external_plugin` for broad third-party ecosystems that do not belong in
core EEGPrep. Those workflows should be built through the extension API,
catalog/trust model, packaged help, tests, and docs.

## Documentation Architecture

Phase 7 should reorganize and expand Sphinx docs into an EEGLAB-style manual
with EEGPrep-specific behavior:

1. Installation and optional dependencies.
2. Concepts guide: EEG structures, events, epochs, channel locations, ICA,
   STUDY, history, `EEG`, `ALLEEG`, `CURRENTSET`, and indexing boundaries.
3. GUI tutorials and menu workflows.
4. Command line, `eegprep-console`, `LASTCOM`, `ALLCOM`, and script replay.
5. Preprocessing, filtering, cleaning, and artifact diagnostics.
6. ICA, ICLabel, rejection, and visual diagnostics.
7. STUDY, statistics, PAC, source localization, and optional backend limits.
8. Bundled plugins and external extensions.
9. API reference.
10. EEGLAB migration notes mapping familiar MATLAB workflows to EEGPrep GUI,
    Python, and console workflows.
11. Developer parity contracts, matrix maintenance, visual evidence, and the
    standalone runtime boundary.

The current docs already have useful pages under `docs/source/user_guide/`,
`docs/source/api/`, and `docs/source/examples/`. Phase 7 should reorganize and
extend those pages rather than generating API dumps or copying EEGLAB prose.

## Runtime Contract

The installed `eegprep` package must not read, import from, or shell out to
`src/eegprep/eeglab`. That checkout is a development oracle for audits, parity
tests, and tooling only. Runtime help text, options, sample resources, and docs
must be EEGPrep-owned packaged resources.

This phase adds `tests/test_runtime_eeglab_independence.py` to scan package
Python files for vendored-reference dependency patterns. Tooling under `tools/`
may read the vendored EEGLAB tree because validation is a development task.

## Validator Contract

`tools.eeglab_final_parity_matrix` discovers 180 final-epic EEGLAB reference
paths from the vendored checkout:

- plugin paths from the four bundled plugin roots, excluding MatConvNet and
  Manopt third-party library internals/examples/tests;
- object/storage MATLAB class folder files;
- EEGLAB tutorial scripts and Live Scripts.

Rows may group several source paths into one workflow, but each discovered
reference path must appear exactly once. Missing paths, duplicated paths,
missing optional-dependency contracts, stale-skip policy mistakes, missing docs
architecture sections, and inconsistent phase ownership fail validation.

The existing core command remains separate and must still pass:

```bash
uv run --no-sync python -m tools.eeglab_parity_matrix --json
```

The final epic command is:

```bash
uv run --no-sync python -m tools.eeglab_final_parity_matrix --json
```
