# EEGPrep Final Standalone Epic Closeout Evidence

Date: 2026-06-07
Phase issue: #165
Epic issue: #157
Branch: `phase/165-final-integration-release-hardening`
Base: `origin/feature/eeglab-full-standalone-completion`
Final epic PR target after this phase: `origin/develop`

Phase 8 closes the phase stack on the epic branch. It prepares evidence for the
future epic PR to `develop`, but does not open that final PR.

## Phase Issues and PRs

| Phase | Issue | PR | Scope | Result |
| --- | --- | --- | --- | --- |
| 1 | #158 | #166 | Final standalone audit matrix, validator, runtime contract, docs architecture | Merged |
| 2 | #159 | #171 | clean_rawdata and FIRFilt bundled-plugin completion | Merged |
| 3 | #160 | #169 | DIPFIT and source-localization standalone parity | Merged |
| 4 | #161 | #170 | STUDY PAC, LIMO-compatible design, and advanced statistics boundaries | Merged |
| 5 | #162 | #168 | Large-dataset `.fdt`, memmap, and `storedisk` semantics | Merged |
| 6 | #163 | #167 | ICLabel, viewprops, and component diagnostic parity | Merged |
| 7 | #164 | #172 | Standalone Sphinx manual, tutorials, API pages, and help resources | Merged |
| 8 | #165 | This branch | Integration, QA, release hardening, and evidence | In review |

## Feature Summary

The merged epic branch now covers the remaining standalone EEGLAB parity areas
identified after the core parity PR:

- A machine-readable final parity matrix and validator for bundled plugins,
  object/storage semantics, optional-toolbox boundaries, and documentation.
- clean_rawdata standard ASR, calibration-time Riemannian ASR support, explicit
  full-Riemannian limitation, artifact diagnostics, and FIRFilt helper/dialog
  parity.
- DIPFIT standalone spherical settings, grid search, nonlinear fitting,
  multifit, leadfield, dipplot, coordinate transforms, and explicit source
  backend limits.
- STUDY PAC compute/cache/read/plot workflows, LIMO-compatible design-matrix
  export, neighbor/interpolation helpers, and explicit source-statistics
  boundaries.
- Python-native large-dataset save/load, `.fdt` sidecars, memory maps,
  `storedisk` offload/retrieve behavior, and GUI/console session synchronization.
- ICLabel default-network runtime behavior, label statistics, viewprops-style
  diagnostics, and explicit alternate-network boundaries.
- EEGPrep-owned Sphinx user manual, migration notes, GUI plus console tutorials,
  BIDS workflow docs, bundled plugin docs, generated API pages, and packaged
  help resources.

## Matrix Closeout

`docs/parity/eeglab_final_parity_matrix.json` now validates against the full
final-epic reference surface:

- 31 grouped rows cover 180 final-epic EEGLAB reference paths.
- Status counts: 21 `implemented`, 3 `consolidated`, 2 `optional_dependency`,
  2 `partial`, 2 `stale_skip`, and 1 `matlab_runtime_skip`.
- There are no remaining `port` or `docs_gap` rows.
- The two `partial` rows are intentional backend boundaries, not forgotten work:
  DIPFIT MRI/BEM/LORETA/FieldTrip source workflows and STUDY-level
  FieldTrip/source-statistics workflows.

Phase 8 reclassified the six Phase 7 documentation rows from `docs_gap` to
`implemented` after verifying that the completed Sphinx manual covers console
history migration, event/indexing tutorials, STUDY workflows, source/DIPFIT
boundaries, time-frequency/visual workflow notes, and BIDS tutorials.

## Phase 8 Findings Fixed

- The final matrix still had six Phase 7 documentation rows marked `docs_gap`
  after PR #172 merged. Phase 8 reclassified those rows to `implemented` and
  added regression coverage that no final closeout row remains `port` or
  `docs_gap`.
- The MATLAB-enabled parity run exposed an inconsistent synthetic HDF5 fixture:
  `xmin=-1`, `pnts=1000`, `srate=500`, and `xmax=1.0`. EEGLAB/EEGPrep timing
  semantics make the final point `0.998`, so Phase 8 corrected the fixture and
  assertion in `tests/test_pop_loadset_h5.py`.
- OC autoreview found that the time-frequency/movie documentation row needed
  explicit user-guide coverage. Phase 8 added supported time-frequency and ERP
  image wrapper guidance to `docs/source/user_guide/preprocessing_pipeline.rst`
  and the EEG movie boundary to `docs/source/user_guide/visual_parity.rst`.
- The first MATLAB parity attempt found a truncated ignored
  `sample_data/EmotionValence.set` download. Phase 8 refreshed that local
  fixture to the S3-reported 203,941,008 bytes before rerunning the suite.

## Runtime Independence

Runtime package code must not read, import from, or shell out to
`src/eegprep/eeglab`. The vendored tree remains a development and parity-test
oracle only.

Phase 8 verified this with `tests/test_runtime_eeglab_independence.py` and a
source scan for direct `src/eegprep/eeglab`, `eegprep.eeglab`, importlib
resource, or package-root path-join dependencies. Development validators and
visual parity tools may read the vendored reference tree.

## Public API, Help, Package Data, and Menu Inventory

Phase 8 kept the public API/menu/help/package evidence in tested surfaces:

- `tests/test_package_exports.py` covers lazy public exports.
- `tests/test_public_api_examples.py` covers documented API examples and
  package-data declarations.
- `tests/test_guifunc_pophelp_chansel.py` covers packaged Markdown help
  resources, `pophelp`, and implemented menu help targets.
- `tests/test_menu_placeholder_inventory.py` covers menu placeholder metadata.
- `tests/test_gui_main_window.py` covers main-window menu/help/session wiring.

## Visual Parity Attachment Inventory

Visual artifacts are attached to phase PR comments rather than committed:

- PR #167, ICLabel/viewprops: `iclabel_pop_prop_extended_dashboard`,
  `pop_icflag_dialog`, `pop_iclabel_dialog`, and `pop_viewprops_dialog`
  side-by-side images.
- PR #169, DIPFIT: `pop_dipfit_settings`, `pop_dipfit_gridsearch`,
  `pop_dipfit_nonlinear`, `pop_dipplot`, `pop_multifit`, `pop_leadfield`,
  `pop_dipfit_loreta`, and `pop_dipfit_headmodel` evidence.
- PR #171, FIRFilt: `pop_kaiserbeta_dialog`, `pop_firwsord_dialog`,
  `pop_firpmord_dialog`, `pop_xfirws_dialog`, and refreshed
  `pop_firpmord_dialog_review_fix` evidence.
- PRs #166, #168, #170, and #172 did not add new GUI dialog layouts requiring
  fresh visual attachments.

## GUI Agent Mixed Workflow QA

Computer Use inspected the live `eegprep-gui --window-menu-bar` window and
confirmed the visible startup state: File and Help enabled, dataset-dependent
Edit/Tools/Plot/Study/Datasets menus disabled, and the EEGLAB-style startup
instructions visible. Computer Use click actions against the Python-hosted
Qt app were rejected by the tool as inactive immediately after state capture,
so Phase 8 continued the requested flow QA with Qt-driven actions against the
real main window.

The mixed workflow QA script exercised startup menus, Help menu opening,
packaged `pophelp` dispatch, dataset storage and GUI refresh, GUI dataset
retrieve followed by console namespace inspection, and a bare console
`pop_reref(EEG, [])` call followed by GUI refresh.

## Accepted Non-Goals

These are explicit final-epic boundaries rather than Phase 8 omissions:

- Full Manopt-backed Riemannian ASR processing remains an optional dependency
  decision; standard ASR and calibration-time Riemannian behavior are supported.
- DIPFIT MRI-derived BEM headmodel creation, AFNI atlas clipping, LORETA source
  analysis, and FieldTrip source-statistics workflows remain explicit backend
  limits.
- Full LIMO model fitting, result computation, and browsing remain external
  backend workflows; EEGPrep owns design-matrix preparation.
- ICLabel `lite` and `beta` network artifacts remain explicit MATLAB/Octave
  engine paths until EEGPrep packages tested standalone assets.
- MATLAB object overloads, developer tests, command-window shims, and third-party
  plugin ecosystems remain consolidated, skipped, or extension-owned rather than
  one-for-one runtime ports.

## Verification Log

Phase 8 verification is run from this branch. Results are updated before the
phase PR is opened.

| Command | Result |
| --- | --- |
| `uv sync --group dev --extra gui --extra console --extra docs --extra torch` | Passed |
| `uv pip install /Applications/MATLAB_R2026a.app/extern/engines/python` | Passed: installed local `matlabengine==26.1` for MATLAB parity verification |
| `uv run --no-sync python -m tools.eeglab_final_parity_matrix --json` | Passed: `ok: true`, 31 rows, 180 expected paths |
| `uv run --no-sync pytest tests/test_eeglab_final_parity_matrix.py tests/test_pop_loadset_h5.py::TestPopLoadsetH5::test_basic_h5_loading --tb=short` | Passed: 13 passed |
| `uv run --no-sync pytest tests/test_runtime_eeglab_independence.py tests/test_guifunc_pophelp_chansel.py tests/test_public_api_examples.py tests/test_package_exports.py tests/test_menu_placeholder_inventory.py` | Passed: 37 passed |
| `uv run --no-sync pytest tests/test_gui_main_window.py` | Passed: 69 passed |
| `uv run --no-sync pytest tests/test_console_workspace.py` | Passed: 86 passed |
| `uv run --no-sync ruff check .` | Passed |
| `uv run --no-sync ruff format --check .` | Passed |
| `uv run --no-sync ty check` | Passed |
| `uv run --no-sync sphinx-build -b html docs/source docs/_build/html` | Passed |
| `uv run --no-sync pytest tests/test_visual_parity.py` | Passed: 26 passed |
| `EEGPREP_SKIP_MATLAB=1 uv run --no-sync pytest -m "not slow" --tb=short` | Passed: 1873 passed, 209 skipped, 12 deselected |
| `uv run --no-sync pytest -m "matlab or octave" --tb=short` | Passed: 342 passed, 21 skipped, 1818 deselected |
| GUI Agent mixed workflow QA script | Passed: startup/menu/help, GUI retrieve to console sync, console bare `pop_reref` to GUI refresh |
| `.agents/skills/oc-autoreview-adapted/scripts/autoreview --mode local --codex-bin /tmp/codex-fast-autoreview` | Passed clean after fixing accepted docs findings |
| `./pre-commit.py --fix` | Passed on the staged Phase 8 files |
