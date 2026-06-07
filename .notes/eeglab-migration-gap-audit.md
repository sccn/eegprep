# EEGPrep Remaining EEGLAB Migration Gap Audit

Audit date: 2026-06-07
Branch audited: `feature/eeglab-core-parity-completion` / PR #152
Primary reference: `docs/parity/eeglab_core_parity_matrix.json`

## Summary

This file is the current committed audit of what still remains after the
remaining-core-parity epic. It supersedes the earlier pre-epic audit that listed
items 1-7 as open work.

PR #152 completes or explicitly classifies the first seven migration categories:

1. Long-tail helper coverage
2. Missing or legacy `pop_*` entry points
3. Unsupported options in implemented user-facing functions
4. STUDY and group-level depth
5. Statistics package
6. Time-frequency internals
7. File-format and channel-location long tail

The machine-readable matrix now covers all in-scope EEGLAB functions for those
categories:

- `uv run --no-sync python -m tools.eeglab_parity_matrix --json`
- 608 rows cover 608 expected in-scope EEGLAB functions.
- Status counts: 238 `implemented`, 99 `consolidated`, 104 `stale_skip`,
  121 `matlab_runtime_skip`, and 46 `external_dependency_skip`.
- There are no remaining `port`, `partial`, or unclassified rows for the
  categories covered by PR #152.

## Completed In PR #152

The epic converted the original gap audit into an enforceable parity matrix and
then closed the useful core gaps through phase PRs:

- Phase 1: parity matrix, staleness triage, and standalone runtime contracts.
- Phase 2: file-format and channel-location long-tail helpers.
- Phase 3: EEGLAB-style statistics helpers.
- Phase 4: time-frequency internals and legacy time-frequency entry points.
- Phase 5: unsupported options in already user-facing functions.
- Phase 6: deeper STUDY/group-level helpers.
- Phase 7: remaining audit-approved helpers and `pop_*` wrappers.
- Phase 8: closeout docs, help resources, QA, and matrix validation.

The follow-up issues for time-warped `newtimef`, exact `correct_mc` random-symbol
distribution fitting, PAC classification, and STUDY long-tail helper depth are
also resolved in the same PR stack.

## What Still Remains

The remaining items are no longer generic “missing same-name files.” They are
explicitly outside the current core parity scope, intentionally skipped because
they are stale/MATLAB-only, or dependent on external runtimes/toolboxes. Future
work should start from these product areas, not from a blind same-name port.

### 1. MATLAB Runtime And Figure-Helper Skips

Many EEGLAB files are MATLAB command-window, path, deployed-app, figure-editing,
or low-level GUI compatibility helpers. EEGPrep should not port these
one-for-one unless a real EEGPrep user workflow needs them.

Examples include:

- `abouteeglab`, `eeg_cache`, `eeg_eval`, `eeg_global`, `eeglab_execmenu`,
  `eeglab_new`, and `eeglab_options`
- EEGLAB help menu wrappers such as `eeg_helpadmin`, `eeg_helpgui`,
  `eeg_helppop`, `eeg_helpstudy`, and `eeg_helptimefreq`
- MATLAB dialog or path wrappers such as `questdlg2`, `warndlg2`,
  `uigetfile2`, `uiputfile2`, and `removepath`
- MATLAB plotting utilities such as `axcopy`, `copyaxis`, `plotcurve`,
  `plotdata`, `ploterp`, `plotmesh`, `plotsphere`, `textsc`, and `sbplot`

Recommended handling: keep these rows classified in the parity matrix. Port only
when a concrete EEGPrep GUI/API path requires the behavior, and implement it as
native Python/Qt behavior rather than a MATLAB-runtime imitation.

### 2. External Dependency And Toolbox-Backed Workflows

Some EEGLAB functions depend on external toolboxes, MATLAB-specific runtimes, or
large plugin ecosystems. These are intentionally not silently faked in EEGPrep.

Examples include:

- LIMO workflows: `pop_limo`, `pop_limoresults`, `std_limo`,
  `std_limodesign`, `std_limoresults`, and `std_readfilelimo`
- FieldTrip/neighbour/DIPFIT-dependent STUDY helpers such as
  `std_prepare_neighbors`, `std_interp`, `std_dipplot`, and
  `std_dipoleclusters`
- Legacy ICA backends such as `binica`, `jader`, `sobi`, `acsobiro`,
  `fastif`, and old `runica_ml*` variants
- Direct legacy BIOSIG/EGI import wrappers where a supported modern import path
  already exists or an external backend is required
- PAC compute/plot/cache helpers such as `pac`, `pac_cont`, `std_pac`,
  `std_pacplot`, and `std_readpac`

Recommended handling: create separate product epics only when EEGPrep can offer
a tested standalone implementation or a clearly documented optional dependency.
Do not add placeholder math or fake cache files just to match names.

### 3. MATLAB Object And Memory-Mapped Infrastructure

EEGLAB has MATLAB class-style folders that are not one-for-one Python concepts:

- `functions/@eegobj`
- `functions/@memmapdata`
- `functions/@mmo`

This matters for full MATLAB `storedisk` and memory-mapped dataset behavior.
EEGPrep currently uses explicit Python `EEGPrepSession`, `ALLEEG`, and dataset
storage semantics instead.

Recommended handling: treat this as a future storage/performance design epic,
not a MATLAB class port. If EEGPrep needs large-dataset lazy loading, design a
Python-native data backend with tests for GUI/console synchronization,
`pop_newset`, save/load behavior, and STUDY workflows.

### 4. Bundled Plugin Depth

The bundled in-repo plugin surfaces are represented, but not all plugin internals
are complete one-for-one ports.

Remaining plugin-depth areas:

- `clean_rawdata`: exact Riemannian ASR processing parity, full
  `vis_artifacts` behavior, and Manopt-backed MATLAB helper depth.
- `firfilt`: lower-level helper coverage such as detailed reports, inverse
  order helpers, minimum-phase helpers, frequency-response plotting, and order
  calculator dialogs.
- `ICLabel` / viewprops: alternate network artifacts, exact MATLAB helper
  parity for `eeg_icalabelstat`, and any viewprops helper behavior still
  replaced by EEGPrep-native Qt/Python paths.
- `DIPFIT`: lower-level grid/nonlinear/reject/dipplot helpers, manual/batch
  dialogs, atlas conversion helpers, and private transform utilities.

Recommended handling: split these into plugin-specific epics. Each plugin should
define what “standalone EEGPrep parity” means, which external assets are allowed,
and which EEGLAB MATLAB internals should stay unported.

### 5. External Plugin Ecosystem

EEGPrep now has extension infrastructure, but the broad EEGLAB external plugin
ecosystem is not part of core EEGPrep.

Examples:

- ERPLAB
- LIMO as a full external statistics workflow
- SIFT
- NFT
- MFF/importer plugin ecosystems
- Lab-specific processing plugins

Recommended handling: external plugin work should use EEGPrep’s extension
contracts, catalog/trust model, documentation, and extension-development skill.
Do not merge third-party plugin behavior into core EEGPrep unless it becomes a
maintained bundled plugin with tests, docs, packaging, and GUI/console support.

### 6. Docs, Tutorials, And User Education

EEGPrep has EEGPrep-owned help resources and user-facing docs for the new epic
work, but EEGLAB’s full tutorial corpus is larger than the ported docs.

Remaining useful docs work:

- End-to-end tutorials comparable to EEGLAB’s practical workflows.
- More task-oriented examples for STUDY, EEGBrowser, extension authoring, file
  I/O, time-frequency, and statistics.
- User-facing migration notes for EEGLAB users moving MATLAB commands to
  EEGPrep Python/console workflows.
- Curated visual parity evidence index for major GUI surfaces.

Recommended handling: keep help Markdown next to user-facing `pop_*` features,
but plan tutorials as product documentation rather than generated API dumps.

### 7. Ongoing Parity Matrix Maintenance

The parity matrix is now the source of truth for the first seven audit
categories. Future feature work should update it whenever an EEGLAB-facing
function is added, consolidated, or intentionally skipped.

Required behavior:

- `tools/eeglab_parity_matrix.py` must stay green.
- Runtime package code must not depend on `src/eegprep/eeglab`.
- New GUI features need visual parity evidence.
- New console/API features need replayable history and `eegprep-console`
  synchronization tests where relevant.
- New numerical behavior should have MATLAB parity tests when deterministic and
  feasible.

## Not Recommended

Avoid these patterns in future migration work:

- Blind one-file-to-one-file ports of stale MATLAB helpers.
- Fake implementations that return plausible shapes but do not perform the
  EEGLAB workflow.
- Runtime fallbacks that read the vendored EEGLAB checkout.
- Adding external-toolbox behavior without an explicit optional dependency,
  install docs, tests, and clear user-facing failure mode.
- Implementing old MATLAB GUI shims when the right EEGPrep answer is a native
  Qt or Python API surface.

## Next Planning Step

Epic #157 now turns the remaining product areas above into a scoped issue tree.
Its Phase 1 contract lives in:

- `.notes/eeglab-final-parity-audit.md`
- `docs/parity/eeglab_final_parity_matrix.json`
- `tools/eeglab_final_parity_matrix.py`

The final matrix assigns concrete phase ownership for bundled plugin depth,
object/storage semantics, optional-toolbox workflows, and docs/tutorial gaps.
Future phase agents should update that matrix instead of reclassifying product
scope from this prose audit.

Before the final epic started, the strongest candidates were:

1. Bundled plugin depth, split by plugin family.
2. Large-dataset storage and memory mapping semantics.
3. External dependency workflows such as LIMO or advanced DIPFIT.
4. User documentation/tutorial parity for completed core workflows.

Before changing core parity rows, run:

```bash
uv run --no-sync python -m tools.eeglab_parity_matrix --json
```

Before changing final epic rows, run:

```bash
uv run --no-sync python -m tools.eeglab_final_parity_matrix --json
```

Then decide whether the work changes the existing matrix rows or belongs to a
new matrix/category outside PR #152’s original scope.
