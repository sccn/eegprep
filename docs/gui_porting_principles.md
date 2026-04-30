# GUI Porting Principles

EEGPREP GUI work exists to make EEGLAB users productive in Python without losing the workflows they already trust.

## Foundation

- Preserve EEGLAB familiarity first: labels, field order, defaults, Help/Cancel/OK flow, command history, and menu workflow should match EEGLAB unless there is a documented reason not to.
- Use Python underneath: typed specs, explicit callbacks, optional dependencies, normal Python tests, and clean separation between GUI collection and computation.
- Keep GUI files maintainable across EEGPREP and EEGLAB by tracking the EEGLAB source file, parity scope, and known divergences for every ported dialog.
- Do not line-by-line port arbitrary MATLAB GUI code when it mixes layout, callbacks, validation, and computation. Extract the stable GUI contract instead.
- GUI dependencies are optional. Core `import eegprep` must not import Qt or require a display.

## Data Compatibility

- GUI behavior must respect EEGLAB data semantics: `EEG.data` is channels x timepoints x epochs, events and urevents are distinct, and event `latency` is a 1-based floating sample position.
- Preserve EEGLAB-style tags/control names in GUI specs so MATLAB dialogs and Python dialogs can be compared mechanically.
- Keep computation in non-GUI functions. GUI code should decode user input into the same arguments that CLI/parity tests use.
- Preserve EEGLAB-compatible history commands (`com`) for GUI-backed actions.
- Unknown/plugin metadata must not be dropped as a side effect of GUI use.

## Architecture

- Represent dialogs with renderer-independent specs modeled after EEGLAB `inputgui`: title, geometry, controls, tags, defaults, callbacks, help, source file, and known differences.
- Use PySide6/Qt as the first renderer because it best matches MATLAB/EEGLAB desktop interaction. Keep renderer-specific code isolated.
- Store original MATLAB callback snippets only as traceability metadata. Python callbacks must be explicit, typed, and testable.
- Prefer structural GUI parity tests over brittle pixel snapshots. Screenshot/perceptual tests can be added only for mature visual components.
- Any intentional mismatch from EEGLAB must be documented in the spec and in user/developer docs if user-visible.

## Porting Steps

1. Inspect the EEGLAB source dialog and identify `geometry`, `uilist`, callbacks, result decoding, validation, core computation, and `com` generation.
2. Implement or verify the non-GUI computation path first.
3. Create a GUI spec that mirrors EEGLAB layout, labels, defaults, tags, and help text.
4. Implement Python callbacks only for behavior needed by the dialog.
5. Decode GUI results into the same arguments used by the CLI path.
6. Add tests for computation, spec structure, GUI decoding, callback behavior, and MATLAB parity where available.
7. Update Sphinx docs for public GUI behavior in the same pass.

## Testing Bar

- Every GUI-backed function needs non-GUI tests; GUI tests are not a substitute for computation tests.
- Add MATLAB parity tests for the CLI/core behavior when the corresponding EEGLAB function exists.
- Add spec tests for title, source file, geometry, labels, tags, defaults, and known differences.
- Add headless Qt tests only where renderer behavior matters; skip cleanly when `eegprep[gui]` is not installed.
- Tests should make future agent changes fail loudly when they drift from EEGLAB workflow or data semantics.
