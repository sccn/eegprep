# EEG_HELPGUI - Graphic interface builder functions

EEGPrep GUI helpers provide renderer-independent dialog specifications and Qt
rendering for EEGLAB-style `inputgui` workflows.

Important surfaces:

- `DialogSpec`, `ControlSpec`, and `CallbackSpec`: declarative dialog specs.
- `inputgui`: render a dialog spec and return user-entered values.
- `pophelp`: open packaged EEGPrep help resources.
- `listdlg2`: EEGLAB-like list selection dialogs.
- Main-window menu specs and action dispatchers under `functions.guifunc`.

Dialog specs should keep EEGLAB labels, order, tags, and obvious layout
hierarchy where possible. Toolkit-specific sizing and styling belongs in the
renderer.

See also: POPHELP, INPUTGUI
