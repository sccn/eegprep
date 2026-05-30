# EEG_HELPHELP - How to use EEGPrep help

EEGPrep help follows the EEGLAB convention: menu actions and dialogs point to
the function that implements the workflow. For interactive functions, the Help
button opens a packaged Markdown resource through `pophelp`.

Many user-facing functions support two paths:

- Calling the `pop_*` function with only an EEG or STUDY object opens an
  interactive dialog.
- Passing explicit arguments runs the same workflow directly from Python.

When a function supports `return_com=True`, it returns an EEGLAB-style history
command. The GUI and `eegprep-console` append these commands to the shared
session history so workflows can be repeated from scripts.

Missing packaged help resources are treated as errors. Add new Help-button
content under `src/eegprep/resources/help/<function_name>.md`.

See also: POPHELP, EEG_HELPMENU
