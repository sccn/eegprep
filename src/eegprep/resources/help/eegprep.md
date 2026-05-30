# EEGPrep

EEGPrep is a Python implementation of core EEGLAB preprocessing workflows,
menu organization, history commands, EEG dictionary fields, and GUI patterns.

Use the main window for EEGLAB-style menu workflows, or launch
`eegprep-console` when you want the GUI and an interactive Python workspace to
share one session. The shared workspace exposes `EEG`, `ALLEEG`, `CURRENTSET`,
`LASTCOM`, `ALLCOM`, `STUDY`, and `CURRENTSTUDY`.

Runtime help is provided by Markdown files packaged with EEGPrep under
`eegprep.resources.help`. The vendored EEGLAB source tree is a development
reference only and is not used for installed help lookup.

See also: EEG_HELPHELP, EEG_HELPMENU
