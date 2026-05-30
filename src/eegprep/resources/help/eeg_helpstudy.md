# EEG_HELPSTUDY - Group data (STUDY) functions

EEGPrep exposes STUDY/group-level session surfaces for workflows that are
implemented on this branch and marks later STUDY work explicitly in the menu
inventory.

Implemented STUDY actions:

- `pop_study`: create a STUDY structure from loaded `ALLEEG` datasets.
- `pop_studywizard`: browse for dataset files and create a STUDY.
- `pop_studyerp`: create a simple ERP STUDY.
- `pop_loadstudy`: load an EEGPrep `.study` JSON file.
- `pop_savestudy`: save the current STUDY.
- `pop_chanplot`: plot ERP channel measures from epoched datasets.

Pending STUDY design, precompute, preclustering, clustering, and cluster-edit
actions remain Phase 5 placeholders until those branches land. Their Help
entries should be expanded when the corresponding implementation is merged.

See also: POP_STUDY, POP_LOADSTUDY, POP_SAVESTUDY, POP_CHANPLOT
