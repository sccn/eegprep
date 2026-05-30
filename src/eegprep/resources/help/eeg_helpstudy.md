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
- `pop_precomp`: precompute channel or component ERP, spectrum, ERSP, and ITC measures.
- `pop_chanplot`: plot cached channel or parent-cluster component measures.

Pending preclustering, clustering, and cluster-edit actions remain Phase 5c
placeholders until that branch lands. Component measures are currently stored on
the parent cluster for later Phase 5c consumption.

See also: POP_STUDY, POP_PRECOMP, POP_CHANPLOT, POP_LOADSTUDY, POP_SAVESTUDY
