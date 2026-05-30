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
- `pop_preclust`: build ICA component preclustering arrays.
- `pop_clust`: cluster preclustered ICA components.
- `pop_clustedit`: edit clusters and plot cluster summaries.

Pending STUDY precompute and full measure plotting actions remain Phase 5
placeholders until those branches land. Component ERP, spectrum, ERSP, and ITC
clustering inputs are read from the documented Phase 5b component-measure
contract.

See also: POP_STUDY, POP_LOADSTUDY, POP_SAVESTUDY, POP_CHANPLOT, POP_PRECLUST, POP_CLUST, POP_CLUSTEDIT
