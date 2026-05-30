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
- `pop_preclust`: build ICA component preclustering arrays.
- `pop_clust`: cluster preclustered ICA components.
- `pop_clustedit`: edit clusters and plot cluster summaries.

Component ERP, spectrum, ERSP, and ITC clustering inputs are cached on the
parent `STUDY.cluster[0]` entry by `pop_precomp`.

See also: POP_STUDY, POP_PRECOMP, POP_CHANPLOT, POP_LOADSTUDY, POP_SAVESTUDY, POP_PRECLUST, POP_CLUST, POP_CLUSTEDIT
