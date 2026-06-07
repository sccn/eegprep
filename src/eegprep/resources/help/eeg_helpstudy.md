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
- `pop_importgroupvar`, `pop_listfactors`, and `std_builddesignmat`: manage
  STUDY design variables and deterministic design matrices.
- `std_limodesign`: build LIMO-compatible categorical and continuous design
  matrices from factors and trial metadata.
- `std_pac`, `std_readpac`, and `std_pacplot`: compute, read, and plot
  EEGPrep-owned STUDY PAC caches for channel workflows.
- `std_prepare_neighbors`: create distance-based channel-neighbor structures
  and LIMO-compatible adjacency matrices.
- `std_interp`: interpolate requested missing channels across STUDY datasets.
- `std_checkfiles`, `std_checkdatasession`, `std_uniformfiles`, and
  `std_uniformsetinds`: audit loaded dataset and cached measure consistency.

Component ERP, spectrum, ERSP, and ITC clustering inputs are cached on the
parent `STUDY.cluster[0]` entry by `pop_precomp`.

LIMO result computation and browsing are not silently emulated. `pop_limo`,
`pop_limoresults`, `std_limo`, `std_limoresults`, and `std_readfilelimo`
report that standalone EEGPrep does not run EEGLAB's external LIMO toolbox
workflow. STUDY source workflows such as `std_dipplot` and
`std_dipoleclusters` remain behind the DIPFIT/FieldTrip backend boundary.

See also: POP_STUDY, POP_PRECOMP, POP_CHANPLOT, POP_LOADSTUDY, POP_SAVESTUDY, POP_PRECLUST, POP_CLUST, POP_CLUSTEDIT, STD_PAC
