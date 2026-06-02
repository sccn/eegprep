# POP_PRECOMP - Precompute STUDY measures

`pop_precomp` computes channel or component STUDY measures and stores them in
the STUDY dictionary using EEGLAB-compatible field names.

```python
STUDY, ALLEEG, com = pop_precomp(
    STUDY,
    ALLEEG,
    "channels",
    erp="on",
    spec="on",
    return_com=True,
)
```

Channel measures are stored in `STUDY.changrp`. Component measures are stored
on the parent `STUDY.cluster[0]` entry until clustering support lands. Cached
fields include `erpdata`, `specdata`, `erspdata`, `itcdata`, and their matching
time/frequency axis fields.

EEGPrep stores measures in the `.study` JSON structure instead of EEGLAB
`.dat*` and `.ica*` sidecar files. This keeps runtime behavior standalone and
replayable from `eegprep-console`.

The selected STUDY design is recorded in measure metadata, but Phase 5b cached
arrays are dataset-level averages and are not split into EEGLAB design cells.

See also: POP_CHANPLOT, STD_PRECOMP, STD_READDATA
