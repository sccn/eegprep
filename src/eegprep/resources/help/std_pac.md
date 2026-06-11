# STD_PAC - STUDY phase-amplitude coupling

`std_pac` computes EEGPrep-owned PAC caches for single EEG datasets or STUDY
channel workflows.

For a single dataset, it returns PAC magnitude arrays, output times,
amplitude-frequency values, and the parameter contract. For a STUDY, it stores
`pacdata`, `pactimes`, `pacfreqs`, and `pacchannels2` under `STUDY.changrp` so
`std_readpac` and `std_pacplot` can read the cache.

STUDY PAC currently supports channel workflows. Component-cluster PAC and
implicit interpolation/ICA-component removal remain explicit boundaries; run
the relevant preprocessing step before computing PAC.

See also: PAC, PAC_CONT, STD_READPAC, STD_PACPLOT
