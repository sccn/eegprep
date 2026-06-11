# STD_READPAC - Read STUDY PAC caches

`std_readpac` returns explicit EEGPrep-owned `pacdata` caches from
`STUDY.changrp` or `STUDY.cluster`.

The cache must include `pacdata`, `pactimes`, and `pacfreqs`. EEGPrep does not
interpret external PAC or LIMO sidecar files as native cache data. Use
`timerange`, `freqrange`, `channels1`, and `channels2` to slice cached channel
PAC results.

See also: STD_PAC, STD_PACPLOT
