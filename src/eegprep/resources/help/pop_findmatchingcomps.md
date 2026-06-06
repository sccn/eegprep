# pop_findmatchingcomps

`pop_findmatchingcomps(EEG, "matchcomps", maps)` finds ICA components whose
scalp maps have high absolute correlation with provided component maps.

Useful options:

- `corrthresh`: minimum absolute map correlation, default `0.92`.
- `dataset`: another EEG dictionary whose rejected components should be
  matched.
- `matchcomps`: a channel-by-component matrix of maps to match.
- `rejflag`: set matching components in `EEG["reject"]["gcompreject"]`.
- `nomatcherror`: set to `"on"` to fail when not every input map matches.
