POP_INTERP - interpolate data channels.

Usage:

    EEGOUT = pop_interp(EEG, badchans, method, t_range)

Calling `pop_interp(EEG)` opens the interactive channel interpolation dialog.

Inputs:

- `EEG`: input EEG dataset.
- `badchans`: 1-based channel indices (channel 1 is the first channel), channel labels, or channel-location dictionaries.
- `method`: interpolation method. Defaults to `"spherical"`.
- `t_range`: inclusive `[start, end]` time range in seconds. Samples on `EEG.times` inside the range are interpolated; samples outside it stay unchanged, including epoched data. Omit it, or leave the dialog field empty, to interpolate the whole recording.

Pinned EEGLAB `pop_interp.m` line 61 sets `t_range = ''`, so a scripted fourth argument is discarded and the whole recording is interpolated. EEGPrep keeps the range. The GUI history in that pinned file also omits the range.

Interpolation methods:

- `spherical`: spherical interpolation.
- `sphericalKang`: Kang et al. spherical interpolation for epoched data.
- `invdist` or `v4`: inverse-distance scalp interpolation.
- `spacetime`: space-time interpolation.

Graphic interface:

- "Select from removed channels" uses channels stored as removed/non-data channels.
- "Select from data channels" selects currently present data channels.
- "Use specific channels of other dataset" selects missing channels from another dataset.
- "Use all channels from other dataset" uses all missing channels from another dataset.
- "Interpolation method" selects the interpolation method.
- "Time range [min max] (s)" limits continuous-data interpolation to a time range.

Output:

- `EEGOUT`: EEG dataset with selected channels interpolated.

See also: EEG_INTERP
