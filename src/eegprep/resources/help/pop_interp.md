POP_INTERP - interpolate data channels.

Usage:

    EEGOUT = pop_interp(EEG, badchans, method, t_range)

Calling `pop_interp(EEG)` opens the interactive channel interpolation dialog.

Inputs:

- `EEG`: input EEG dataset.
- `badchans`: channel indices or channel-location dictionaries for channels to interpolate.
- `method`: interpolation method. Defaults to `"spherical"`.
- `t_range`: two-element time range in seconds for continuous-data interpolation.

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
