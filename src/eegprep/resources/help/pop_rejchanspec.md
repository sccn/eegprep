# pop_rejchanspec

`pop_rejchanspec(EEG, "key", value, ...)` rejects channels whose average
spectral power is an outlier in one or more frequency ranges.

Useful options:

- `elec`: 1-based channel indices to inspect.
- `freqlims`: `[low high]` frequency range, or multiple rows.
- `stdthresh`: standard-deviation threshold around the median spectrum value.
- `absthresh`: absolute lower/upper spectrum thresholds.
- `averef`: `"on"` to average-reference before measuring spectra.
- `indexonly`: `"on"` to return rejected indices without removing channels.
