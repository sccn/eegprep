# eegplot

`eegplot` opens EEGPrep's EEGLAB-style scrolling browser for continuous,
epoched, component, spectral, or overlay data. It accepts channel-major arrays
with shape `channels x samples` or `channels x samples x trials`, and EEG
dictionaries with EEGLAB fields such as `data`, `srate`, `chanlocs`, and
`event`.

Core options:

- `srate`: sampling rate in Hz. EEG dictionaries default to `EEG["srate"]`;
  arrays default to 256 Hz.
- `spacing`: amplitude range per channel. A value of `0` or omission uses a
  robust standard-deviation estimate from the first 1000 displayed samples.
- `limits`: epoch time limits in milliseconds, used for labels.
- `winlength`: visible duration in seconds for continuous data, or epochs for
  epoched data. For spectral data, this is a visible frequency span in Hz.
- `time`: visible-window start time. Epoched inputs use EEGLAB-style one-based
  epoch display while the internal model stores zero-based epoch offsets.
- `dispchans`: number of visible channels.
- `title` and `plottitle`: window and plot titles.
- `xgrid`, `ygrid`, `submean`, and `scale`: `"on"` or `"off"` toggles.
- `data2`: overlay data with the same normalized shape as the primary data.
- `winrej`: rejection-mark rows `[start end R G B channel_mask...]` in
  EEGLAB eegplot frame coordinates.
- `events`: EEGLAB event dictionaries. Event `latency` values are interpreted
  as one-based sample latencies.
- `eloc_file`: channel-location structures or channel numbers for labels.
- `color`: `"off"`, `"on"`, or a sequence of Qt-compatible colors.
- `freqs` and `freqlimits`: frequency-bin values and `[min max]` selection
  bounds for spectral browser input. EEGPrep keeps the selected frequency axis
  in Hz for display; internally, `time` is an offset within that selected range.

Phase 1 provides the browser model and non-mutating PySide6/pyqtgraph
rendering foundation. Interactive rejection updates are reserved for later
EEGBrowser phases.
