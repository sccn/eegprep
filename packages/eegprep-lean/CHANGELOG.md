# Changelog

## Unreleased

- `load_recording` reads the electrode positions a NEMAR store declares into `Window.electrodes`, with one more request for the store's root `zarr.json`; `read_electrode_attributes` makes the same read for callers that read windows themselves. `read_window` is unchanged.
- `to_eeg` fills `chanlocs` from `window.electrodes`: `X`, `Y`, `Z` in millimeters in EEGLAB's ALS frame, `sph_theta`, `sph_phi`, `sph_radius`, `theta` and `radius`, by exact label, with the values `pop_load_frombids` gives from the same `electrodes.tsv` and coordsystem file, except that `CTF`, `4DBti` and `KitYokogawa` stay ALS as the BIDS appendix has them. A unit or coordinate system outside the BIDS appendix fills no positions and is never guessed. `etc["nemar"]["locations"]` records what the store stated, how many channels were located and, when none was, the reason. A store that declares no positions gives labels alone, as before.
- `etc["nemar"]["reader"]` names `recording-v2`, so a record made by a reader that fills locations differs from one made by a reader that never looked for them.

## 0.5.0

First version number shared with eegprep. The previous package version was 0.1.0.dev3.

- `eegprep_lean.help()` prints each public function with a one-line summary from its docstring, the extra that supplies it, what is not available yet, and a short example. The text is read from source, so calling it does not import an extra.
- `dir(eegprep_lean)` lists the lazy public names, including names that live behind an extra, without importing them.
