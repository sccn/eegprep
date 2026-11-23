# Changelog

## Unreleased

- `load_recording` reads the electrode positions a NEMAR store declares into `Window.electrodes`, with one more request for the store's root `zarr.json`; `read_electrode_attributes` makes the same read for callers that read windows themselves. `read_window` is unchanged.
- `to_eeg` fills `chanlocs` from `window.electrodes`: `X`, `Y`, `Z` in millimeters in EEGLAB's ALS frame, `sph_theta`, `sph_phi`, `sph_radius`, `theta` and `radius`, by exact label, with the values `pop_load_frombids` gives from the same `electrodes.tsv` and coordsystem file, except that `CTF`, `4DBti` and `KitYokogawa` stay ALS as the BIDS appendix has them. The import also reads a coordsystem whose `EEGCoordinateSystemDescription` is `EEGLAB` as ALS whatever `EEGCoordinateSystem` says; a store carries no description, so a RAS system with that description is turned by 90 degrees here. The converter writes no such description today, and reading it is a follow-up if it ever does. A unit or coordinate system outside the BIDS appendix fills no positions and is never guessed. `etc["nemar"]["locations"]` records what the store stated, how many channels were located and, when none was, the reason. A store that declares no positions gives labels alone, as before.
- A store root's electrode positions are read as NEMAR's locations digest defines them.
  JSON integers are read as binary64, so an integer too large for a float (one of 400 digits, say) leaves its channel without a position instead of failing the read, and an integer literal of more than 4300 digits no longer raises `ValueError`.
  Every zero is read as `+0.0` whatever its spelling (`-0`, `-0.0`), so two stores that differ only in a signed zero give the same `chanlocs`.
  `pop_load_frombids` keeps the sign of a zero that a float column of an `electrodes.tsv` spells with a minus sign, so for such a file `Y` and `theta` can differ in sign (`theta` 180 or -180) between the two routes.
  A store whose `electrode_positions` is not an object raises `ElectrodePositionsNotAnObject`, an `IndexError_`, which names the refusal.
- `read_window` and `load_recording` set `Window.types` from each channel's `channel_type` in the group metadata, and `to_eeg` writes it to `chanlocs[i]["type"]`, so selecting channels by type (for example `pop_select(EEG, chantype=["EEG"])`) leaves out an ECG or EOG channel stored in the EEG group. A channel with no `channel_type`, or one that is blank, `n/a` or `nan`, is `EEG`: EEGLAB counts an empty or `n/a` type with the EEG channels and does not guess from the label, and a missing type is what MNE's EEGLAB reader calls `eeg`. One untyped channel no longer drops the types of the rest. `to_mne_raw` maps the declared types with the set `mne.create_info` accepts (`EOG` to `eog`, `ECG` to `ecg`, `EMG` to `emg`, `MISC` to `misc`, `STIM` to `stim`). A declared type that set does not contain, such as `TRIG` or `EKG`, becomes `misc`. `ch_types` still overrides that, and a value or a count MNE will not take raises `ValueError` from `to_mne_raw`.
- `etc["nemar"]["reader"]` names `recording-v2`, so a record made by a reader that fills locations differs from one made by a reader that never looked for them.

## 0.5.0

First version number shared with eegprep. The previous package version was 0.1.0.dev3.

- `eegprep_lean.help()` prints each public function with a one-line summary from its docstring, the extra that supplies it, what is not available yet, and a short example. The text is read from source, so calling it does not import an extra.
- `dir(eegprep_lean)` lists the lazy public names, including names that live behind an extra, without importing them.
