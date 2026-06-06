# pop_writelocs

Write EEGPrep channel locations to a text location file.

`pop_writelocs(EEG["chanlocs"], filename)` is the EEGLAB-style wrapper around
`writelocs`. It writes common EEGLAB channel-location formats, including
`chanedit`, `.locs`, `.sfp`, `.xyz`, and `.ced`-style files.

Use options such as `filetype`, `format`, `header`, `customheader`, and
`elecind` to choose columns or write a subset of channels.

Example:

```python
com = pop_writelocs(EEG["chanlocs"], "subject01_locations.ced")
```
