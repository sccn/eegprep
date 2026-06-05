# pop_readlocs

Read an electrode or channel-location file into an EEGPrep `chanlocs` list.

`pop_readlocs(filename)` is the EEGLAB-style wrapper around `readlocs`. It
supports common channel-location text formats such as `.locs`, `.ced`, `.sfp`,
`.xyz`, `.sph`, `.elp`, `.elc`, and tab-separated files, plus packaged montage
resources when they are supplied explicitly.

Use `return_com=True` to get the replayable console command together with the
loaded channel locations.

Example:

```python
chanlocs, com = pop_readlocs("standard-10-5-cap385.elp", return_com=True)
EEG["chanlocs"] = chanlocs
```
