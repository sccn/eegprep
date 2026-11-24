# Import an EGI MFF recording

`pop_mffimport` reads an EGI `.mff` folder. Segmented recordings stay as epochs. Continuous recordings stay continuous. Synthetic stimulus channels (`stm+`, `resp`, `fbk+`, `LRND`) are not kept as EEG channels.

Choose the event field to use as the event type. The default is `code`. Cancel closes the dialog and does not change the current dataset.

Events that share a latency stay in file order. Each track event includes `begintime`, `classid` (`EVNT` when the file omits it), `relativebegintime`, `mffkeys` (`[cel#:1, ...]`), `mffkeysbackup`, `status`, and `urevent`. A clock latency is the exact microsecond difference between the event time and the recording time, times the sampling rate. EEGLAB's `mff_decodetime` reports a different sample count; that difference is described in the migration notes.
