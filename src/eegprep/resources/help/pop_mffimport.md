# Import an EGI MFF recording

`pop_mffimport` reads an EGI `.mff` folder. Segmented recordings stay as epochs. Continuous recordings stay continuous. Synthetic stimulus channels (`stm+`, `resp`, `fbk+`, `LRND`) are not kept as EEG channels.

Choose the event field to use as the event type. The default is `code`. Cancel closes the dialog and does not change the current dataset.

Events that share a latency stay in file order. Each track event includes `begintime`, `classid`, `relativebegintime`, `mffkeysbackup`, and `urevent`. Clock latencies use the `mff_decodetime` day-count difference times `86400 * srate`.
