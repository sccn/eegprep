# Import an EGI MFF recording

`pop_mffimport` reads an EGI `.mff` folder. Segmented recordings stay as epochs. Continuous recordings stay continuous. Synthetic stimulus channels (`stm+`, `resp`, `fbk+`, `LRND`) are not kept as EEG channels.

Choose the event field to use as the event type. The default is `code`. Cancel closes the dialog and does not change the current dataset.
