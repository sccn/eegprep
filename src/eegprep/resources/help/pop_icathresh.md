# pop_icathresh

`pop_icathresh(EEG, threshval, rejmethod, rejvalue)` updates legacy ICA
threshold fields and computes `EEG["reject"]["gcompreject"]`.

`threshval` contains entropy, activity-kurtosis, and map-kurtosis thresholds.
`rejmethod="current"` applies those thresholds to the current dataset stats.
`rejmethod="percent"` derives thresholds from the highest percentage of
components. The old interactive threshold-tuning window is not recreated.
