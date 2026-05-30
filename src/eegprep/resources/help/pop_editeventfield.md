# pop_editeventfield

Add, remove, rename, or type-convert fields in `EEG.event`.

User-facing event indices are 1-based. Field values may be scalar or a list
matching the selected event indices. Latency values supplied through the
`latency` field are interpreted in seconds by default and converted to EEGLAB
sample latencies.

The helper updates matching `EEG.urevent` entries when an event has a valid
`urevent` pointer.
