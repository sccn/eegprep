# pop_selectevent

Select, rename, or delete events and event-related epochs.

Criteria may target event fields such as `type`, `latency`, `duration`, or
custom fields. Event indices are 1-based. Continuous data keeps boundary events
when deleting non-selected events, matching EEGLAB's expectation that boundaries
preserve discontinuity information.

The function returns selected event indices for programmatic calls, and returns
an EEG plus replayable command when `return_com=True`.
