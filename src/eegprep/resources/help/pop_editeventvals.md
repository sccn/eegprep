# pop_editeventvals

Edit individual event values.

Supported command-line actions include `changefield`, `delete`, `insert`,
`append`, `add`, and `sort`. Event indices shown to users are 1-based, matching
EEGLAB. Continuous event latencies are displayed in seconds and stored as
EEGLAB sample latencies.

The current GUI keeps the EEGLAB-style field layout. Type an event number and
press Enter, or use the arrows, to show another event. The number stays inside
1 through the event count, matching the clamp in EEGLAB `pop_editeventvals.m`
lines 153-160. An edit is kept on the event you leave, and Ok records every
edit.
