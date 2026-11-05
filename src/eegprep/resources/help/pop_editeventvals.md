# pop_editeventvals

Edit individual event values.

Supported command-line actions include `changefield`, `delete`, `insert`,
`append`, `add`, and `sort`. Event indices shown to users are 1-based, matching
EEGLAB. Continuous event latencies are displayed in seconds and stored as
EEGLAB sample latencies.

The current GUI edits one event per submission while keeping the EEGLAB-style
field/value layout. Type an event number and press Enter to show that event.
The number is kept inside 1 through the event count, matching the clamp in
EEGLAB `pop_editeventvals.m` lines 159-160. Ok writes the values on screen to
the event those fields belong to. Change the event number before editing a
field; an edit still showing the previous event is not copied onto another
event.
