# POP_FILEIO_CNT - Import Neuroscan CNT recordings

The CNT File-IO menu action imports Neuroscan `.cnt` files through the
standalone `pop_loadcnt` reader. `pop_fileio` selects that reader automatically.

Usage:

```python
EEG = pop_fileio("recording.cnt")
```

For explicit sample ranges, count scaling, keyboard events, or disk-backed
loading, call `pop_loadcnt` directly.

Use this action when the dataset is a CNT recording. For text or array data,
use `pop_importdata`.

See also: POP_FILEIO
