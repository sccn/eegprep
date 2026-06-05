POP_TIMEF - Legacy time-frequency wrapper.

`pop_timef` is provided for EEGLAB command compatibility. In EEGPrep it runs the
same standalone numerical and plotting implementation as `pop_newtimef`, while
returning a replayable `pop_timef(...)` history command.

Example:

```python
result, com = pop_timef(EEG, 1, 1, [-100, 200], [3, 0.5], return_com=True)
```
