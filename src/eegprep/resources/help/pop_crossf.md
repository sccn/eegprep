POP_CROSSF - Legacy cross-coherence wrapper.

`pop_crossf` is provided for EEGLAB command compatibility. In EEGPrep it runs
the same standalone numerical and plotting implementation as `pop_newcrossf`,
while returning a replayable `pop_crossf(...)` history command.

Example:

```python
result, com = pop_crossf(EEG, 1, 1, 2, [-100, 200], [3, 0.5], return_com=True)
```
