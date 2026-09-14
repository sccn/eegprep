# pop_chancoresp

Pair corresponding channels between two channel-location sets.

`pop_chancoresp(chans1, chans2)` compares labels and returns two 1-based index
lists. The default `autoselect="all"` mode pairs labels that match in both
inputs. Use `autoselect="fiducials"` to pair only common fiducial aliases such
as nasion, LPA, and RPA. Netstation's `FidT10` and `FidT9` labels correspond to
LPA and RPA, respectively, matching EEGLAB.

Example:

```python
chanlist1, chanlist2, com = pop_chancoresp(
    EEG["chanlocs"],
    template_locs,
    "autoselect",
    "all",
    return_com=True,
)
```
