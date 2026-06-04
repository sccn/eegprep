# pop_template_gain

Scale `EEG.data` by a numeric gain and return an EEGLAB-style history command.

Usage:

```python
EEG, com = pop_template_gain(EEG, 2, return_com=True)
```

GUI cancel returns the original EEG and an empty command. This help file is
packaged inside the extension; it does not depend on an EEGLAB checkout.
