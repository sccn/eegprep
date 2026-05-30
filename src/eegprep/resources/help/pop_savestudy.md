# pop_savestudy

Save the current STUDY as an EEGPrep-owned `.study` JSON file.

Example:

```python
STUDY, LASTCOM = pop_savestudy(STUDY, EEG, filename="demo.study", filepath="/data")
```

The saved file contains STUDY metadata, dataset membership, designs, and
consistency diagnostics. It does not save measure precompute arrays or cluster
measure data in Phase 5a.
