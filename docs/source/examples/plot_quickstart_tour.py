"""
Quickstart Tour: Load, Inspect Events, Scroll
=============================================

This example mirrors the four steps of the EEGLAB Quickstart guide on the
checked-in tutorial dataset: load the sample dataset, read the summary that the
main window shows, explore event values, read the dataset comments, and open the
scrolling channel browser. Everything runs headless.
"""

# %%
# Load the sample dataset (File > Load existing dataset).

from pathlib import Path

from eegprep import eeg_checkset, eegplot, pop_comments, pop_editeventvals, pop_loadset

REPO_ROOT = Path(__file__).resolve().parents[3]
dataset = REPO_ROOT / "sample_data" / "eeglab_data.set"

EEG = pop_loadset(dataset)
EEG = eeg_checkset(EEG)

print("setname:", EEG["setname"])
print("channels:", EEG["nbchan"], "frames/epoch:", EEG["pnts"], "epochs:", EEG["trials"])
print("srate:", EEG["srate"], "Hz  epoch range:", (EEG["xmin"], EEG["xmax"]), "s")
print("data shape:", EEG["data"].shape)
print("events:", len(EEG["event"]))

# %%
# Exploring event values (Edit > Event values).
#
# Event indices are 1-based, matching the dialog and EEGLAB.

event_fields = sorted(EEG["event"][0].keys())
print("event fields:", event_fields)
for index in (1, 2, 3):
    event = EEG["event"][index - 1]
    print(index, {field: event[field] for field in event_fields})

print("latency is 1-based samples; urevent is a 0-based index into EEG['urevent']")

types = sorted({str(event["type"]) for event in EEG["event"]})
print("event types:", types)
print("counts:", {t: sum(str(e["type"]) == t for e in EEG["event"]) for t in types})

# %%
# The dialog's edits are available as command-line actions. Sorting by latency
# is the same call the dialog records in the history.

EEG, com = pop_editeventvals(EEG, "sort", ["latency", 0], return_com=True)
print(com)

# %%
# About this dataset (Edit > About this dataset).

EEG, comments_com = pop_comments(EEG, "", "Quickstart tour of the tutorial dataset.", return_com=True)
print(comments_com)
print("comments:", EEG["comments"])

# %%
# Scrolling through the data (Plot > Channel data (scroll)).
#
# ``show=False`` builds the browser model without opening a Qt window, so the
# same call is usable in scripts and tests.

browser = eegplot(EEG, winlength=5, dispchans=16, title="Scroll channel activities", show=False)
print("browser type:", type(browser).__name__)
print("window length (s):", browser.state.winlength, "displayed channels:", browser.state.dispchans)
print("browsed channels:", browser.data.n_channels, "samples:", browser.data.total_samples)
print("event overlays:", len(browser.state.events))
print("first channels:", [chan["labels"] for chan in EEG["chanlocs"][:5]])
