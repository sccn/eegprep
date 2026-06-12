"""
Public API, Plugins, and Console Session
========================================

This example shows how to use EEGPrep's public imports for the shared
GUI/console session and bundled plugin inventory without opening a GUI window.
The same session object is used by ``eegprep-console`` and the main window.
"""

# %%
# Create a small EEGLAB-like dataset.

import numpy as np

import eegprep


def make_demo_eeg() -> dict:
    """Return a tiny continuous EEG dictionary for API smoke examples."""
    eeg = eegprep.eeg_emptyset()
    data = np.vstack(
        [
            np.sin(np.linspace(0, 4 * np.pi, 100)),
            np.cos(np.linspace(0, 4 * np.pi, 100)),
        ]
    )
    eeg.update(
        {
            "setname": "public API demo",
            "data": data,
            "nbchan": data.shape[0],
            "pnts": data.shape[1],
            "trials": 1,
            "srate": 100.0,
            "xmin": 0.0,
            "xmax": (data.shape[1] - 1) / 100.0,
            "times": np.arange(data.shape[1]) * 10.0,
            "chanlocs": [{"labels": "Cz"}, {"labels": "Pz"}],
        }
    )
    return eeg


EEG = make_demo_eeg()

# %%
# Store it through the console-aware public workspace.

session = eegprep.EEGPrepSession()
workspace = eegprep.EEGPrepConsoleWorkspace(session)

try:
    result = workspace.namespace["pop_newset"]([], EEG, 0, "setname", "console demo")
    print(result)
    print("CURRENTSET:", session.current_set_value())
    print("Dataset name:", session.EEG["setname"])

    # %%
    # Inspect bundled plugin metadata without opening Qt.

    plugins = eegprep.plugin_menu(session=session, show=False)
    print("Bundled plugins:", ", ".join(plugin["name"] for plugin in plugins))
    print("ICLabel status:", eegprep.plugin_status("ICLabel", exactmatch=True)[0])
finally:
    workspace.close()
