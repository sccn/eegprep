"""Small graphics-lifetime boundaries for Python-owned source contracts."""

from matplotlib import pyplot as plt


def close_reference_gui(eeglab_backend, request, *, window=None, all_figures=False):
    """Close the source's current figure, supplied native dialog, or all figures."""
    if request.config.getoption("--eeglab-backend") == "matlab":
        args = ("all",) if all_figures else ()
        eeglab_backend("close", *args, nargout=0)
    elif window is not None:
        window.close()
    else:
        plt.close("all" if all_figures else None)
