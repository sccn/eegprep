"""Pop the clicked figure axes out into an enlarged window, as EEGLAB ``axcopy``."""

from __future__ import annotations

from typing import Any, Callable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from eegprep.functions.popfunc.plot_utils import backend_can_display


def axcopy(figure: Figure, redraws: dict[Axes, Callable[[Axes], Any]]) -> None:
    """Enlarge the left-clicked axes into a pop-up window, like EEGLAB ``axcopy``.

    ``redraws`` maps each interactive axes to a closure that redraws its content,
    full size, into a fresh axes. matplotlib cannot copy artists between figures
    the way EEGLAB copies graphic objects, so each axes carries a redraw closure
    instead. This is a no-op on non-interactive backends, which deliver no click
    events; tests exercise it by dispatching a synthetic click.
    """

    def _on_click(event: Any) -> None:
        if event.button != 1 or event.inaxes is None:
            return
        redraw = redraws.get(event.inaxes)
        if redraw is None:
            return
        popup = plt.figure(figsize=(5, 5))
        redraw(popup.add_axes([0.13, 0.10, 0.80, 0.80]))
        # The GUI/console run with interactive mode off, so display the pop-up
        # explicitly (as show_figures does); a no-op on file-output backends.
        if backend_can_display():
            popup.show()

    figure.canvas.mpl_connect("button_press_event", _on_click)


__all__ = ["axcopy"]
