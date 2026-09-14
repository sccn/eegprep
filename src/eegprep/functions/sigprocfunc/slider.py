"""Add EEGLAB-style viewport sliders to a Matplotlib figure."""

from __future__ import annotations

from dataclasses import dataclass

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Button, Slider


@dataclass
class SliderControls:
    """Handles and original layout for controls created by :func:`slider`."""

    figure: Figure
    target_axes: tuple[Axes, ...]
    original_positions: tuple[tuple[float, float, float, float], ...]
    horizontal: Slider | None
    vertical: Slider | None
    dismiss: Button

    def remove(self) -> None:
        """Restore the original axes positions and remove all viewport controls."""
        for axis, position in zip(self.target_axes, self.original_positions):
            axis.set_position(position)
        for widget in (self.horizontal, self.vertical, self.dismiss):
            if widget is not None and widget.ax in self.figure.axes:
                self.figure.delaxes(widget.ax)
        if getattr(self.figure, "_eegprep_slider_controls", None) is self:
            delattr(self.figure, "_eegprep_slider_controls")
        self.figure.canvas.draw_idle()


def slider(
    handler: Figure,
    horiz: bool | int,
    vert: bool | int = False,
    horizmag: float = 1.0,
    vertmag: float = 1.0,
    allowsup: bool | int = True,
) -> SliderControls:
    """Add horizontal and/or vertical viewport controls to a figure.

    The figure's existing axes are magnified using the same normalized-position
    transformation as EEGLAB. Moving a control pans those axes; the ``x``
    button restores their exact original positions and removes the controls.

    Args:
        handler: Figure whose axes should be magnified and panned.
        horiz: Whether to add a horizontal slider.
        vert: Whether to add a vertical slider.
        horizmag: Horizontal magnification, at least one.
        vertmag: Vertical magnification, at least one.
        allowsup: Whether the dismiss button is active.

    Returns:
        Handles for programmatic control and removal.
    """
    if not isinstance(handler, Figure):
        raise TypeError("slider handler must be a Matplotlib Figure")
    horizmag, vertmag = float(horizmag), float(vertmag)
    if horizmag < 1 or vertmag < 1:
        raise ValueError("slider magnification factors must be at least 1")
    target_axes = tuple(handler.axes)
    original = tuple(axis.get_position().bounds for axis in target_axes)

    horizontal_widget = None
    vertical_widget = None
    if vert:
        vertical_widget = Slider(
            handler.add_axes([0.94, 0.15, 0.02, 0.70]), "", 0.0, 1.0, valinit=1.0, orientation="vertical"
        )
    if horiz:
        horizontal_widget = Slider(handler.add_axes([0.15, 0.04, 0.70, 0.03]), "", 0.0, 1.0, valinit=0.0)
    dismiss = Button(handler.add_axes([0.94, 0.04, 0.035, 0.04]), "x")
    controls = SliderControls(handler, target_axes, original, horizontal_widget, vertical_widget, dismiss)

    def update(_value: float | None = None) -> None:
        horizontal_value = horizontal_widget.val if horizontal_widget is not None else 1.0
        vertical_value = vertical_widget.val if vertical_widget is not None else 1.0
        for axis, (left, bottom, width, height) in zip(target_axes, original):
            axis.set_position(
                [
                    left * horizmag - (horizmag - 1.0) * horizontal_value,
                    bottom * vertmag - (vertmag - 1.0) * vertical_value,
                    width * horizmag,
                    height * vertmag,
                ]
            )
        handler.canvas.draw_idle()

    if horizontal_widget is not None:
        horizontal_widget.on_changed(update)
    if vertical_widget is not None:
        vertical_widget.on_changed(update)
    if allowsup:
        dismiss.on_clicked(lambda _event: controls.remove())
    else:
        dismiss.set_active(False)
    setattr(handler, "_eegprep_slider_controls", controls)
    update()
    return controls


__all__ = ["SliderControls", "slider"]
