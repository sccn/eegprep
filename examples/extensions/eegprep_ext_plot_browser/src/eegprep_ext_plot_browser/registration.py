"""Extension registration for the plot/browser example."""

from __future__ import annotations

from eegprep.extensions import (
    EXTENSION_API_VERSION,
    ExtensionAction,
    ExtensionMenu,
    ExtensionPopFunction,
    ExtensionSourceType,
    ExtensionSpec,
    LazyImport,
)


def register() -> ExtensionSpec:
    """Return the plot/browser example extension spec."""
    return ExtensionSpec(
        name="eegprep_ext_plot_browser",
        display_name="Plot Browser Example",
        version="0.1.0",
        api_version=EXTENSION_API_VERSION,
        package_name="eegprep_ext_plot_browser",
        source_type=ExtensionSourceType.INSTALLED,
        capabilities=("plot", "browser", "history"),
        menus=(ExtensionMenu(path=("Plot", "Example browser"), action="pop_demo_browser", label="Browser demo"),),
        actions=(
            ExtensionAction(
                name="pop_demo_browser",
                target=LazyImport("eegprep_ext_plot_browser.browser", "pop_demo_browser"),
                capabilities=("plot", "browser", "history"),
            ),
        ),
        pop_functions=(
            ExtensionPopFunction(
                name="pop_demo_browser",
                target=LazyImport("eegprep_ext_plot_browser.browser", "pop_demo_browser"),
            ),
        ),
        eegprep_requires=">=0.2.23",
    )
