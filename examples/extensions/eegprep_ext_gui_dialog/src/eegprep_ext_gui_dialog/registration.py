"""Extension registration for the GUI dialog example."""

from __future__ import annotations

from eegprep.extensions import (
    EXTENSION_API_VERSION,
    ExtensionAction,
    ExtensionDependency,
    ExtensionMenu,
    ExtensionPopFunction,
    ExtensionSourceType,
    ExtensionSpec,
    LazyImport,
)


def register() -> ExtensionSpec:
    """Return the GUI dialog example extension spec."""
    return ExtensionSpec(
        name="eegprep_ext_gui_dialog",
        display_name="GUI Dialog Example",
        version="0.1.0",
        api_version=EXTENSION_API_VERSION,
        package_name="eegprep-ext-gui-dialog",
        source_type=ExtensionSourceType.INSTALLED,
        capabilities=("gui-dialog", "history"),
        dependencies=(ExtensionDependency("numpy", ">=1.23"),),
        menus=(ExtensionMenu(path=("Tools", "Example dialogs"), action="pop_demo_threshold", label="Threshold demo"),),
        actions=(
            ExtensionAction(
                name="pop_demo_threshold",
                target=LazyImport("eegprep_ext_gui_dialog.pop_demo_threshold", "pop_demo_threshold"),
                capabilities=("gui-dialog", "mutates-eeg", "history"),
            ),
        ),
        pop_functions=(
            ExtensionPopFunction(
                name="pop_demo_threshold",
                target=LazyImport("eegprep_ext_gui_dialog.pop_demo_threshold", "pop_demo_threshold"),
            ),
        ),
        eegprep_requires=">=0.2.23",
    )
