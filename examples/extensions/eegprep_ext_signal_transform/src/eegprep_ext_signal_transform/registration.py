"""Extension registration for the pure signal-transform example."""

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
    """Return the signal-transform example extension spec."""
    return ExtensionSpec(
        name="eegprep_ext_signal_transform",
        display_name="Signal Transform Example",
        version="0.1.0",
        api_version=EXTENSION_API_VERSION,
        package_name="eegprep_ext_signal_transform",
        source_type=ExtensionSourceType.INSTALLED,
        capabilities=("signal-transform",),
        dependencies=(ExtensionDependency("numpy", ">=1.23"),),
        menus=(ExtensionMenu(path=("Tools", "Example transforms"), action="pop_demo_center", label="Center channels"),),
        actions=(
            ExtensionAction(
                name="pop_demo_center",
                target=LazyImport("eegprep_ext_signal_transform.pop_demo_center", "pop_demo_center"),
                display_name="Center channels",
                capabilities=("mutates-eeg", "history"),
            ),
        ),
        pop_functions=(
            ExtensionPopFunction(
                name="pop_demo_center",
                target=LazyImport("eegprep_ext_signal_transform.pop_demo_center", "pop_demo_center"),
            ),
        ),
        eegprep_requires=">=0.2.23",
    )
