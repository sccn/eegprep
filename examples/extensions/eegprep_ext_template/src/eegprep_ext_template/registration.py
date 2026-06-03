"""Extension registration for the template package."""

from __future__ import annotations

from eegprep.extensions import (
    EXTENSION_API_VERSION,
    ExtensionAction,
    ExtensionDependency,
    ExtensionMenu,
    ExtensionPopFunction,
    ExtensionResource,
    ExtensionSourceType,
    ExtensionSpec,
    LazyImport,
)

from ._version import __version__


def register() -> ExtensionSpec:
    """Return the EEGPrep extension spec for this package."""
    return ExtensionSpec(
        name="eegprep_ext_template",
        display_name="EEGPrep Template Extension",
        version=__version__,
        api_version=EXTENSION_API_VERSION,
        package_name="eegprep_ext_template",
        source_type=ExtensionSourceType.INSTALLED,
        description="Minimal extension showing pop function, menu, help, tests, and sample data.",
        maintainer="Your lab",
        capabilities=("signal-transform", "menu", "help", "sample-data"),
        dependencies=(ExtensionDependency("numpy", ">=1.23"),),
        menus=(
            ExtensionMenu(
                path=("Tools", "Template extension"),
                action="pop_template_gain",
                label="Apply template gain",
            ),
        ),
        actions=(
            ExtensionAction(
                name="pop_template_gain",
                target=LazyImport("eegprep_ext_template.pop_template_gain", "pop_template_gain"),
                display_name="Apply template gain",
                description="Scale EEG.data by a numeric gain.",
                capabilities=("mutates-eeg", "history"),
            ),
        ),
        pop_functions=(
            ExtensionPopFunction(
                name="pop_template_gain",
                target=LazyImport("eegprep_ext_template.pop_template_gain", "pop_template_gain"),
                description="Scale EEG.data by a numeric gain.",
            ),
        ),
        help_resources=(ExtensionResource("eegprep_ext_template", "resources/help/pop_template_gain.md"),),
        package_data_resources=(ExtensionResource("eegprep_ext_template", "resources/sample_data/template_eeg.json"),),
        eegprep_requires=">=0.2.23",
    )
