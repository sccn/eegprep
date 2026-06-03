"""Extension registration for the optional dependency example."""

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


def register() -> ExtensionSpec:
    """Return the optional dependency example extension spec."""
    return ExtensionSpec(
        name="eegprep_ext_optional_dependency",
        display_name="Optional Dependency Example",
        version="0.1.0",
        api_version=EXTENSION_API_VERSION,
        package_name="eegprep_ext_optional_dependency",
        source_type=ExtensionSourceType.INSTALLED,
        capabilities=("optional-dependency", "packaged-model"),
        dependencies=(ExtensionDependency("eegprep-template-optional-model", ">=1.0", optional=True),),
        menus=(
            ExtensionMenu(
                path=("Tools", "Example optional dependency"),
                action="pop_demo_optional_score",
                label="Optional model score",
            ),
        ),
        actions=(
            ExtensionAction(
                name="pop_demo_optional_score",
                target=LazyImport("eegprep_ext_optional_dependency.optional_model", "pop_demo_optional_score"),
                capabilities=("optional-dependency", "mutates-eeg", "history"),
            ),
        ),
        pop_functions=(
            ExtensionPopFunction(
                name="pop_demo_optional_score",
                target=LazyImport("eegprep_ext_optional_dependency.optional_model", "pop_demo_optional_score"),
            ),
        ),
        package_data_resources=(
            ExtensionResource("eegprep_ext_optional_dependency", "resources/model/model-card.json"),
        ),
        eegprep_requires=">=0.2.23",
    )
