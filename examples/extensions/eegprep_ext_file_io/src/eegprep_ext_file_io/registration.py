"""Extension registration for the file importer/exporter example."""

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
    """Return the file I/O example extension spec."""
    return ExtensionSpec(
        name="eegprep_ext_file_io",
        display_name="File I/O Example",
        version="0.1.0",
        api_version=EXTENSION_API_VERSION,
        package_name="eegprep-ext-file-io",
        source_type=ExtensionSourceType.INSTALLED,
        capabilities=("file-import", "file-export"),
        dependencies=(ExtensionDependency("numpy", ">=1.23"),),
        menus=(
            ExtensionMenu(path=("File", "Import data"), action="pop_demo_import_csv", label="Example CSV file"),
            ExtensionMenu(path=("File", "Export"), action="pop_demo_export_csv", label="Example CSV file"),
        ),
        actions=(
            ExtensionAction(
                name="pop_demo_import_csv",
                target=LazyImport("eegprep_ext_file_io.io", "pop_demo_import_csv"),
                capabilities=("imports-eeg", "history"),
            ),
            ExtensionAction(
                name="pop_demo_export_csv",
                target=LazyImport("eegprep_ext_file_io.io", "pop_demo_export_csv"),
                capabilities=("exports-eeg", "history"),
            ),
        ),
        pop_functions=(
            ExtensionPopFunction(
                name="pop_demo_import_csv",
                target=LazyImport("eegprep_ext_file_io.io", "pop_demo_import_csv"),
            ),
            ExtensionPopFunction(
                name="pop_demo_export_csv",
                target=LazyImport("eegprep_ext_file_io.io", "pop_demo_export_csv"),
            ),
        ),
        eegprep_requires=">=0.2.23",
    )
