PLUGIN_MENU - EEGPrep Extension Manager

Use `plugin_menu` to inspect EEGPrep extensions from scripts, the console, or
the GUI Extension Manager.

Examples:

```python
plugins = plugin_menu(show=False)
status, names, pluginstruct = plugin_status("ICLabel", exactmatch=True)
print(format_plugin_menu())
```

The Extension Manager combines installed extension records from the EEGPrep
extension registry with optional curated catalog metadata. Catalog entries point
to package names, repositories, documentation, maintainers, and capabilities.
They do not contain extension code, zip archives, or automatic installers.

Status meanings:

- `bundled`: shipped with EEGPrep.
- `installed`: discovered from a Python package entry point.
- `curated`: listed in the metadata catalog but not installed.
- `disabled`: installed but disabled by registry configuration.
- `incompatible`: installed but incompatible with the current EEGPrep version or
  extension API version.
- `failed_import`: entry-point discovery failed.
- `invalid_spec`: the extension returned invalid metadata.
- `missing_dependency`: a required dependency is not installed.

The GUI details panel shows description, version, maintainer, package name,
documentation URL, source, capabilities, errors, catalog conflicts, and
copyable install/update commands when catalog metadata is available.

Installing Python packages executes third-party code. Review the package,
maintainer, source repository, and documentation before running any command.
EEGPrep displays install guidance but never downloads, unzips, installs,
updates, or removes extension code from this manager.
