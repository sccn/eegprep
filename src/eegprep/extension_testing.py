"""Reusable assertions for EEGPrep extension authors."""

from __future__ import annotations

from importlib import metadata
from typing import Any

from eegprep.extensions import (
    EXTENSION_ENTRY_POINT_GROUP,
    ExtensionAction,
    ExtensionPopFunction,
    ExtensionRegistry,
    ExtensionSpec,
    ExtensionStatus,
    validate_extension_spec,
)


class ExtensionTestHarness:
    """Assertion helper for extension author test suites."""

    def __init__(self, spec: ExtensionSpec) -> None:
        if not isinstance(spec, ExtensionSpec):
            raise TypeError("ExtensionTestHarness requires an ExtensionSpec")
        self.spec = spec

    @classmethod
    def from_entry_point(
        cls,
        entry_point_name: str,
        *,
        entry_points_provider: Any = metadata.entry_points,
        current_version: str | None = None,
        version_provider: Any = metadata.version,
    ) -> "ExtensionTestHarness":
        """Load an extension spec from the ``eegprep.extensions`` entry-point group."""
        return cls(
            assert_extension_entry_point_loads(
                entry_point_name,
                entry_points_provider=entry_points_provider,
                current_version=current_version,
                version_provider=version_provider,
            )
        )

    def assert_spec_valid(
        self,
        *,
        current_version: str | None = None,
        version_provider: Any = metadata.version,
        check_resources: bool = True,
    ) -> None:
        """Assert that the spec passes EEGPrep registry validation."""
        result = validate_extension_spec(
            self.spec,
            current_version=current_version,
            version_provider=version_provider,
            check_resources=check_resources,
        )
        if not result.ok:
            raise AssertionError("Extension spec is invalid: " + "; ".join(result.errors))

    def assert_menus_register(self) -> None:
        """Assert that every declared menu references a declared action or ``pop_*`` function."""
        registered_actions = {action.name for action in self.spec.actions}
        registered_actions.update(pop_function.name for pop_function in self.spec.pop_functions)
        for menu in self.spec.menus:
            if menu.action not in registered_actions:
                raise AssertionError(
                    f"Extension menu {' > '.join(menu.path)!r} references undeclared action {menu.action!r}"
                )

    def assert_help_resources_exist(self) -> None:
        """Assert that every declared help resource is packaged and readable."""
        for resource in self.spec.help_resources:
            if not resource.exists():
                raise AssertionError(f"Extension help resource {resource.package}:{resource.path} is missing")

    def assert_actions_load(self) -> None:
        """Assert that every declared action lazy target loads to a callable object."""
        for action in self.spec.actions:
            self.assert_action_callable(action.name)

    def assert_pop_functions_load(self) -> None:
        """Assert that every declared ``pop_*`` lazy target loads to a callable object."""
        for pop_function in self.spec.pop_functions:
            self.assert_pop_function_callable(pop_function.name)

    def assert_action_callable(self, action_name: str) -> Any:
        """Load and return one action target, asserting that it is callable."""
        action = self._action(action_name)
        target = action.load()
        if not callable(target):
            raise AssertionError(f"Extension action {action_name!r} did not load to a callable")
        return target

    def assert_pop_function_callable(self, pop_function_name: str) -> Any:
        """Load and return one ``pop_*`` target, asserting that it is callable."""
        pop_function = self._pop_function(pop_function_name)
        target = pop_function.load()
        if not callable(target):
            raise AssertionError(f"Extension pop function {pop_function_name!r} did not load to a callable")
        return target

    def assert_action_history_result(self, action_name: str, *args: Any, **kwargs: Any) -> tuple[Any, str]:
        """Call an action and assert it returns an EEGLAB-style ``(EEG, com)`` result."""
        target = self.assert_action_callable(action_name)
        return _assert_history_result(target(*args, **kwargs), action_name)

    def assert_pop_function_history_result(self, pop_function_name: str, *args: Any, **kwargs: Any) -> tuple[Any, str]:
        """Call a ``pop_*`` function with ``return_com=True`` and assert ``(EEG, com)`` output."""
        target = self.assert_pop_function_callable(pop_function_name)
        kwargs.setdefault("return_com", True)
        return _assert_history_result(target(*args, **kwargs), pop_function_name)

    def assert_all_static_contracts(self) -> None:
        """Assert static spec, menu, help, action, and ``pop_*`` loading contracts."""
        self.assert_spec_valid()
        self.assert_menus_register()
        self.assert_help_resources_exist()
        self.assert_actions_load()
        self.assert_pop_functions_load()

    def _action(self, name: str) -> ExtensionAction:
        for action in self.spec.actions:
            if action.name == name:
                return action
        raise AssertionError(f"Extension action {name!r} is not declared")

    def _pop_function(self, name: str) -> ExtensionPopFunction:
        for pop_function in self.spec.pop_functions:
            if pop_function.name == name:
                return pop_function
        raise AssertionError(f"Extension pop function {name!r} is not declared")


def assert_extension_entry_point_loads(
    entry_point_name: str,
    *,
    entry_points_provider: Any = metadata.entry_points,
    current_version: str | None = None,
    version_provider: Any = metadata.version,
) -> ExtensionSpec:
    """Assert that an ``eegprep.extensions`` entry point loads to a usable spec."""
    registry = ExtensionRegistry(
        include_bundled=False,
        entry_points_provider=entry_points_provider,
        current_version=current_version,
        version_provider=version_provider,
    )
    records = registry.discover()
    candidates = [
        record for record in records if record.entry_point_name == entry_point_name or record.name == entry_point_name
    ]
    if not candidates:
        raise AssertionError(
            f"No {EXTENSION_ENTRY_POINT_GROUP!r} entry point or extension named {entry_point_name!r} was discovered"
        )
    record = candidates[0]
    if record.status not in {ExtensionStatus.BUNDLED, ExtensionStatus.INSTALLED, ExtensionStatus.CURATED}:
        details = "; ".join(record.errors) if record.errors else record.status.value
        raise AssertionError(f"Extension entry point {entry_point_name!r} is not loadable: {details}")
    if record.spec is None:
        raise AssertionError(f"Extension entry point {entry_point_name!r} did not return an ExtensionSpec")
    return record.spec


def _assert_history_result(result: Any, name: str) -> tuple[Any, str]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise AssertionError(f"{name!r} must return an (EEG, com) tuple for console/history integration")
    eeg, command = result
    if not isinstance(command, str) or not command.strip():
        raise AssertionError(f"{name!r} must return a non-empty command history string")
    return eeg, command


__all__ = [
    "ExtensionTestHarness",
    "assert_extension_entry_point_loads",
]
