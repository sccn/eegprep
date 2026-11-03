"""The joint release stops when eegprep and eegprep-lean do not share a version."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = REPO_ROOT / "scripts" / "release_packages.py"
    spec = importlib.util.spec_from_file_location("release_packages", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


release_packages = _load()


def _tree(tmp_path: Path, eegprep: str = "1.2.3", lean_init: str = "1.2.3", lean_project: str = "1.2.3") -> Path:
    init = tmp_path / "src" / "eegprep" / "__init__.py"
    init.parent.mkdir(parents=True)
    init.write_text(f'__version__ = "{eegprep}"\n', encoding="utf-8")
    lean = tmp_path / "packages" / "eegprep-lean" / "src" / "eegprep_lean" / "__init__.py"
    lean.parent.mkdir(parents=True)
    lean.write_text(f'__version__ = "{lean_init}"\n', encoding="utf-8")
    project = tmp_path / "packages" / "eegprep-lean" / "pyproject.toml"
    project.write_text(f'[project]\nversion = "{lean_project}"\n', encoding="utf-8")
    return tmp_path


def test_matching_versions_are_one_string(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    assert release_packages.require_one_version(release_packages.read_versions(root)) == "1.2.3"


def test_each_source_can_reject_the_release(tmp_path: Path) -> None:
    cases = (
        {"eegprep": "1.2.4"},
        {"lean_init": "1.2.4"},
        {"lean_project": "1.2.4"},
    )
    for case in cases:
        root = _tree(tmp_path / next(iter(case)), **case)
        with pytest.raises(release_packages.VersionMismatch, match="Set all three to the same version"):
            release_packages.require_one_version(release_packages.read_versions(root))


def test_a_tag_must_name_the_same_version() -> None:
    release_packages.require_tag("1.2.3", None)
    release_packages.require_tag("1.2.3", "1.2.3")
    with pytest.raises(release_packages.VersionMismatch, match="Tag v1.2.3") as caught:
        release_packages.require_tag("1.2.3", "9.9.9")
    assert "v9.9.9" in str(caught.value)
    assert "set both packages to 9.9.9" in str(caught.value)


def test_check_succeeds_only_when_the_tree_agrees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_REF", raising=False)
    monkeypatch.delenv("RELEASE_TAG", raising=False)
    assert release_packages.main(["--check"], root=_tree(tmp_path / "ok")) == 0
    assert release_packages.main(["--check"], root=_tree(tmp_path / "bad", lean_project="0.0.1")) == 1
    assert not (tmp_path / "bad" / "dist").exists()


def test_publish_does_not_build_when_versions_differ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_REF", raising=False)
    monkeypatch.delenv("RELEASE_TAG", raising=False)
    root = _tree(tmp_path, eegprep="0.4.0", lean_init="0.1.0.dev3", lean_project="0.1.0.dev3")
    assert release_packages.main(["--publish"], root=root) == 1
    assert not (root / "dist").exists()


def _artifacts(dist: Path, version: str = "0.5.0") -> None:
    for path in release_packages.artifact_paths(dist, version):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")


def _names(command: list[str]) -> list[str]:
    return [Path(part).name for part in command if part not in {"uv", "publish", "--dry-run"}]


def test_publish_uploads_the_artifact_files_lean_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UV_PUBLISH_TOKEN", "test-token")
    dist = tmp_path / "dist"
    _artifacts(dist)
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(release_packages.subprocess, "run", fake_run)
    release_packages.publish(dist, "0.5.0")

    lean = ["eegprep_lean-0.5.0-py3-none-any.whl", "eegprep_lean-0.5.0.tar.gz"]
    full = ["eegprep-0.5.0-py3-none-any.whl", "eegprep-0.5.0.tar.gz"]
    assert [_names(command) for command in calls] == [lean, full, lean, full]
    assert [command[2] for command in calls[:2]] == ["--dry-run", "--dry-run"]
    assert all(
        part != str(dist / project) for command in calls for part in command for project in ("eegprep", "eegprep-lean")
    )


def test_publish_stops_before_either_upload_without_a_shared_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("UV_PUBLISH_TOKEN", raising=False)
    monkeypatch.delenv("UV_PUBLISH_PASSWORD", raising=False)
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "github-oidc")
    dist = tmp_path / "dist"
    _artifacts(dist)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        release_packages.subprocess,
        "run",
        lambda command, check: calls.append(command),
    )
    with pytest.raises(release_packages.VersionMismatch, match="UV_PUBLISH_TOKEN") as caught:
        release_packages.publish(dist, "0.5.0")
    assert "Nothing was uploaded" in str(caught.value)
    assert "eegprep-lean" in str(caught.value)
    assert calls == []


def test_publish_names_the_missing_files_and_says_to_build_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UV_PUBLISH_TOKEN", "test-token")
    with pytest.raises(release_packages.VersionMismatch, match="Build both packages first") as caught:
        release_packages.publish(tmp_path / "dist", "0.5.0")
    message = str(caught.value)
    assert "eegprep-0.5.0-py3-none-any.whl" in message
    assert "eegprep_lean-0.5.0-py3-none-any.whl" in message
    assert "--publish" in message


def test_publish_does_not_upload_eegprep_when_lean_is_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UV_PUBLISH_TOKEN", "test-token")
    dist = tmp_path / "dist"
    _artifacts(dist)
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if any("eegprep_lean" in part for part in command):
            raise subprocess.CalledProcessError(2, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(release_packages.subprocess, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        release_packages.publish(dist, "0.5.0")
    assert calls
    assert all("eegprep-0.5.0" not in part for command in calls for part in command)


def test_github_tag_ref_is_the_version_without_the_v(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_REF", "refs/tags/v1.2.3")
    monkeypatch.delenv("RELEASE_TAG", raising=False)
    assert release_packages.release_tag() == "1.2.3"
    monkeypatch.setenv("GITHUB_REF", "refs/heads/develop")
    monkeypatch.setenv("RELEASE_TAG", "v4.5.6")
    assert release_packages.release_tag() == "4.5.6"
    monkeypatch.delenv("RELEASE_TAG")
    assert release_packages.release_tag() is None
