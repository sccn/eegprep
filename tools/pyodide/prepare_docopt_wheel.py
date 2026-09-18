"""Build the pure-Python docopt wheel required by the Pyodide harness."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DOCOPT_VERSION = "0.6.2"
DOCOPT_SDIST_URL = (
    "https://files.pythonhosted.org/packages/a2/55/8f8cab2afd404cf578136ef2cc5dfb50baa1761b68c9da1fb1e4eed343c9/"
    "docopt-0.6.2.tar.gz"
)
DOCOPT_SDIST_SHA256 = "49b3a825280bd66b3aa83585ef59c4a8c82f2c8a522dbe754a8bc8d08c85c491"
HTTP_TIMEOUT_S = 60


def verify_sha256(path: Path, expected: str) -> None:
    """Raise when ``path`` does not match the expected SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}")


def _download_sdist(destination: Path) -> None:
    request = urllib.request.Request(DOCOPT_SDIST_URL, headers={"User-Agent": "eegprep-pyodide-harness"})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def _extract_sdist(archive_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if destination.resolve() not in target.parents:
                raise ValueError(f"Unsafe path in docopt archive: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Links are not allowed in docopt archive: {member.name}")
        archive.extractall(destination)

    roots = [path for path in destination.iterdir() if path.is_dir() and (path / "setup.py").is_file()]
    if len(roots) != 1:
        raise RuntimeError(f"Expected one docopt source directory, found {len(roots)}")
    return roots[0]


def _validate_wheel(wheel_path: Path) -> None:
    expected_prefix = f"docopt-{DOCOPT_VERSION}-"
    if not wheel_path.name.startswith(expected_prefix) or not wheel_path.name.endswith("-none-any.whl"):
        raise RuntimeError(f"Expected a universal docopt wheel, got {wheel_path.name}")
    with zipfile.ZipFile(wheel_path) as wheel:
        wheel_metadata = [name for name in wheel.namelist() if name.endswith("/WHEEL")]
        if len(wheel_metadata) != 1:
            raise RuntimeError("Generated docopt wheel has no unique WHEEL metadata file")
        metadata = wheel.read(wheel_metadata[0]).decode("utf-8")
    if "Root-Is-Purelib: true" not in metadata:
        raise RuntimeError("Generated docopt wheel is not marked as pure Python")


def build_docopt_wheel(output_dir: Path) -> Path:
    """Build and validate ``docopt==0.6.2`` into ``output_dir``."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError(f"Docopt output directory must be empty: {output_dir}")

    with tempfile.TemporaryDirectory(prefix="eegprep-docopt-") as temporary:
        temporary_path = Path(temporary)
        archive_path = temporary_path / "docopt-0.6.2.tar.gz"
        _download_sdist(archive_path)
        verify_sha256(archive_path, DOCOPT_SDIST_SHA256)
        source_root = _extract_sdist(archive_path, temporary_path / "source")
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(output_dir)],
            cwd=source_root,
            check=True,
        )

    wheels = sorted(output_dir.glob(f"docopt-{DOCOPT_VERSION}-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one generated docopt wheel, found {len(wheels)}")
    _validate_wheel(wheels[0])
    return wheels[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    wheel = build_docopt_wheel(args.output_dir)
    print(f"Built {wheel} from {DOCOPT_SDIST_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
