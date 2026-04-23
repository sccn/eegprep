"""Oracle backends for parity validation."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


class OracleBackend:
    """Names of supported oracle backends."""

    LIVE_MATLAB_ENGINE = "live_matlab_engine"
    LIVE_MATLAB_BATCH = "live_matlab_batch"
    ARTIFACT_ORACLE = "artifact_oracle"


@dataclass(frozen=True)
class BackendAvailability:
    """Availability information for a single backend."""

    name: str
    available: bool
    detail: str = ""


class MatlabEngineOracle:
    """Live oracle backed by the Python MATLAB engine."""

    backend_name = OracleBackend.LIVE_MATLAB_ENGINE

    def __init__(self, engine: Optional[Any] = None):
        if engine is None:
            import matlab.engine

            engine = matlab.engine.start_matlab()
        self.engine = engine

    def eval(self, command: str, *, nargout: int = 0) -> Any:
        """Evaluate MATLAB code through the live engine."""
        return self.engine.eval(command, nargout=nargout)


class MatlabBatchOracle:
    """Live oracle backed by the MATLAB CLI."""

    backend_name = OracleBackend.LIVE_MATLAB_BATCH

    def __init__(self, executable: Optional[str] = None):
        self.executable = executable or shutil.which("matlab")
        if not self.executable:
            raise RuntimeError("MATLAB CLI not found on PATH")

    def run(self, command: str, *, cwd: Optional[str | Path] = None, timeout: Optional[int] = None) -> subprocess.CompletedProcess[str]:
        """Run a MATLAB batch command."""
        cmd = [self.executable, "-batch", command]
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )


class ArtifactOracle:
    """Oracle that serves reference artifacts from disk."""

    backend_name = OracleBackend.ARTIFACT_ORACLE

    def __init__(self, artifact_root: str | Path):
        self.artifact_root = Path(artifact_root)

    def resolve(self, relative_path: str | Path) -> Path:
        """Resolve an artifact path relative to the configured root."""
        return self.artifact_root / Path(relative_path)

    def exists(self, relative_path: str | Path) -> bool:
        """Return whether an artifact exists."""
        return self.resolve(relative_path).exists()

    def load_text(self, relative_path: str | Path) -> str:
        """Load a text artifact."""
        return self.resolve(relative_path).read_text()

    def load_json(self, relative_path: str | Path) -> Any:
        """Load a JSON artifact."""
        return json.loads(self.resolve(relative_path).read_text())


def detect_oracle_backends() -> dict[str, BackendAvailability]:
    """Detect which oracle backends are available in the current environment."""
    results: dict[str, BackendAvailability] = {}

    try:
        import matlab.engine  # noqa: F401
    except Exception as exc:
        results[OracleBackend.LIVE_MATLAB_ENGINE] = BackendAvailability(
            name=OracleBackend.LIVE_MATLAB_ENGINE,
            available=False,
            detail=str(exc),
        )
    else:
        results[OracleBackend.LIVE_MATLAB_ENGINE] = BackendAvailability(
            name=OracleBackend.LIVE_MATLAB_ENGINE,
            available=True,
            detail="python matlab.engine import succeeded",
        )

    matlab_cli = shutil.which("matlab")
    results[OracleBackend.LIVE_MATLAB_BATCH] = BackendAvailability(
        name=OracleBackend.LIVE_MATLAB_BATCH,
        available=bool(matlab_cli),
        detail=matlab_cli or "matlab not found on PATH",
    )

    return results


def resolve_oracle_backend(
    preferred: Optional[str] = None,
    *,
    artifact_root: Optional[str | Path] = None,
) -> Any:
    """Return an oracle backend instance using the preferred or best available backend."""
    availability = detect_oracle_backends()
    order = [preferred] if preferred else [
        OracleBackend.LIVE_MATLAB_ENGINE,
        OracleBackend.LIVE_MATLAB_BATCH,
        OracleBackend.ARTIFACT_ORACLE,
    ]
    if preferred is None and artifact_root is None:
        order = [OracleBackend.LIVE_MATLAB_ENGINE, OracleBackend.LIVE_MATLAB_BATCH]
    for backend_name in order:
        if backend_name == OracleBackend.LIVE_MATLAB_ENGINE and availability[backend_name].available:
            return MatlabEngineOracle()
        if backend_name == OracleBackend.LIVE_MATLAB_BATCH and availability[backend_name].available:
            return MatlabBatchOracle()
        if backend_name == OracleBackend.ARTIFACT_ORACLE and artifact_root is not None:
            return ArtifactOracle(artifact_root)
    if artifact_root is not None:
        return ArtifactOracle(artifact_root)
    details = {name: info.detail for name, info in availability.items()}
    raise RuntimeError(f"No parity oracle backend is available: {details}")
