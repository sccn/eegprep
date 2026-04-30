"""Manifest and deviation loading for the parity harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    tomllib = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ToleranceProfile:
    """Numerical tolerance settings for a parity case."""

    name: str
    rtol: float = 0.0
    atol: float = 0.0
    equal_nan: bool = True
    note: str = ""


@dataclass(frozen=True)
class ParityCase:
    """A single parity target defined in the manifest."""

    id: str
    surface: str
    tiers: tuple[str, ...]
    oracle: str
    comparison: str
    tolerance: str
    datasets: tuple[str, ...] = ()
    owner: str = ""
    module: str = ""
    notes: str = ""
    blocking: str = "release"
    workflow_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParityDeviation:
    """An approved temporary deviation from strict parity."""

    id: str
    case_id: str
    issue: str
    owner: str
    reason: str
    scope: str = ""
    expires_on: str = ""
    blocking: str = "nightly"
    max_rtol: float | None = None
    max_atol: float | None = None


@dataclass(frozen=True)
class ParityManifest:
    """Loaded parity manifest plus any known deviations."""

    version: int
    default_backend: str
    tolerances: dict[str, ToleranceProfile]
    cases: tuple[ParityCase, ...]
    deviations: tuple[ParityDeviation, ...] = ()

    def get_tolerance(self, name: str) -> ToleranceProfile:
        """Return a named tolerance profile."""
        try:
            return self.tolerances[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tolerance profile: {name}") from exc

    def iter_cases(self, *tiers: str) -> Iterable[ParityCase]:
        """Iterate cases, optionally filtering by one or more tiers."""
        if not tiers:
            return iter(self.cases)
        required = set(tiers)
        return (case for case in self.cases if required.intersection(case.tiers))

    def deviations_for_case(self, case_id: str) -> tuple[ParityDeviation, ...]:
        """Return approved deviations for a parity case."""
        return tuple(dev for dev in self.deviations if dev.case_id == case_id)

    def summary(self) -> dict[str, Any]:
        """Return a machine-readable coverage summary."""
        tier_counts: dict[str, int] = {}
        surface_counts: dict[str, int] = {}
        for case in self.cases:
            surface_counts[case.surface] = surface_counts.get(case.surface, 0) + 1
            for tier in case.tiers:
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
        return {
            "version": self.version,
            "default_backend": self.default_backend,
            "case_count": len(self.cases),
            "deviation_count": len(self.deviations),
            "tiers": tier_counts,
            "surfaces": surface_counts,
        }


def _resource_path(filename: str) -> Path:
    return Path(__file__).resolve().parent.parent / "resources" / filename


def default_manifest_path() -> Path:
    """Return the packaged default parity manifest."""
    return _resource_path("parity_manifest.toml")


def default_deviations_path() -> Path:
    """Return the packaged default known-deviations registry."""
    return _resource_path("parity_known_deviations.toml")


def _load_toml(path: Path) -> dict[str, Any]:
    parser = tomllib
    if parser is None:  # pragma: no cover - Python < 3.11 without test/dev extras
        try:
            import tomli as parser
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Loading parity TOML manifests on Python <3.11 requires tomli; "
                "install eegprep[tests] or eegprep[dev]."
            ) from exc
    with path.open("rb") as handle:
        return parser.load(handle)


def _load_tolerances(raw: Mapping[str, Mapping[str, Any]]) -> dict[str, ToleranceProfile]:
    tolerances: dict[str, ToleranceProfile] = {}
    for name, data in raw.items():
        tolerances[name] = ToleranceProfile(
            name=name,
            rtol=float(data.get("rtol", 0.0)),
            atol=float(data.get("atol", 0.0)),
            equal_nan=bool(data.get("equal_nan", True)),
            note=str(data.get("note", "")),
        )
    return tolerances


def _load_cases(raw_cases: Iterable[Mapping[str, Any]]) -> tuple[ParityCase, ...]:
    cases = []
    for entry in raw_cases:
        cases.append(
            ParityCase(
                id=str(entry["id"]),
                surface=str(entry["surface"]),
                tiers=tuple(str(v) for v in entry.get("tiers", [])),
                oracle=str(entry.get("oracle", "")),
                comparison=str(entry["comparison"]),
                tolerance=str(entry["tolerance"]),
                datasets=tuple(str(v) for v in entry.get("datasets", [])),
                owner=str(entry.get("owner", "")),
                module=str(entry.get("module", "")),
                notes=str(entry.get("notes", "")),
                blocking=str(entry.get("blocking", "release")),
                workflow_paths=tuple(str(v) for v in entry.get("workflow_paths", [])),
            )
        )
    return tuple(cases)


def _load_deviations(raw_deviations: Iterable[Mapping[str, Any]]) -> tuple[ParityDeviation, ...]:
    deviations = []
    for entry in raw_deviations:
        deviations.append(
            ParityDeviation(
                id=str(entry["id"]),
                case_id=str(entry["case_id"]),
                issue=str(entry["issue"]),
                owner=str(entry["owner"]),
                reason=str(entry["reason"]),
                scope=str(entry.get("scope", "")),
                expires_on=str(entry.get("expires_on", "")),
                blocking=str(entry.get("blocking", "nightly")),
                max_rtol=(None if entry.get("max_rtol") is None else float(entry["max_rtol"])),
                max_atol=(None if entry.get("max_atol") is None else float(entry["max_atol"])),
            )
        )
    return tuple(deviations)


def load_deviations(path: Path | str | None = None) -> tuple[ParityDeviation, ...]:
    """Load the known-deviations registry."""
    source = Path(path) if path is not None else default_deviations_path()
    data = _load_toml(source)
    return _load_deviations(data.get("deviations", []))


def load_manifest(
    path: Path | str | None = None,
    *,
    deviations_path: Path | str | None = None,
) -> ParityManifest:
    """Load the parity manifest and attach known deviations."""
    source = Path(path) if path is not None else default_manifest_path()
    data = _load_toml(source)
    deviations = load_deviations(deviations_path)
    return ParityManifest(
        version=int(data.get("version", 1)),
        default_backend=str(data.get("default_backend", "artifact_oracle")),
        tolerances=_load_tolerances(data.get("tolerances", {})),
        cases=_load_cases(data.get("cases", [])),
        deviations=deviations,
    )
