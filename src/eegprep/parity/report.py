"""Reporting helpers for parity harness results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .compare import ComparisonResult
from .config import ParityManifest


def _result_payload(result: ComparisonResult | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(result, ComparisonResult):
        return result.to_dict()
    return result


def render_markdown_report(
    *,
    title: str,
    manifest: ParityManifest,
    result: ComparisonResult | Mapping[str, Any],
    backend: str,
) -> str:
    """Render a compact markdown report."""
    payload = _result_payload(result)
    summary = manifest.summary()
    lines = [
        f"# {title}",
        "",
        f"- Backend: `{backend}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Deviations: `{summary['deviation_count']}`",
        "",
        "## Coverage",
        "",
    ]
    for tier, count in sorted(summary["tiers"].items()):
        lines.append(f"- `{tier}`: {count}")
    lines.extend(["", "## Result", "", "```json", json.dumps(payload, indent=2, sort_keys=True), "```", ""])
    return "\n".join(lines)


def write_markdown_report(
    path: str | Path,
    *,
    title: str,
    manifest: ParityManifest,
    result: ComparisonResult | Mapping[str, Any],
    backend: str,
) -> Path:
    """Write a markdown report to disk."""
    target = Path(path)
    target.write_text(
        render_markdown_report(title=title, manifest=manifest, result=result, backend=backend),
        encoding="utf-8",
    )
    return target


def write_json_report(path: str | Path, result: ComparisonResult | Mapping[str, Any]) -> Path:
    """Write a machine-readable JSON report to disk."""
    target = Path(path)
    target.write_text(json.dumps(_result_payload(result), indent=2, sort_keys=True), encoding="utf-8")
    return target
