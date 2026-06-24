"""Shared HTML report rendering for CLI report commands."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from eegprep.cli.core import EEGPrepCLIError, file_sha256


def write_report_html(
    EEG: dict[str, Any],
    output_path: str | Path,
    *,
    metrics: dict[str, Any],
    dataset_path: str | Path | None = None,
    title: str = "EEGPrep Report",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write an HTML report for an EEG dict and return a manifest output entry."""
    from eegprep.cli.core import output_path as core_output_path
    del EEG, dataset_path
    target = core_output_path(output_path, overwrite=overwrite)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_render_html(title, metrics), encoding="utf-8")
    return {"path": str(target), "type": "html_report", "sha256": file_sha256(target)}


def _render_html(title: str, metrics: dict[str, Any]) -> str:
    recording = metrics["recording"]
    channels = metrics["channels"]
    events = metrics["events"]
    data_quality = metrics["data_quality"]
    ica = metrics["ica"]
    recommendations = metrics["recommendations"]
    metrics_json = json.dumps(metrics, indent=2, sort_keys=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      color: #172026;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
      margin: 2rem;
      max-width: 1100px;
    }}
    h1, h2 {{
      line-height: 1.2;
      margin-bottom: 0.5rem;
    }}
    table {{
      border-collapse: collapse;
      margin: 1rem 0 2rem;
      width: 100%;
    }}
    th, td {{
      border-bottom: 1px solid #d8dee4;
      padding: 0.45rem 0.55rem;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f6f8fa;
      font-weight: 650;
    }}
    code, pre {{
      background: #f6f8fa;
      border-radius: 4px;
    }}
    pre {{
      overflow-x: auto;
      padding: 1rem;
    }}
    .pill {{
      border-radius: 999px;
      display: inline-block;
      font-size: 0.85rem;
      font-weight: 650;
      padding: 0.15rem 0.55rem;
    }}
    .info {{ background: #ddf4ff; color: #0969da; }}
    .warning {{ background: #fff8c5; color: #9a6700; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <h2>Recording</h2>
  <table>
    <tr><th>Channels</th><td>{recording["nbchan"]}</td></tr>
    <tr><th>Samples</th><td>{recording["pnts"]}</td></tr>
    <tr><th>Trials</th><td>{recording["trials"]}</td></tr>
    <tr><th>Sampling rate</th><td>{recording["srate_hz"]} Hz</td></tr>
    <tr><th>Duration</th><td>{recording["duration_seconds"]}</td></tr>
  </table>
  <h2>Channels And Events</h2>
  <table>
    <tr><th>Channel locations</th><td>{channels["chanloc_count"]} / {channels["count"]}</td></tr>
    <tr><th>Missing locations</th><td>{channels["missing_location_count"]}</td></tr>
    <tr><th>Events</th><td>{events["count"]}</td></tr>
    <tr><th>Invalid event latencies</th><td>{events["invalid_latency_count"]}</td></tr>
  </table>
  <h2>Data Quality</h2>
  <table>
    <tr><th>Nonfinite samples</th><td>{data_quality["nonfinite_count"]}</td></tr>
    <tr><th>Flat channels</th><td>{data_quality["flat_channel_count"]}</td></tr>
    <tr><th>Median channel RMS</th><td>{data_quality["channel_rms_uv"]["median"]}</td></tr>
  </table>
  <h2>ICA</h2>
  <table>
    <tr><th>Has ICA</th><td>{ica["has_ica"]}</td></tr>
    <tr><th>Components</th><td>{ica["component_count"]}</td></tr>
  </table>
  <h2>Recommendations</h2>
  <table>
    <tr><th>Severity</th><th>Code</th><th>Message</th><th>Suggestion</th></tr>
    {''.join(_recommendation_row(item) for item in recommendations)}
  </table>
  <h2>Machine-Readable QC</h2>
  <pre>{html.escape(metrics_json)}</pre>
</body>
</html>
"""


def _recommendation_row(item: dict[str, Any]) -> str:
    severity = html.escape(str(item.get("severity", "info")))
    code = html.escape(str(item.get("code", "")))
    message = html.escape(str(item.get("message", "")))
    suggestion = html.escape(str(item.get("suggestion", "")))
    return (
        "<tr>"
        f"<td><span class=\"pill {severity}\">{severity}</span></td>"
        f"<td><code>{code}</code></td>"
        f"<td>{message}</td>"
        f"<td>{suggestion}</td>"
        "</tr>"
    )
