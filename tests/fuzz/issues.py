"""File confirmed fuzz divergences and keep the rolling dashboard issue current."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
import re
import subprocess
import tempfile

logger = logging.getLogger(__name__)

DASHBOARD_TITLE = "Fuzz dashboard"
DASHBOARD_MARKER = "<!-- fuzz-dashboard -->"
SIGNATURE_MARKER = "<!-- fuzz-signature: {signature} -->"
FUNCTION_ROWS = (
    "pop_eegfiltnew",
    "pop_reref",
    "pop_resample",
    "clean_asr",
    "pop_clean_rawdata",
)
_SIGNATURE_RE = re.compile(r"<!-- fuzz-signature: (.+?) -->")
_TREND_RE = re.compile(r"^- (\d{8}): .+$")


@dataclass
class OpenIssue:
    number: int
    title: str
    body: str
    labels: list[str] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)


@dataclass
class PublishResult:
    created: list[int] = field(default_factory=list)
    commented: list[int] = field(default_factory=list)
    unfiled: list[str] = field(default_factory=list)


def is_dashboard(issue: OpenIssue) -> bool:
    return DASHBOARD_MARKER in issue.body or issue.title.strip() == DASHBOARD_TITLE


def signature_of(issue: OpenIssue) -> str | None:
    match = _SIGNATURE_RE.search(issue.body)
    return match.group(1).strip() if match else None


def trend_line(seed: int, functions: dict[str, dict[str, int]]) -> str:
    parts = [f"{name}={functions[name]['divergences']}" for name in FUNCTION_ROWS]
    return f"- {seed}: " + " ".join(parts)


def dashboard_body(
    *,
    seed: int,
    functions: dict[str, dict[str, int]],
    run_url: str,
    harness_ok: bool,
    previous: str,
    unfiled: list[str],
) -> str:
    kept = []
    for line in previous.splitlines():
        stripped = line.strip()
        if _TREND_RE.match(stripped) and not stripped.startswith(f"- {seed}:"):
            kept.append(stripped)
    kept.append(trend_line(seed, functions))
    kept = kept[-30:]
    run = run_url or "local"
    status = "ok" if harness_ok else "sanity failed; no divergence issues filed"
    lines = [
        DASHBOARD_MARKER,
        "",
        "# Fuzz dashboard",
        "",
        "Nightly differential fuzz of eegprep against EEGLAB. EEGLAB output is the oracle.",
        "",
        "## Latest",
        "",
        f"- Seed: {seed}",
        f"- Run: {run}",
        f"- Harness: {status}",
        "",
        "| Function | Examples | Divergences |",
        "| --- | --- | --- |",
    ]
    for name in FUNCTION_ROWS:
        row = functions[name]
        lines.append(f"| {name} | {row['examples']} | {row['divergences']} |")
    if unfiled:
        lines.extend(["", "Unfiled this night (new-issue cap): " + ", ".join(unfiled)])
    lines.extend(["", "## Trend", ""])
    lines.extend(kept)
    lines.append("")
    return "\n".join(lines)


def publish_findings(
    *,
    seed: int,
    functions: dict[str, dict[str, int]],
    findings: list,
    tracker,
    cap: int,
    run_url: str,
    harness_ok: bool,
) -> PublishResult:
    """File confirmed divergences, comment on repeats, and refresh the dashboard.

    ``findings`` are confirmed outcomes ordered by the caller. Sanity failures
    must pass ``harness_ok=False`` and an empty finding list.
    """
    result = PublishResult()
    if harness_ok:
        open_issues = [issue for issue in tracker.list_open() if not is_dashboard(issue)]
        by_signature = {}
        for issue in open_issues:
            signature = signature_of(issue)
            if signature and signature not in by_signature:
                by_signature[signature] = issue
        created = 0
        for outcome in findings:
            signature = outcome.signature
            existing = by_signature.get(signature)
            if existing is not None:
                tracker.comment(existing.number, _comment_body(outcome, seed, run_url))
                result.commented.append(existing.number)
                logger.info("fuzz comment on #%s for %s", existing.number, signature)
                continue
            if created >= cap:
                result.unfiled.append(signature)
                logger.info("fuzz cap left %s unfiled", signature)
                continue
            number = tracker.create(
                _issue_title(outcome), _issue_body(outcome, seed, run_url), ["fuzz", "fuzz-unconfirmed"]
            )
            result.created.append(number)
            logger.info("fuzz issue #%s for %s", number, signature)
            by_signature[signature] = OpenIssue(number, _issue_title(outcome), _issue_body(outcome, seed, run_url))
            created += 1
    previous = ""
    for issue in tracker.list_open():
        if is_dashboard(issue):
            previous = issue.body
            break
    tracker.write_dashboard(
        dashboard_body(
            seed=seed,
            functions=functions,
            run_url=run_url,
            harness_ok=harness_ok,
            previous=previous,
            unfiled=result.unfiled,
        )
    )
    return result


def _issue_title(outcome) -> str:
    return f"[fuzz] {outcome.signature}"


def _issue_body(outcome, seed: int, run_url: str) -> str:
    artifact = f"{run_url}#artifacts" if run_url else "local run (no artifact)"
    params = json.dumps(outcome.params, sort_keys=True)
    return "\n".join(
        [
            SIGNATURE_MARKER.format(signature=outcome.signature),
            "",
            "Confirmed differential fuzz divergence. EEGLAB output is the oracle.",
            "",
            f"- Function: `{outcome.report}`",
            f"- Seed: `{seed}`",
            f"- Params: `{params}`",
            f"- Max abs error: `{outcome.max_abs}`",
            f"- Max rel error: `{outcome.max_rel}`",
            f"- Error class: `{outcome.error_class}`",
            f"- Repro: `{outcome.repro_command}`",
            f"- Artifact: {artifact}",
            "",
            outcome.detail,
            "",
        ]
    )


def _comment_body(outcome, seed: int, run_url: str) -> str:
    run = run_url or "local"
    return "\n".join(
        [
            f"Reproduced on seed {seed}.",
            f"- Max abs error: `{outcome.max_abs}`",
            f"- Max rel error: `{outcome.max_rel}`",
            f"- Repro: `{outcome.repro_command}`",
            f"- Run: {run}",
        ]
    )


class MemoryIssueTracker:
    """In-memory tracker used by harness tests. Same publish path as GitHub."""

    def __init__(self, issues: list[OpenIssue] | None = None):
        self.issues = list(issues or [])
        self._next = 1 + max((issue.number for issue in self.issues), default=0)

    def list_open(self) -> list[OpenIssue]:
        return list(self.issues)

    def create(self, title: str, body: str, labels: list[str]) -> int:
        issue = OpenIssue(self._next, title, body, list(labels))
        self._next += 1
        self.issues.append(issue)
        return issue.number

    def comment(self, number: int, body: str) -> None:
        self._get(number).comments.append(body)

    def write_dashboard(self, body: str) -> None:
        for issue in self.issues:
            if is_dashboard(issue):
                issue.body = body
                return
        self.issues.append(OpenIssue(self._next, DASHBOARD_TITLE, body, ["fuzz"]))
        self._next += 1

    def _get(self, number: int) -> OpenIssue:
        for issue in self.issues:
            if issue.number == number:
                return issue
        raise KeyError(number)


class GithubIssueTracker:
    """GitHub issues via the ``gh`` CLI. ``runner`` exists so tests can stub the process boundary."""

    def __init__(self, repo: str, run_url: str, runner=None):
        self.repo = repo
        self.run_url = run_url
        self._run = runner or _gh

    def list_open(self) -> list[OpenIssue]:
        args = [
            "issue",
            "list",
            "--state",
            "open",
            "--label",
            "fuzz",
            "--limit",
            "200",
            "--json",
            "number,title,body,labels",
        ]
        if self.repo:
            args.extend(["--repo", self.repo])
        payload = json.loads(self._run(args) or "[]")
        issues = []
        for item in payload:
            labels = [label["name"] if isinstance(label, dict) else str(label) for label in item.get("labels", [])]
            issues.append(OpenIssue(int(item["number"]), item.get("title", ""), item.get("body") or "", labels))
        return issues

    def create(self, title: str, body: str, labels: list[str]) -> int:
        path = _write_temp(body)
        args = ["issue", "create", "--title", title, "--body-file", path]
        if self.repo:
            args.extend(["--repo", self.repo])
        for label in labels:
            args.extend(["--label", label])
        stdout = self._run(args)
        return _issue_number(stdout)

    def comment(self, number: int, body: str) -> None:
        path = _write_temp(body)
        args = ["issue", "comment", str(number), "--body-file", path]
        if self.repo:
            args.extend(["--repo", self.repo])
        self._run(args)

    def write_dashboard(self, body: str) -> None:
        existing = next((issue for issue in self.list_open() if is_dashboard(issue)), None)
        path = _write_temp(body)
        if existing is None:
            args = ["issue", "create", "--title", DASHBOARD_TITLE, "--body-file", path, "--label", "fuzz"]
            if self.repo:
                args.extend(["--repo", self.repo])
            stdout = self._run(args)
            number = _issue_number(stdout)
            self._pin(number)
            return
        args = ["issue", "edit", str(existing.number), "--body-file", path]
        if self.repo:
            args.extend(["--repo", self.repo])
        self._run(args)

    def _pin(self, number: int) -> None:
        try:
            args = ["issue", "view", str(number), "--json", "id"]
            if self.repo:
                args.extend(["--repo", self.repo])
            node = json.loads(self._run(args))["id"]
            mutation = "mutation($id: ID!) { pinIssue(input: {issueId: $id}) { issue { number } } }"
            self._run(["api", "graphql", "-f", f"query={mutation}", "-f", f"id={node}"])
        except (RuntimeError, KeyError, json.JSONDecodeError) as exc:
            logger.info("Fuzz dashboard %s stays rolling; pin was not accepted (%s)", number, exc)


def tracker_from_env(run_url: str) -> GithubIssueTracker | None:
    if not (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")):
        return None
    return GithubIssueTracker(os.environ.get("GITHUB_REPOSITORY", ""), run_url)


def _write_temp(body: str) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
    try:
        handle.write(body)
    finally:
        handle.close()
    return handle.name


def _issue_number(stdout: str) -> int:
    match = re.search(r"/issues/(\d+)", stdout)
    if not match:
        raise RuntimeError(f"gh did not return an issue URL: {stdout!r}")
    return int(match.group(1))


def _gh(args: list[str]) -> str:
    result = subprocess.run(["gh", *args], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def default_run_url() -> str:
    explicit = os.environ.get("EEGPREP_FUZZ_RUN_URL", "")
    if explicit:
        return explicit
    server = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if server and repo and run_id:
        return f"{server}/{repo}/actions/runs/{run_id}"
    return ""
