# EEGPrep task tracking

EEGPrep uses [Beads](https://github.com/gastownhall/beads) for project tasks,
dependencies, and handoffs. This setup was verified with **bd 1.2.2**. Its Dolt
database runs **embedded** in the `bd` process: no Dolt server or Python runtime
dependency is required.

## Set up a clone

Install Beads using its [installation instructions](https://github.com/gastownhall/beads#installation).
For a fresh clone without a local task database, run from the repository root:

```bash
bd version
bd bootstrap --dry-run
bd bootstrap
bd where
bd ready
```

`bootstrap` uses the configured `sync.remote` to recover the shared database
from `refs/dolt/data` in `sccn/eegprep`. In an already initialized checkout, use
`bd ready` and the synchronization commands below instead: running `bootstrap`
again with Beads 1.2.2 can fail with "database exists".
If cloning the database fails, fix access to that remote; do not initialize a
replacement database or import an audit log. Fork contributors can read the
shared remote, but publishing changes requires write access to `sccn/eegprep`.

## Work on tasks

```bash
bd prime
bd ready
bd show <id>
bd update <id> --claim
bd create "Describe the work" --type task --description "Scope and acceptance criteria"
bd close <id> --reason "Completed and verified"
```

Git tracks the setup, not the live database or local interaction audit log.
Keep project tasks in Beads; report public bugs through GitHub Issues as usual.

## Synchronize task data

Task synchronization is separate from code commits and `git push`. When
authorized to publish task changes, commit the database's working set first:

```bash
bd dolt commit -m "Update project tasks"
bd dolt pull
bd dolt push
```

Resolve reported conflicts before pushing; do not force-push shared task data.
Neither the Python test suite nor a code push publishes the task database.

## Coding-agent integration

`AGENTS.md` contains the shared task-tracking rules. `CLAUDE.md` imports that
file, so the instructions must not be duplicated there. Codex and Claude
session hooks load Beads context and require `bd` on their process's `PATH`;
restart the application after installing it if necessary.

Git hooks are optional and do not replace explicit Dolt synchronization or
`./pre-commit.py`. Inspect `git config --get core.hooksPath` before enabling
the checked-in shims with `git config --local core.hooksPath .beads/hooks`;
do not replace another hook manager without preserving its checks.
The commit-message shim suppresses Beads' automatic agent-attribution trailer
to respect this repository's commit policy.
