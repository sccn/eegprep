# EEGPrep threat model

## Trust boundaries and scope

EEGPrep ports EEGLAB EEG preprocessing workflows to Python APIs, a Qt GUI, a shared console, and a CLI. The separate `packages/eegprep-lean` distribution reads remote NEMAR/Zarr recordings in native Python and browser/Pyodide runtimes.
Treat dataset selection as permission to parse data, not to execute its contents or access unrelated resources.

- **Adversarial:** recording bytes and dimensions; compressed arrays; metadata/history strings; events and channel locations; supplied model files; companion references; dataset filenames, directory trees and symlinks; remote index, metadata and chunk responses.
- **Trusted:** installed code/resources and extensions, executable selection, environment configuration, deliberately typed Python and user-chosen scripts. A YAML job authorizes its declared operations and path arguments, not arbitrary Python execution. Uploading jobs or accepting untrusted subject/task labels inside an authorized output root creates an additional application boundary.
- **In scope:** EEGPrep-owned Python processing, readers/writers, plugins, GUI/console history, CLI/pipelines, lean, and unsafe integration with dependencies. Operator scripts are included where dataset-controlled paths reach them in legitimate use.

Consider researchers loading shared/downloaded datasets from OpenNeuro, NEMAR or collaborators; compromise can expose participant data and credentials under an ordinary account.
Also consider automated pipelines on HPC clusters and applications processing uploads in web services or browser/Pyodide apps. These are embedding scenarios, not upload services EEGPrep ships.
Establish actual job/service permissions and isolation before claiming cross-user or cluster-wide impact; browser claims must respect origin/CORS and virtual-filesystem boundaries.

**Out of scope:** vendored MATLAB EEGLAB under `src/eegprep/eeglab/`; numerical-parity differences from MATLAB; parity harnesses (`tests/matlab*`, `tests/eeglab_tests`, parity fixtures) and parity-only bridge use; deliberate execution of scripts, console input, chosen engine code or installed extensions; DoS proportional to legitimate recordings/computation; dependency-only bugs.
EEGPrep-side flaws reachable through public optional-engine APIs such as `pop_resample(engine=...)` remain in scope. Dataset text promoted into generated executable commands also remains in scope.

## Highest-value investigation targets

Line numbers were checked against develop at `7ac29e10ccda2030ed639f3cc4b3f22aa22a7a1e` and may drift; search for the named function if a line has moved.
These are source-supported targets, not reproduced vulnerabilities. Paths below are relative to `src/eegprep/` except repo-root `pyproject.toml` and paths starting with `packages/`, `scripts/`, `tests/` or `.oss-scanner/`.

### Small inputs driving large allocations

- **SET normalization:** `functions/adminfunc/eeg_checkset.py:240` accepts stored `pnts`/`trials` without reconciling embedded sample dimensions. With events, `trials` drives loops/epoch dictionaries (`:381,402,408`); `pnts` drives `np.linspace` (`:483`). Ordinary `functions/popfunc/pop_loadset.py:123` and `pop_loadset_h5.py:280` reach this. Probe small arrays with inconsistent metadata; compression is unnecessary.
- **Native BDF/EGI:** `functions/sigprocfunc/openbdf.py:95` allocates indices from samples-per-record before payload validation; only record count `-1` triggers file-size inference (`:79`). `readbdf.py:55` allocates before its short-read check (`:65`). `readegi.py:37,110` allocates `arange(count)` before sample reads. Direct BDF APIs differ from `functions/popfunc/pop_biosig.py:48`, which imports through MNE before native status extraction.
- **Containers and copies:** `functions/popfunc/pop_loadset.py:52,86` recursively converts SciPy MAT structures; `pop_loadset_h5.py:62,214,244,263` dereferences and materializes HDF5 datasets/arrays. Generic HDF5 groups (`:130`) are not an unrestricted recursive graph walk. `functions/adminfunc/storage.py:605` reads all FDT samples before size comparison. Include NPZ, atlas compression, Zarr chunks/shards and nested JSON/YAML/AST inputs; quantify amplification.
- Account for existing checks: BrainVision binary `DataPoints` is compared with actual file size (`functions/popfunc/pop_loadbv.py:274,301`); CNT validates header/data layout (`functions/sigprocfunc/loadcnt.py:110`). Investigate bypasses, separate ASCII paths and downstream expansion rather than assuming every declared dimension allocates unchecked.

### Paths, companions and writes

- Non-HDF5 SET sidecars follow string `data`, or `datfile` for `"in set file"`; embedded numeric arrays bypass that branch (`functions/popfunc/pop_loadset.py:198`). `_sidecar_path` (`:229`) preserves absolute references but reduces relative ones to basenames beside the SET: `../x.fdt` does not escape through this helper. Test absolute paths and symlinks separately.
- BrainVision preserves absolute/parent-relative references and tries case-insensitive/same-stem fallbacks (`functions/popfunc/pop_loadbv.py:619`); prove which file opened. STUDY dataset paths accept absolute/existing CWD-relative paths, otherwise resolve beside the study (`functions/studyfunc/pop_loadstudy.py:81`, `_study_utils.py:281`). Headplot can follow a spline's mesh filename (`functions/sigprocfunc/headplot.py:146,259`).
- HDF5 reads have no explicit rejection of external links, external raw storage or virtual datasets (`functions/popfunc/pop_loadset_h5.py:62,214`). Demonstrate unintended access through EEGPrep, not merely use of an HDF5 feature. Deferred reads differ: `functions/adminfunc/eeg_getdatact.py:140` joins paths, while `eeg_checkset.py:275` concatenates strings. HDF5 converts `data` to an array before normalization (`pop_loadset_h5.py:244,263`), so do not assume its string-data branch is reachable.
- **Resave sinks:** `functions/popfunc/pop_saveset.py:541` uses dataset metadata via `functions/adminfunc/storage.py:620`; `functions/studyfunc/pop_savestudy.py:49,105` optionally resaves datasets. Establish the required save action and surviving attacker control: ordinary SET loading replaces `filepath`/`filename` (`pop_loadset.py:91`), STUDY resaving defaults off, and Save As derives its FDT basename from the chosen output (`pop_saveset.py:395`).
- FDT memmaps default to `r+` (`functions/adminfunc/storage.py:612`), but mapping is off by default (`eeg_options.py:23`), checks backing size (`storage.py:380`), and detaches shared backing files before writes (`:390`). Show a reachable mutation and affected bytes; mapping alone is not overwrite.
- **BIDS trees/labels:** inspect inheritance (`plugins/EEG_BIDS/bids.py:162`), derived paths (`:80`) and miscellaneous copies (`bids_preproc.py:74`). Inheritance uses `abspath`, not realpath confinement; copying uses relative destinations and `copyfile`. Export subject/task labels enter paths (`pop_exportbids.py:33,35`); CLI normalization only strips prefixes (`cli/commands/bids.py:332`). Distinguish untrusted labels from explicitly authorized path arguments. Participant tables are written/copied; do not invent participant-field execution.

### Data becoming code or configuration

- **Generated history:** `cli/commands/bids.py:298` inserts a SET filename directly inside single quotes, assigns dataset history, then the import saves the SET and includes history in an optional manifest (`:165,172,299`). Shared `functions/popfunc/_pop_utils.py:37` uses MATLAB-style quote doubling. Probe quotes, backslashes and newlines through actual conversion/replay, naming the necessary victim action.
- Stored `EEG.history`/`STUDY.history` is distinct from session `ALLCOM`. Built-in load/session flows record generated commands, not loaded history (`functions/guifunc/menu_actions.py:480,685`, `session.py:136,259,387`; `functions/adminfunc/console.py:1413`). Console `eegh(n)` replays an `ALLCOM` entry through conversion and `exec` (`console.py:310,601,606,774`). Membership is not sanitization. A stored history string alone is not load-time RCE; injection reached by ordinary replay of generated history is reportable.
- `functions/popfunc/pop_saveh.py:15` writes history text; `pop_runscript.py:18` deliberately executes selected Python and rejects MATLAB/text scripts. Saving and running are separate actions. CLI migration parses text into supported operations (`cli/commands/migrate.py:172`).
- **Restricted expressions:** `functions/popfunc/pop_export.py:99,104,133` validates an AST before restricted `eval`. `functions/miscfunc/make_timewarp.py:175` and `functions/sigprocfunc/erpimage_engine.py:1750` interpret allowed AST nodes. Investigate language escapes and amplification with ordinary data; arbitrary Python objects/callables supplied by trusted code do not establish a dataset attack.
- **Jobs/manifests:** pipeline/batch use `yaml.safe_load` (`cli/commands/pipeline.py:229`, `batch.py:113`); transforms use fixed dispatch (`cli/commands/transforms.py:154`). Check expansion and unintended writes across declared boundaries. Output normalization/resolution is at `pipeline.py:310,559,565`. `cli/core.py:300` resolves artifact paths only for current v2 manifests; other versions return unchanged.
- **Optional engines:** public `functions/popfunc/pop_resample.py:84`, `plugins/ICLabel/iclabel.py:37` and exported `eeglab_clean_artifacts` (`__init__.py:138`) reach `functions/adminfunc/eeglabcompat.py`. Inspect generated statements and temporary-path interpolation (`:190,244,251,278`), distinguishing serialized MAT arguments from source. `clean_artifacts` has its own file workflow (`:530,552`). Prove attacker control beyond trusted `TEMP_DIR`/`TMPDIR`.
- Engine reproduction requires an external checkout selected by `EEGPREP_EEGLAB_ROOT` or a sibling `eeglab` directory (`eeglabcompat.py:111`), plus MATLAB Engine/runtime or Oct2Py/Octave (`:352,362`). The scanner image installs neither runtime; its vendored checkout is not automatically selected by this resolver. These paths remain statically reviewable, not runtime-verified by this image.
- Installed extension registration executes before activation filtering (`extensions.py:536,549,582`); “disabled” is not a security boundary for installed code. The catalog manager renders copyable commands, not an installer (`extension_catalog.py:161`, `functions/adminfunc/plugin_menu.py:30`); assess metadata-to-command consequences without presuming shell quoting validates installer options.
- AMICA uses argument-list subprocesses (`functions/sigprocfunc/runamica.py:417`) and explicit/`AMICA_BINARY`/source-tree/PATH binary selection (`:215`). The optional development binary is under `src/eegprep/bin`; wheels exclude it (`pyproject.toml:235`). By contrast, `scripts/build_derivative.py:27,48,265` interpolates paths into shell commands: establish dataset control in a legitimate launcher workflow.

### Remote data in lean

Lean paths here are relative to `packages/eegprep-lean/src/eegprep_lean/`.

- `index.py:207,219,238` parses remote JSON and accepts `contract_base`/store entries; `store.py:64` builds URLs, `channels.py:130` parses metadata, and `window.py` reads array windows. Investigate index-controlled destinations, schemes, redirects and compressed shapes/chunks.
- `transport.py:136` follows native urllib redirects, passes a timeout and reads the entire body before range checking. `_check_ranged` (`:100`) checks status only, not byte count or `Content-Range`. Browser `pyfetch` also materializes bytes first (`:188`); `PyfetchTransport.timeout_s` is stored but unused (`:202`).
- No explicit scheme/destination policy appears in that index/transport path. Unauthorized native access requires an actual embedding boundary; caller-authorized URLs alone are not SSRF. Test browser timeout/origin behavior separately. The physical-unit formula is matched to a fixed supported string (`window.py:124`), not executed.

## Compact reader map

Paths again use the `src/eegprep/` base. Braces group module filenames; follow wrappers to helpers.

- **SET/MAT/HDF5/FDT:** `functions/popfunc/{pop_loadset,pop_loadset_h5}.py`; deferred samples/ICA companions in `functions/adminfunc/{storage,eeg_checkset,eeg_getdatact}.py`.
- **Text, MAT arrays, NPY/NPZ, float32 and ICA matrices:** `functions/popfunc/{pop_importdata,_file_io,pop_editset,pop_fileio_brainvision_mat}.py`. `_file_io.py:94,96` disables NumPy pickle loading.
- **BrainVision and Neuroscan:** `functions/popfunc/{pop_loadbv,pop_loadcnt}.py`, `functions/sigprocfunc/{loadcnt,loadeeg}.py` (the latter reads epoched EEG). `functions/popfunc/pop_fileio.py:49` dispatches CNT natively; its MNE CNT suffix entry is unreachable through that dispatcher.
- **EDF/BDF/GDF/MFF:** `functions/popfunc/{pop_fileio,pop_biosig}.py` delegates applicable recording reads to MNE with preload; native BioSemi helpers are `functions/sigprocfunc/{openbdf,readbdf}.py`. File-IO `.eeg` means BrainVision; `.vhdr` without `blockrange` uses the native loader.
- **EGI RAW/NetStation MAT:** `functions/popfunc/{pop_readegi,pop_readsegegi,pop_importegimat}.py`, `functions/sigprocfunc/{readegi,readegihdr}.py`.
- **ERPSS RAW/RDF, SMA, BCI ASCII/MAT:** `functions/popfunc/{pop_read_erpss,pop_snapread,pop_loadbci}.py`, `functions/sigprocfunc/{read_erpss,snapread}.py`; BCI support is the implemented ASCII tables/MAT `signal` form.
- **Locations/events:** `functions/sigprocfunc/{readlocs,readegilocs,readneurodat}.py`; `functions/popfunc/{pop_readlocs,pop_importevent,pop_importepoch,pop_importpres,pop_importerplab,pop_editeventfield,_file_io}.py`; group-value text in `functions/studyfunc/pop_importgroupvar.py:67`.
- **BIDS JSON/TSV and trees:** `plugins/EEG_BIDS/{pop_importbids,bids_list_eeg_files,raw,bids,sidecar}.py`, `functions/popfunc/pop_load_frombids.py`. Primary formats are SET/VHDR/EDF/BDF; FDT/VMRK/EEG are companions. Sidecars include inherited EEG JSON, channels, electrodes, coordsystem and events.
- **STUDY/LIMO/manifests:** `functions/studyfunc/pop_loadstudy.py` reads JSON and referenced datasets by default; `{_limo_io,std_readfilelimo}.py` in that directory reads NPZ/JSON, not arbitrary MATLAB LIMO; CLI manifests use `cli/core.py`.
- **Auxiliary MAT/NPZ/atlas files:** `functions/sigprocfunc/headplot.py`, `plugins/EEG_BIDS/montage.py`, `plugins/ICLabel/iclabel_net.py`, `plugins/dipfit/{_fieldtrip_workflows,_mri,load_afni_atlas}.py`. Distinguish caller-selected inputs from normally packaged ICLabel weights/standard MRI.
- **Other binary/text readers:** `functions/sigprocfunc/{floatread,loadtxt,readtxtfile,runamica}.py`, `functions/miscfunc/shortread.py`, `functions/adminfunc/{eeg_readoptions,gethelpvar}.py` (option/help text parsing).

## Exercising the offline container

[OSS Scanner](https://github.com/anthropics/oss-scanner#how-to-sign-up) builds online and audits offline; [guidance](https://red.anthropic.com/oss-scanner/) and [template](https://github.com/anthropics/oss-scanner/blob/main/templates/threat_model.md) inform this scope.
`.oss-scanner/Dockerfile:23,40` installs the checkout at `/src`, Python 3.12 and all root extras/dev dependencies in `/opt/venv` on PATH.
The build **requires** the pinned EEGLAB checkout and initializes it when absent (`:27–37`); it remains out of audit scope.
A separate `/opt/lean-venv` installs `packages/eegprep-lean[zarr,plot]`, `zarr==3.4.0` and pytest (`:44–45`).
The image sets `EEGPREP_SKIP_MATLAB=1`, `QT_QPA_PLATFORM=offscreen`, `UV_OFFLINE=1`, fetches FlankerTest and warms matplotlib/PyBIDS caches. Network isolation is supplied by the runner, not `UV_OFFLINE`.
Use synthetic fixtures/injected transports; avoid live NEMAR/OpenNeuro requests. Lean transport tests use `127.0.0.1` (`packages/eegprep-lean/tests/test_transport.py:93`), available with [Docker `--network none`](https://docs.docker.com/engine/network/drivers/none/).

From `/src`, run targeted tests first, then the relevant suites; do not select `--eeglab-backend`:

```sh
pytest -q tests/test_pop_loadset.py tests/test_pop_loadset_h5.py tests/test_pop_loadbv.py tests/test_bids_load_frombids_helpers.py
pytest -q tests/test_console_workspace.py tests/test_extension_catalog.py
pytest tests
(cd packages/eegprep-lean && /opt/lean-venv/bin/python -m pytest tests -m "not network")
```

Baseline from the same environment with no external network (2026-10-16): root `pytest tests` gave **1919 passed, 140 skipped, 815 deselected, 0 failed** in about two minutes; lean gave **105 passed, 2 skipped, 7 deselected**. The Dockerfile smoke tests turn failures into warnings (`:56–59`), so record the counts you actually get.
Fixture patterns: MAT in `tests/test_pop_loadset.py:18,21`, BrainVision in `tests/test_pop_loadbv.py:24`, BIDS/EDF/BDF in `tests/test_bids_load_frombids_helpers.py:21,144,306`. HDF5 tests include non-MATLAB coverage (`tests/test_pop_loadset_h5.py:65`).

This bounded example creates a two-channel synthetic SET and fresh output directory. Flags/config keys were checked against `cli/main.py:157,189,196` and `cli/commands/pipeline.py:159,310,378`:

```sh
cd /src
eegprep schema pipeline --json
audit_dir="$(mktemp -d /tmp/eegprep-audit.XXXXXX)"
python - "$audit_dir" <<'PY'
import sys
from pathlib import Path
import numpy as np
import yaml
from eegprep import pop_importdata, pop_saveset

root = Path(sys.argv[1])
eeg = pop_importdata(data=np.arange(256, dtype=float).reshape(2, 128), srate=128)
pop_saveset(eeg, root / "input.set")
config = {
    "schema_version": "eegprep.pipeline.v1",
    "input": {"path": "input.set"},
    "output": {"directory": "output", "overwrite": False},
    "steps": [{"name": "rereference", "method": "average"}],
}
(root / "pipeline.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
PY
eegprep inspect "$audit_dir/input.set" --json
eegprep validate "$audit_dir/input.set" --json
eegprep pipeline validate "$audit_dir/pipeline.yaml" --json
eegprep pipeline plan "$audit_dir/pipeline.yaml" --json
eegprep pipeline run "$audit_dir/pipeline.yaml" --json
```

Validation/planning (including `run --dry-run`) do not exercise processing/writes; `pipeline.py:95` returns before dataset loading. The final command above runs the transform and saves outputs.
Keep probes bounded in time/memory; mutate temporary copies, preserve sample originals and never use participant recordings.

## Severity: project triage defaults

These are project triage defaults, not CVSS assignments. Rate demonstrated impact, attacker control, required interaction and actual deployment privileges.

- **Critical:** crafted recording/sidecar/index content causes arbitrary code execution during ordinary loading/processing. Document optional backends or replay interactions; do not imply automatic execution from a stored history string.
- **High:** meaningful unauthorized disclosure or overwrite of unrelated files, such as companion contents exported to an attacker or output escaping into another project; controllable native memory corruption pending stronger evidence; sustained cross-user service loss with a demonstrated embedding/retry model.
- **Medium:** small malformed metadata, compressed objects or responses cause disproportionate memory/CPU/disk consumption or sustained job hangs. Quantify input size, expansion, peak resources and duration; limited unintended file access without demonstrated disclosure/destruction may fit here.
- **Low:** bounded disruption or limited exposure with a demonstrated security consequence. Ordinary exceptions, unsupported formats, numerical differences and recoverable malformed-data rejection are not vulnerabilities.

Calibration only: [MNE-Python reporting policy](https://github.com/mne-tools/mne-python/blob/main/SECURITY.md); [SciPy security boundaries](https://scipy.github.io/devdocs/tutorial/security.html). Neither supplies EEGPrep's severity ratings.

## Reports, PoCs, patches and deduplication

Follow [SECURITY.md](../SECURITY.md): private reports go to **suranganath@ucsd.edu by email only**; GitHub private vulnerability reporting is not enabled. Do not publish findings or contact third parties as part of the audit.
Reports should identify revision/environment, file:line, public entry point, attacker-controlled fields, victim actions, runtime prerequisites and the complete flow to impact. Separate source hypotheses from reproduced observations.
Prefer a short script generating minimal synthetic input, an exact API/CLI invocation, expected/observed results and bounded measurements. Use temporary sentinels for read/write demonstrations; never include participant data, credentials or unrelated files. Reproduce network behavior with controlled responses and state browser/native limitations.
Provide a focused root-cause patch in idiomatic Python and a regression test written before the fix that locks in observable behavior. Do not substitute exception swallowing, permissive fallbacks or other workarounds.
Preserve EEGLAB/MATLAB parity unless the parity behavior itself is the vulnerability; document deliberate security divergence. Keep module layout/import style and update user docs for behavior changes. Comments must explain non-obvious intent/parity, never restate code or justify workarounds.
Never loosen numerical tolerances, edit parity fixtures or weaken assertions/skips. Run narrow relevant tests and `./pre-commit.py`; report unavailable checks and label untested proposals.
Submit **one report per root cause**: shared faulty helper across wrappers means one report with all affected entry points. Similar bugs in unrelated readers remain separate unless they share faulty logic or a necessary fix; shared CWE/format/symptoms alone do not justify merging reports.
