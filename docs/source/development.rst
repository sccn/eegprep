.. _development:

==================
Development Setup
==================

This guide covers setting up a development environment for EEGPrep and contributing to the project.

Prerequisites
=============

System Requirements
-------------------

- **Python**: 3.12 or higher
- **Git**: For version control
- **uv**: Default package and environment manager

Check your Python version:

.. code-block:: bash

    python --version

Required Tools
--------------

- **Git**: `https://git-scm.com/ <https://git-scm.com/>`_
- **Python**: `https://www.python.org/ <https://www.python.org/>`_
- **uv**: `https://docs.astral.sh/uv/ <https://docs.astral.sh/uv/>`_

Optional Tools
--------------

- **Conda**: For environment management (`https://conda.io/ <https://conda.io/>`_)
- **Docker**: For containerized development
- **Make**: For running build commands

Installation from Source
========================

Clone the Repository
--------------------

.. code-block:: bash

    git clone https://github.com/sccn/eegprep.git
    cd eegprep

Create the uv Environment
-------------------------

Install the default development environment:

.. code-block:: bash

    uv python install 3.12
    uv sync --group dev

``uv sync`` creates ``.venv/`` and installs EEGPrep in editable mode from the
locked dependency set. The development environment includes the GUI and
``eegprep-console`` runtime dependencies so ``uv run eegprep-console --full``
works from a fresh checkout. Use ``uv run`` for commands so they execute inside
this environment.

Install Documentation Dependencies
----------------------------------

.. code-block:: bash

    uv sync --extra docs --group dev

This installs:

- The eegprep package in editable mode
- Development dependencies used by repo tooling
- Documentation dependencies

Running Tests
=============

Porting The Current EEGLAB Tests
--------------------------------

Ports of the upstream MATLAB tests use only the current
`sccn/eeglab_tests <https://github.com/sccn/eeglab_tests>`_ repository. The
older ``sccn/eeglab-testcases`` repository is stale and must not be used.
EEGPrep currently pins ``eeglab_tests`` commit
``ff605546f3f70868916fb8d49c007472b3257b50`` and the EEGLAB submodule commit
``8ac485f654d6bbb1a6acb8dc9ef3f2eaf3d409ba``.

Translate the behavior and assertions of each MATLAB scenario into the closest
existing pytest module. Decorate the Python test with its upstream path and
test name so coverage remains traceable without a separate conversion matrix:

.. code-block:: python

    from tests.eeglab_tests import eeglab_test

    @eeglab_test("regression_tests/t_pop_selectevent.m", "testRetainsMatchingEpochs")
    def test_pop_selectevent_retains_matching_epochs():
        ...

One Python test may carry more than one decorator when it genuinely covers
multiple equivalent upstream scenarios. Do not combine tests merely to reduce
the number of ports. Preserve input shapes, dtypes, empty values, indexing,
warnings, errors, and all scientifically relevant output fields. Use a live
MATLAB comparison when practical or small expected data generated from the
pinned suite when ordinary CI must run without MATLAB.

When a faithful port exposes missing behavior or a defect, keep the failing
scenario visible and record it in Beads without changing runtime capabilities
as part of the test port. Excluding pure MATLAB runtime behavior requires an
explicitly agreed scope decision and a concrete technical rationale.
Never replace an applicable assertion with a no-crash smoke test or broaden a
numerical tolerance simply to make the port pass.

Before declaring the port complete, clone the current suite at its pinned
commit and run the source-driven audit:

.. code-block:: bash

    git clone https://github.com/sccn/eeglab_tests.git /tmp/eeglab_tests
    git -C /tmp/eeglab_tests checkout ff605546f3f70868916fb8d49c007472b3257b50
    git -C /tmp/eeglab_tests submodule update --init eeglab
    uv run python -m tools.eeglab_test_port_audit /tmp/eeglab_tests

The command discovers wrapper and regression methods directly from MATLAB,
collects ``eeglab_test`` metadata through pytest, and prints every missing or
stale reference. Pass ``--json`` for automation. It deliberately rejects the
old ``eeglab-testcases`` repository, a checkout at another commit, and
provenance that does not exist in the pinned suite.

Native MATLAB statement coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tools/eeglab_statement_coverage.py`` freezes the approved source denominator
independently of test selection. It includes root/core MATLAB files, private and
class helpers, GUI/admin functions, the named workflow plugins and importers.
Both installed BIDS trees are included. Dependency internals (Manopt,
MatConvNet, JSONio, LIMO external code and FieldTrip), native/opaque binaries,
and other installed plugins are hashed and reported separately, not credited
as covered. Downloaded plugin directories have content provenance rather than
an invented Git revision. Missing required plugins or any subsequent source
addition, edit or deletion invalidate the frozen inventory.

For an initial, explicitly partial headless measurement, copy the native test
sources into scratch and select existing suites that do not require datasets.
Use a separate writable copy of the entire EEGLAB tree, including sample data;
its source hashes must match the frozen denominator. Never use hard links or
writable data symlinks to the reference. APFS ``cp -cR`` can clone these files
efficiently on macOS; elsewhere use an ordinary recursive copy. Do not put the
runtime copy at a different path from the one supplied to the runner. Prefer
the real copy at ``native-tests/eeglab``: native regression fixtures expect
this original relative layout. If that directory exists, the driver requires
it to resolve to the verified runtime root, not another EEGLAB tree:

.. code-block:: bash

    uv run python -m tools.eeglab_statement_coverage freeze \
      --suite-root /path/to/eeglab_tests --output /scratch/scope.json
    rsync -a --exclude /eeglab --exclude .git --include '*/' --include '*.m' \
      --exclude '*' /path/to/eeglab_tests/ /scratch/native-tests/
    cp -cR /path/to/eeglab_tests/eeglab /scratch/native-tests/eeglab
    uv run python -m tools.eeglab_statement_coverage run \
      --manifest /scratch/scope.json --test-root /scratch/native-tests \
      --runtime-root /scratch/native-tests/eeglab \
      --test-file unittesting_sigprocfunc/epoch/sigprocfunc_epoch_wrapperTest.m \
      --output /scratch/coverage-run --matlab /path/to/matlab --timeout 1800

The output directory must be new. Add repeated ``--test-file`` and
``--support-path`` arguments explicitly. Dataset workflows require their real
data copied into scratch as well; this command does not download dependencies,
rewrite source tests, approve corrections or shorten workloads. Original
suite sources must remain byte-identical, so approved LIMO source overlays
need an explicitly recorded overlay boundary before joining this native run.
Keep scratch tests projectless: omit only ``*.prj``, ``resources/project`` and
legacy ``.SimulinkProject`` metadata when copying full datasets. MATLAB R2026a
automatically attaches a ``ProjectFixture`` for project members; the upstream
project startup adds unrelated fixture paths and runs ``add_plugins``, which
can install unpinned plugins. Controlled initialization here replaces that
project startup, not any scientific source test or dataset. The driver rejects
such project metadata before MATLAB starts.

For newly authored native tests, add ``--additional-test-file`` for each
self-contained MATLAB function-test file. These are EEGPrep-owned additions,
not tests attributed to the upstream suite's pinned commit. The driver copies
their exact bytes into the new output directory before launching MATLAB and
records original paths, snapshot paths and SHA-256 hashes in ``run.json``.
Subsequent edits in the working checkout cannot change an in-flight test;
snapshot changes during execution are rejected. Original test-source hashes
and the implementation denominator remain unchanged. Fixtures are resolved
from the scratch suite working directory; use local functions for new helpers.
An additions-only run can omit ``--test-file``:

.. code-block:: bash

    uv run python -m tools.eeglab_statement_coverage run \
      --manifest /scratch/scope.json --test-root /scratch/native-tests \
      --runtime-root /scratch/native-tests/eeglab \
      --additional-test-file tests/matlab/expanded/test_eegprep_structures_expanded.m \
      --output /scratch/added-tests-run --matlab /path/to/matlab --timeout 1800

Run original selections and additions together for a combined statement union,
or pass ``--baseline-run /scratch/completed-run`` to extend a retained native
measurement. The latter requires identical frozen source/test manifests, the
same runtime path and MATLAB release, and a completed baseline (not a timeout).
It hashes the retained config, report and coverage MAT file before execution.
MATLAB's native ``matlab.coverage.Result`` ``+`` operator forms the statement
union; overlapping statements are counted once. The report retains the current
batch's cases separately, links its baseline and reports newly covered
statements. A failed baseline remains a failed validation even when all new
cases pass. Never add independently measured covered-statement totals.
The driver checks both reference and runtime source inventories before and
after execution. Existing home options are copied into the test-local options
directory; the pinned ``icadefs`` script is wrapped only to redirect
``EEGOPTION_PATH`` there. The home options hash must remain unchanged.

The native runner follows ``eeglab_tests/example_local_test.m`` using the public
``TestRunner`` and ``CodeCoveragePlugin.forFile`` with ``MetricLevel='statement'``.
``CoverageResult.Result.coverageSummary(..., 'statement')`` supplies executed
and executable statement counts; this is not Cobertura's line-rate metric.
The live harness regression proves two statements on one line are counted
separately and an entirely uncalled file remains in the denominator. Function
counts and native statement locations/hit counts are retained too. Decision
coverage is explicitly not collected. Parallel-worker hits are not yet proven
to be collected; treat this as client-side coverage, not full parallel coverage.
MATLAB R2026a can return invalid source files as 0/0 instead of omitting them.
The runner explicitly reads ``matlab.coverage.Result.Invalid`` (public getter,
hidden property in the installed R2026a API) and marks these files unmeasurable.
They remain in the manifest; measured statement totals are not a complete
denominator and no full-scope percentage is valid while such files remain.

For the approved expansion target, use
``--approved-unmeasurable tools/eeglab_coverage_unmeasurable.json``. On
2026-09-28 the user approved measuring the 90% target over all 1,942 measurable
files, with 21 pinned BioSig files retained as unmeasurable compatibility gaps.
That approval file records each exact source hash and the MATLAB release; it
does not mark these files covered or remove them from the 1,963-file inventory.
``denominator_complete`` still describes the full inventory, while
``approved_denominator_complete`` describes the approved measurable target.
Unexpected invalid files or missing coverage results continue to fail the run.
The measured baseline is 31,951/197,022 statements (16.217%); accepting the
compatibility gaps does not improve that number.

``run.json`` records inputs and source hashes; the exact native runner is copied
beside it. All selected native suite files form one batch, avoiding repeated
full-scope instrumentation/report construction. ``report.json``, ``coverage.mat``
and ``results.xml`` retain actual results, including failures, before acceptance
checks. The report distinguishes requested files, discovered native case names,
executed cases, unselected files and unmeasurable source files. ``driver.json``
and ``matlab.log`` retain timeout/error status. A timeout leaves the selected
batch incomplete; partial-batch coverage is not recovered or claimed.
The timeout kills only this runner's process group. A failing source test does
not prevent measurement; it still prevents claiming successful validation.
Neither a partial run nor this instrumentation establishes 90% coverage.

Source fidelity and graphical validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Source-reference counts are not execution results or scientific coverage.
Backend-dispatched contracts must still be checked against the actual source
body and executed against the pinned MATLAB reference. Keep generated-data
Python regression tests as separate supplements; they are not substitutes for
the source recording, workload, call sequence, or assertions. A source wrapper
whose body is entirely commented out or immediately returns is inactive, not a
successful behavioral test. Such definitions remain visible in the inventory
without attaching misleading provenance to a different Python scenario.

Mark every contract that creates plots, windows, dialogs, or graphical
callbacks with ``gui``, even when MATLAB figures would be invisible. This
includes progress bars in numerical helpers such as ``eeg_context`` and
browser-opening paths in either backend, even if the other backend is non-GUI.
The marker allows explicitly deferred graphical validation to remain excluded using
``-m "not gui"``. Collecting those contracts or checking their syntax does not
validate their runtime behavior. Modal Python dialogs may require interaction;
do not substitute a mock renderer or invent accept/cancel actions to make an
original interactive workflow automatic.

MATLAB-specific graphics objects, caller workspaces, and object arrays sometimes
require small test-only native boundaries. Keep the workflow and assertions in
Python and dispatch actual EEGLAB processing functions. Translate ordinary
language operations such as file deletion or closing a figure using real
Python APIs, not nonexistent EEGPrep counterparts to MATLAB builtins. Do not
replace a missing operation with a fabricated failure or a passing assertion
that the feature is absent.

Preserve the limitations of the source oracle too: a no-assertion smoke test
does not establish numerical correctness, and a legacy wrapper that ignores a
returned failure status does not establish that its operation succeeded.
Record source failures and unresolved language-specific observables separately
instead of weakening comparisons or counting a native-only diagnostic as a
completed Python port.

MATLAB tables nested in reference outputs, such as FieldTrip ``trialinfo``,
cross the test bridge as MATLAB-written MAT bytes and are restored to native
tables before the next call. This preserves variable classes, dimensions and
table properties; it does not replace a table with a lossy struct or omit its
metadata. The transport regression compares the real FieldTrip preprocessing
chain with direct native execution using the original epoched sample dataset.

Large outputs retain their complete arrays. When the aggregate output container
approaches MAT v7's 2 GB variable limit, the test bridge writes numerical leaves
to per-call sidecars and reassembles their original classes and dimensions
before deleting the temporary directory. The slow transport regression sends
more than 2 GiB through the bridge and back to MATLAB for exact comparison.
Large Python inputs use the same sidecars before SciPy's MAT5 writer reaches
its container or individual-matrix size limit. Buffered Fortran-order chunks
preserve complete values, classes and dimensions without another full-array
copy. Separate slow checks exercise both a nested aggregate and a single
matrix larger than 4 GiB against native MATLAB assertions.
This is test transport, not subject selection, downsampling or a production
dataset serializer.

A user-approved Python-only exception applies to MATLAB caller-workspace and
copy-count introspection in ``checkmmo`` and the nested helpers of ``checkmmo2``.
Actual construction, copying, writes, data-integrity checks and debug-output
assertions remain in scope. MATLAB retains the original count assertions.
Because ``checkmmo``'s source oracle consists entirely of count assertions, its
Python result records execution only, not count or scientific parity. The
audit reports these exceptions separately without removing source definitions
or treating unresolved diagnostic failures as passes.

Explicit MATLAB support paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Backend-neutral contracts run with ``--eeglab-backend=matlab`` and an explicit
``--eeglab-root``. The automatically assigned ``eeglab_contract`` marker selects
tests using the backend fixture, excluding ordinary direct-Python regressions
and transport-only harness checks. For a non-graphical, non-slow contract run:

.. code-block:: bash

    uv run --no-sync pytest tests --eeglab-backend=matlab \
      --eeglab-root=/absolute/path/to/eeglab_tests/eeglab \
      -m "eeglab_contract and not gui and not slow" --junitxml=contracts.xml

This is a selected run, not full acceptance: graphical and slow contracts remain
unvalidated, and original prerequisite guards may return without exercising
their scientific branches. Report those limits separately from pass counts.
The guarded statistical and legacy ERSP contracts record
``eeglab_source_body_entered`` in each JUnit test case's ``properties``; a
``False`` value means the original prerequisite/version guard returned before
the scientific body. The clustering contract separately records
``eeglab_optional_kmeans_branch_entered``, since its first clustering operation
still runs when the optional branch is unavailable. These properties do not
change the source guards, assertions, or pytest outcomes. A passing case with a
false guard property is not evidence that the guarded behavior was validated.

For the full contract selection, omit the ``not gui`` and ``not slow`` filters.
MATLAB windows, progress dialogs, video rendering, full-recording ICA and
multi-subject bootstraps can run for substantially longer than the numerical
subset. Run deliberately on a machine where graphical interaction is allowed:

.. code-block:: bash

    uv run --no-sync pytest tests --eeglab-backend=matlab \
      --eeglab-root=/absolute/path/to/eeglab_tests/eeglab \
      -m eeglab_contract --junitxml=all-contracts.xml

For long or concurrent runs, supply a distinct ``--basetemp`` directory for
each run if generated datasets and model files must be retained. Pytest's
default temporary-directory retention can remove earlier runs while another
process is still using their evidence. Use a new directory: pytest clears an
existing ``--basetemp`` before running.

Known reference findings must stay visible in that report. In the pinned
suite, the three legacy ``erpimage`` leaves ``pass_general``, ``pass_many_args``
and ``pass_times`` return ``tc_notpassed`` on MATLAB R2026a: they compare a
numeric handle vector with modern cell outputs. Their original wrappers
discard the returned status. A green wrapper therefore does not validate
those predicates. The Python contracts retain the comparisons and fail;
the graphics transport only removes handles the source never compares.

The original ``statcond`` cases 9, 10, 13 and 14 also fail directly in native
MATLAB R2026a with their stored zero absolute/relative tolerances. Their
maximum p-value residuals are approximately ``6.07e-18``, ``2.78e-17``,
``7.81e-18`` and ``1.73e-17``; the direct fixture run passed 74 of 78 cases,
matching the hybrid failures. These are retained reference fixture/platform
discrepancies, not established EEGPrep defects. The original tolerances and
failing assertions remain unchanged.

The original ``std_topoplot`` workflow requires ``.icatopo`` files generated
by the suite's earlier ``std_precomp`` workflow. A fresh source data checkout
does not contain them. Its independent port creates the real scalp cache in
a writable copy of the original STUDY, then reloads the original STUDY before
the unchanged four plotting calls. No synthetic cache or reference-tree write
is used.

The test session initializes the pinned EEGLAB checkout and
its already installed plugins without installing anything. Optional upstream
dependency files can live outside that checkout: pass their directory using
``--eeglab-support-path PATH``. The directory must exist. This option is
repeatable; each directory is added at the beginning of the session's MATLAB
path after EEGLAB plugin initialization, so the last directory takes precedence
over earlier support directories and installed plugins. Subdirectories are not
added recursively. The session quits its engine on teardown; it never calls
``savepath`` or changes the package runtime, reference source, or user options.

Two upstream packaging gaps encountered with the pinned suite have been
validated using source-exact files in an ignored local directory:

* JSONio bundled with ``bids-matlab-tools8.0``/``EEG-BIDS`` lacks an Apple
  Silicon MEX. Official JSONio commit
  ``e6c5b3ea16142e8e428aa254fc042dfad6011c30`` adds
  ``jsonread.mexmaca64`` (Git blob
  ``396b1d429c4887446c1c0a20b4e6f457942abe1b``). The bundled ``jsonread.c``
  has blob ``fa7902c4a5bbb74ddd02e76ed423a781602b33b6``, identical to the
  preceding official revision ``82d835d17348b0d060e8af881da308b897a627ff``.
  The Apple Silicon commit changes only that C file's header comment, not
  parser logic; ``jsmn.c`` and ``jsmn.h`` are unchanged. This binary is for
  native Apple Silicon MATLAB only.
* ``Fileio260210/private/ft_datatype_sens.m`` calls the absent
  ``ft_deleteopt``. That caller's blob
  ``c66bdfebf2528518981ef75dc257d5b15c9da403`` exactly matches official
  FieldTrip revision ``efdd7db8bf5623e580c8f12f81bb57b325662e29``.
  The matching revision's ``utilities/ft_deleteopt.m`` has blob
  ``2eebd35ca862eac7001bb3d01e49250501e849ff``. Use that original helper,
  not a replacement implementation. Its ``removefields`` dependency is
  already supplied by the installed FieldTrip utilities.

For that Apple Silicon setup, download the two pinned files explicitly from
their official repositories. From the main EEGPrep checkout (not a temporary
worktree), run:

.. code-block:: bash

    mkdir -p .notes/reference/matlab-test-dependencies
    curl --fail --location https://raw.githubusercontent.com/gllmflndn/JSONio/e6c5b3ea16142e8e428aa254fc042dfad6011c30/jsonread.mexmaca64 \
      --output .notes/reference/matlab-test-dependencies/jsonread.mexmaca64
    curl --fail --location https://raw.githubusercontent.com/fieldtrip/fieldtrip/efdd7db8bf5623e580c8f12f81bb57b325662e29/utilities/ft_deleteopt.m \
      --output .notes/reference/matlab-test-dependencies/ft_deleteopt.m
    git hash-object .notes/reference/matlab-test-dependencies/jsonread.mexmaca64
    git hash-object .notes/reference/matlab-test-dependencies/ft_deleteopt.m

Check that the two printed hashes match the blobs above before execution.
The legacy BIDS importer prepends its own ``JSONio`` folder during import,
and starting EEGLAB again also reorders plugin paths. For full BIDS/LIMO
workflows on Apple Silicon, install that same verified binary alongside
``jsonread.m`` in each active BIDS plugin's ``JSONio`` directory (both
``EEG-BIDS`` and ``bids-matlab-tools8.0`` if both are present). A symlink to
the verified support file is sufficient. This is an explicit local native
dependency installation, not a MATLAB-source correction; do not overwrite
source files or change the pinned revision. The standalone support path alone
is insufficient once the importer changes path precedence.

The STUDY-statistics and advanced source-reconstruction tutorials require the
full FieldTrip distribution. Fileio also bundles a partial FieldTrip whose
``ft_defaults`` can shadow the full plugin and look for ``statfun``/``preproc``
modules in the wrong directory. These two contracts temporarily prioritize
the full plugin owning ``ft_freqstatistics`` after EEGLAB startup, clear its
cached initialization and invoke its actual ``ft_defaults``. They restore the
previous path afterward; no scientific function or option is substituted.

Full LIMO validation also requires Parallel Computing Toolbox (the reference
calls ``gcp`` during cleanup) and Image Processing Toolbox for ``bwlabeln`` in
cluster analysis, or the reference's supported SPM alternative. A license
entitlement alone does not install either toolbox. Record installed versions
alongside the reference revisions; do not patch LIMO to ignore missing tools.

No downloads occur automatically in pytest. Run the non-GUI parser/BIDS
metadata smoke and the original Fileio contract with normal test selection:

.. code-block:: bash

    uv run --no-sync pytest \
      tests/test_eeglab_hybrid.py::test_matlab_bids_metadata_loaders \
      tests/test_sample_data_pop_functions.py::test_upstream_pop_fileio_original_sample_options \
      --eeglab-backend=matlab \
      --eeglab-root=/absolute/path/to/eeglab_tests/eeglab \
      --eeglab-suite-root=/absolute/path/to/eeglab_tests \
      --eeglab-support-path=/absolute/path/to/eegprep/.notes/reference/matlab-test-dependencies

The metadata smoke calls the actual JSONio parser and original BIDS loaders on
``ds002718`` metadata. The Fileio source contract only requires its three
original calls to complete; it does not establish numerical or channel-location
correctness, and existing import warnings remain visible. These checks do not
validate LIMO preprocessing, bootstraps, or GUI workflows.

Tutorial-wrapper provenance
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tutorial-wrapper ports additionally pin
``sccn/eeglab-tutorial-scripts`` commit
``58bf12dd53e894dd3ee1285946563cd94999db16``. The four tutorial references
``plot_study_erp``, ``source_reconstruction_advanced``,
``source_reconstruction_eeg``, and ``time_freq_all_elec`` are MATLAB Live
Scripts (``.mlx``) in that commit; they are not missing source files. MATLAB
can execute them, while Octave cannot execute the Live Script format.

The ten active wrapper contracts in ``tests/test_tutorial_eeglab_tests.py``
retain the original scripts' operation sequences, arguments, and full inputs:
the continuous and ICA-epoched sample recordings, five-subject N400 study,
and P300 BIDS tree (with the original subjects 1--2 selection). Workflows that
write datasets, measure caches, spline files, or videos use isolated working
directories and complete copies of the source input trees when needed. The
original FieldTrip, DIPFIT, PICARD, and ICLabel dependencies remain required;
missing Python capabilities are not converted into successful skips or
expected failures. Python owns the workflow sequence, with small test-only
boundaries for native caller workspaces, function handles, and video objects.

All ten active tutorials start the EEGLAB GUI or create figures and are
marked ``gui``. They require explicit graphical validation; ordinary
generated-data regression results do not validate these source contracts.
Future Python video-export runs also require ``ffmpeg`` for the equivalent
30-fps MPEG-4 or uncompressed AVI writer.

``tutorial2_wrapperTest.test_bids_process_face_experiment`` is inactive because
its entire wrapper body is commented out. It carries no source provenance and
is not represented by a no-op passing test. The existing face-recognition and
other generated-data workflows remain supplemental Python regressions, without
claims that their synthetic recordings are the original source inputs.

The original ``event_processing_study`` script leaves ``EEG`` unchanged after
editing and reloading ``ALLEEG``. On the pinned reference, its final
``eeglab redraw`` therefore opens ``pop_newset``'s dataset-change dialog.
The source specifies no answer. A manual run that selects Cancel exercises
that cancellation path but does not prove an unattended workflow or assert
the resulting STUDY contents. Do not silently overwrite a dataset, synchronize
the variables, or invent a dialog response inside the port.

The movie tutorial requires visible MATLAB figures for native frame capture.
On R2026a, the original 2-D section produces inconsistent frame dimensions
under the harness's hidden-figure default, but completes with 91 equal-sized
frames using the original visible default. Its port temporarily enables
visible figures for this capture workflow and restores the previous setting;
it does not resize frames, change plotting options or patch the source.
Native tutorial figure creation also finishes pending GUI activation and
restores the created figure as MATLAB's current target before the next Engine
call. Otherwise R2026a can reactivate an older docked figure between calls,
causing later plots to reuse its colorbar axes. This test-only boundary keeps
the source's figure selection without changing docking or plotting options.

LIMO preprocessing source correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The separately approved ``limo_preproc_stats_hw.m`` correction captures
``pop_limo``'s existing third output and resolves the adjacency and Beta/LIMO/
contrast text-list paths from the generated model directory. The pinned
reference writes these under ``derivatives2/derivatives`` with a study-prefixed
GLM name; the original script expects the older paths. The correction preserves
all 18 subjects, preprocessing options, text-list inputs, 1,000 second-level
bootstraps, calculations and plots.

``prepare_limo_preprocessing_source_overlay`` in
``tests.test_limo_eeglab_tests`` verifies the original SHA-256 and applies
``tests/matlab/limo_preproc_stats_hw.source-corrections.patch`` only to a scratch
copy. Its integrity regression reverses the approved edits byte-for-byte and
checks that the pinned source remains unchanged. The Python-owned workflow uses
the same path expressions without evaluating the complete native script.
Ordinary integrity checks do not establish full live validation.

The port also publishes the current ``STUDY`` before ``pop_limo`` and the
original three-argument contrast call. Native ``limo_settings_script`` reads
base ``STUDY`` even when its caller has a valid local value; the earlier redraw
precedes the script's new design. These explicit assignments reproduce the
script's current workspace instead of supplying its stale pre-design copy.
The Python-owned workflow snapshots/restores base/global state, and the scratch
overlay's inverse removes both assignments to recover the pinned source.

The pinned preprocessing script is interactive even after all input paths are
resolved: ``limo_add_plots`` plots its supplied files, then asks for another
central-tendency file until Cancel is selected. The first such dialog follows
the three unweighted ERP files. Its port and scratch source overlay now use
``eegprep_test_limo_add_plots``: the original native plotting function executes
with every supplied file and unchanged options, while a scoped file-input
fixture declines exactly one optional next-file request. A different prompt,
repeated request or missing expected request fails. The original path and file
chooser are restored even on failure. Live checks assert the actual plotted
values and confidence-interval patches, and reject a required metadata chooser.
This is explicit scripted file-input cancellation, not validation of clicking
the native chooser. No extra file, computed result or plotting implementation
is fabricated. The overlay's inverse also removes these eight wrapper names
and recovers the original source byte-for-byte.

The subsequent original per-subject plotting call exposes a separate pinned
LIMO failure: its ``'variable'`` parser assigns the numeric subject index to
``infile`` instead of retaining the supplied file list. ``load(file)`` then
fails because the index is not a filename. Both the direct native call and
the test wrapper reproduce this with the retained real six-file ERP inputs.
The wrapper does not change this parser, suppress the error or omit the
18-subject plotting loop. Successful optional-file cancellation therefore
does not establish full preprocessing-workflow success.

LIMO integration source correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The companion ``limo_test_integration.m`` overlay applies five separately
approved path-only corrections: the generated derivative/list directory,
study-prefixed GLM names, ``Gp-1`` group spelling, grouped contrast basenames
from the first subject's returned contrast files, and the second-level
``H0/Betas_desc-H0.mat`` filename. The Python workflow and native overlay use
the same generated model paths. Original text-list, cell and ragged-array
inputs, all 18 subjects, OLS/WLS models, nine status sections, 101 bootstraps,
statistical options and assertions remain unchanged. Historical cleanup is not
corrected by this overlay.

A separately authorized workspace-setup correction publishes the current local
``STUDY`` into MATLAB's base workspace immediately before the original
two ``pop_limo`` model calls and the three-argument WLS
``limo_batch('contrast only', [], contrast)`` call. Native
function-scope reproduction with all 18 retained models confirmed that the
earlier no-argument ``eeglab`` call clears base/global ``STUDY`` and that LIMO's
implicit discovery reads base, not the caller's valid local variable. The
``limo_settings_script`` called from ``limo_batch`` also overwrites an explicitly
passed local ``STUDY`` from base before metadata export, requiring the same setup
before both model calls. After the original post-load 6/7/5 group assignment,
the overlay also refreshes only ``STUDY.group`` using the original
``std_checkset`` expression ``unique_bc({STUDY.datasetinfo.group})``. The load
checked this summary before the manual assignments, retaining an empty label; native
LIMO requires it to emit grouped text lists. Dataset membership and order are
unchanged. The three-argument discovery path remains exercised; no fourth
argument or runtime repair is substituted. The native overlay uses the existing state-only
``eegprep_test_base_workspace`` helper with ``onCleanup``; the Python-owned
MATLAB workflow uses the same snapshot/restore helper and a pytest finalizer.
Both preserve the prior base/global workspace even on failure. The native
snapshot helper remains locked until restoration because EEGLAB initialization
executes ``clear functions``, which otherwise discards its saved state. The native
overlay requires ``tests/matlab`` on the path, as supplied by the hybrid harness.

``prepare_limo_integration_source_overlay`` in ``tests.test_limo_eeglab_tests``
checks the original SHA-256 before applying the exact substrings and occurrence
counts in ``tests/matlab/limo_test_integration.source-corrections.json`` to a
scratch copy. JSON preserves literal trailing spaces on affected source lines
that the required whitespace checks would remove from a unified patch. An
independent inverse regression recovers the pinned source byte-for-byte; no
reference checkout is edited. Full live validation remains a separate check.

Standalone IRLS source correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The standalone IRLS contract translates ``limo_zIRLS_validation_4_Arno.m``
and its full ``limo_test_glmboot`` helper into Python-owned workflows and
bootstrap-analysis loops. It retains all subjects, cleaning/PICARD/ICLabel/ASR,
ERP/spectrum/ERSP/ITC precomputation, 2,500 null bootstraps per subject, the
1,000-test/1,500-null split, 300-step convergence analysis, six result fields,
``results.mat`` and native ``.fig`` exports. Collection and ordinary helper
tests do not establish live validation of this full workflow.

The approved corrections are retained separately in
``tests/matlab/limo_zIRLS_validation_4_Arno.source-corrections.patch``. It assigns the
otherwise undefined ``chanlocs`` variable to
``fullfile(STUDY.filepath, 'derivatives', 'limo_gp_level_chanlocs.mat')``,
which the reference ``std_limo`` generates. It does not invent adjacency data.
It also captures ``pop_limo``'s third output, loads each model from
``LIMOfiles.mat{s}``, and derives its ``H0`` directory from that returned path.
This uses the generated model paths without duplicating version-dependent
directory names. Scientific options, subjects and bootstrap counts are unchanged.
The third approved correction changes only the validation helper's discovery
pattern from ``H0_*.mat`` to ``*H0.mat``, matching LIMO 4.1.2's generated
``sub-*_desc-Condition_effect_1H0.mat`` files. Existing ``Betas``, ``tfce`` and
``R2`` exclusions, variable selection and statistical calculations are unchanged.
``prepare_irls_source_overlay`` in ``tests.eeglab_tests.limo_irls`` verifies
the original script/helper hashes and applies the patch only to a scratch
copy. Its regression reverses all approved edits to recover both pinned files
byte-for-byte. A corrected native run needs a writable ``ds002718`` copy beside the
returned ``unittesting_limo`` directory. The Python contract never executes
the complete native test script or its helper.

Other source behavior remains unchanged. Full live IRLS validation is pending.
Missing Python LIMO primitives and unsupported MATLAB ``.fig`` export remain
visible failures, not replacement algorithms or differently formatted files.

Test Discovery
--------------

Tests are located in the ``tests/`` directory. Run all tests:

.. code-block:: bash

    uv run pytest tests

Run specific test file:

.. code-block:: bash

    uv run pytest tests/test_clean_artifacts.py

Run specific test function:

.. code-block:: bash

    uv run pytest tests/test_clean_artifacts.py::TestClassName::test_method_name

Run a marker subset:

.. code-block:: bash

    uv run pytest -m "not slow"

Markers include ``slow``, ``matlab``, ``octave``, ``gui``, ``visual``, and
``parity``. Legacy ``unittest`` tests are categorized during collection in
``tests/conftest.py`` so marker expressions work without rewriting the tests.

Continuous Integration
----------------------

Tests run automatically on:

- Every push to a branch
- Every pull request
- Scheduled nightly runs

Check CI status on GitHub Actions.

Full-Pipeline Parity Harness
----------------------------

The ``tests/`` suite covers parity function by function. End-to-end parity — does
a complete preprocessing pipeline in EEGPrep reproduce EEGLAB numerically, step
by step, on real data? — lives in a **separate repository**:
`sccn/eegprep_parity_test <https://github.com/sccn/eegprep_parity_test>`_.

It runs the same seven-step pipeline twice, once in EEGPrep and once in
MATLAB/EEGLAB through the ``eeglabcompat`` MATLAB Engine bridge, then reports the
per-step difference for every subject:

1. Import (BIDS to EEG, select EEG channels)
2. Average re-reference
3. ``clean_rawdata`` (ASR)
4. Picard ICA (identity init, deterministic)
5. ICLabel and component removal
6. Spherical-spline interpolation
7. Epoch and baseline

For each step it records ``max_abs_diff`` and ``rms_diff`` in microvolts on the
common channels and minimum length, plus ICA decomposition parity (AMARI
distance, component correlation) and ICLabel rejection counts. Reference
summaries are committed in the harness so a run can be checked against a known
good result with its ``verify.py``.

Why it is not part of ``tests/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Data size.** The input is a 13-subject, 64-channel BIDS P300 dataset of about
  790 MB, tracked with Git LFS. That is far too large to vendor into this
  repository.
- **Runtime.** A full run over all subjects and steps takes roughly 45 to 60
  minutes.
- **MATLAB requirement.** The oracle side needs a licensed MATLAB plus the MATLAB
  Engine for Python, with the interpreter architecture matching MATLAB's, and an
  EEGLAB checkout pointed to by ``EEGPREP_EEGLAB_ROOT``.

Bit-level agreement also depends on the NumPy and MATLAB stacks sharing a BLAS
accumulation order, so results are reproducible per platform pair rather than
universally. The harness documents the validated combinations.

Folding a reduced version into ``tests/`` — one subject, a few steps, behind the
existing ``matlab`` and ``slow`` markers — would be a worthwhile future
improvement. It is not wired up today.

Running it
~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/sccn/eegprep_parity_test
    cd eegprep_parity_test
    uv venv --python 3.12 .venv && source .venv/bin/activate
    pip install -r requirements.txt
    pip install "$MATLABROOT/extern/engines/python"
    export EEGPREP_EEGLAB_ROOT=/path/to/eeglab

    python run_parity.py                 # all subjects, all steps
    python run_parity.py sub-005 --steps 1 2 3
    python verify.py summary_new.md expected_results/summary_arm_R2025a.md

Point the harness at the EEGPrep you want to test by installing it into that
environment, for example ``pip install -e /path/to/eegprep`` for a working tree.
See the harness's own ``README.md`` and ``docs/SETUP.md`` for the validated
platform combinations and the required EEGLAB patches.

EEG And Session Contracts
=========================

The EEG dictionary, session-selection, and history contracts that feature code
must honor are documented for users and extension authors in
:ref:`contracts`.

EEGLAB Core Parity Matrix
=========================

The Phase 1 core parity epic uses a committed machine-readable matrix at
``docs/parity/eeglab_core_parity_matrix.json``. The matrix classifies the
EEGLAB public and semi-public functions in the scope categories recorded in
its metadata and source issue. It is a work contract for later phase agents,
not package runtime data.

Rows use these statuses:

- ``implemented``: EEGPrep already covers the behavior.
- ``partial``: EEGPrep has an implementation, but important EEGLAB behavior,
  options, or workflow paths remain.
- ``port``: the row should be ported or wrapped during the responsible phase.
- ``consolidated``: EEGPrep covers the behavior through another module or
  helper, and a duplicate same-name file is not needed.
- ``stale_skip``: the function is obsolete or stale enough to skip.
- ``matlab_runtime_skip``: the function is MATLAB-specific runtime, GUI shim,
  path, deployment, or compatibility behavior that should not exist in
  standalone EEGPrep package code.
- ``external_dependency_skip``: the function depends on external MATLAB
  toolboxes/plugins or web/path integration outside this epic.

Use ``stale_skip`` only when every stale-policy field in the matrix is false:
the function is not menu-reachable, not a documented user API, not called by an
in-scope workflow, not required by parity tests, not needed as a helper for the
remaining phases, and not a likely compatibility alias users type. When in
doubt, leave the row as ``port`` or ``partial`` and add notes for the
responsible phase.

When a later phase implements or intentionally skips a row:

1. Update ``status``, ``eegprep_equivalent``, ``rationale``,
   ``responsible_phase``, ``user_facing_surface``, and ``test_notes`` in the
   JSON row.
2. For a new ``stale_skip`` row, include the complete ``stale_policy`` object
   with all fields set to ``false`` and explain the evidence in ``rationale``.
3. For ``partial`` rows, keep the row ``partial`` until unsupported behavior is
   implemented or explicitly reclassified with a defensible limitation.
   Any row left as ``port`` or ``partial`` after a phase closes must cite a
   concrete follow-up issue in ``follow_up_issue`` or, when needed for prose
   context, in ``rationale`` or ``test_notes``.
4. Run the validator:

   .. code-block:: bash

      uv run --no-sync python -m tools.eeglab_parity_matrix

5. Run the focused matrix tests:

   .. code-block:: bash

      uv run --no-sync pytest tests/test_eeglab_parity_matrix.py

The validator may read ``src/eegprep/eeglab`` because it is development
tooling. Installed package code under ``src/eegprep`` must not read, import
from, or shell out to ``src/eegprep/eeglab``. If EEGLAB-like help text,
examples, options, or resources are needed at runtime, convert them into
EEGPrep-owned packaged resources instead of reaching into the vendored
reference tree.

EEGLAB Final Standalone Parity Matrix
=====================================

The final standalone parity epic uses a second machine-readable matrix at
``docs/parity/eeglab_final_parity_matrix.json``. It extends the core matrix into
the remaining product surfaces that are not simple core-function parity rows:
bundled plugin depth, MATLAB object/storage semantics, optional-toolbox
workflows, and documentation/tutorial coverage.

Rows group source files into user workflows, but every discovered final-epic
EEGLAB reference path must be covered exactly once. The validator discovers:

- ``plugins/clean_rawdata``, ``plugins/firfilt``, ``plugins/ICLabel``, and
  ``plugins/dipfit`` files, excluding vendored third-party MatConvNet and
  Manopt internals, examples, and tests;
- ``functions/@eegobj``, ``functions/@memmapdata``, and ``functions/@mmo``;
- EEGLAB tutorial scripts under ``tutorial_scripts``;
- selected optional-toolbox workflow rows that point back to the core matrix.

Final matrix statuses are ``implemented``, ``partial``, ``port``,
``consolidated``, ``stale_skip``, ``matlab_runtime_skip``,
``optional_dependency``, ``external_plugin``, and ``docs_gap``. Non-skip rows
must name a responsible Phase 2-8 issue. Skip rows must use
``responsible_phase: "none"``. ``optional_dependency`` rows must name the
backend decision, fallback behavior, user-facing message, and phase contract so
later agents do not silently fake external-toolbox behavior.

Validate the final matrix with:

.. code-block:: bash

   uv run --no-sync python -m tools.eeglab_final_parity_matrix --json

The docs architecture for the final epic is recorded directly in the matrix
metadata and its source issue. It should be useful to EEG researchers first:
describe EEGPrep's standalone Python package, Qt GUI, and ``eegprep-console``
behavior accurately, and use EEGLAB comparisons only where they help users
migrate or understand familiar concepts.

Building Documentation
======================

Build HTML Documentation
------------------------

Sync the docs extra, then build with the same command used by the Phase 7
acceptance criteria:

.. code-block:: bash

    uv sync --group dev --extra docs
    uv run --no-sync sphinx-build -b html docs/source docs/_build/html

The ``docs/Makefile`` target remains available for local iteration:

.. code-block:: bash

    uv run make -C docs html

The direct Sphinx command writes to ``docs/_build/html/``. The Makefile target
writes to ``docs/build/html/``.

View Documentation Locally
---------------------------

Open the built documentation in your browser:

.. code-block:: bash

    open docs/build/html/index.html  # macOS
    xdg-open docs/build/html/index.html  # Linux
    start docs/build/html/index.html  # Windows

Or use a local server:

.. code-block:: bash

    cd docs/build/html
    uv run python -m http.server 8000

Then visit ``http://localhost:8000`` in your browser.

Clean Build
-----------

Remove old build files and rebuild:

.. code-block:: bash

    uv run make -C docs clean
    uv run make -C docs html

Build Options
-------------

Build PDF documentation (requires LaTeX):

.. code-block:: bash

    uv run make -C docs latexpdf

Build EPUB documentation:

.. code-block:: bash

    uv run make -C docs epub

Debugging Tips
==============

Logging
-------

Enable debug logging in your code:

.. code-block:: python

    import logging

    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Use logging in your code
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

Breakpoints
-----------

Use Python's built-in debugger:

.. code-block:: python

    import pdb

    def my_function():
        x = 10
        pdb.set_trace()  # Execution pauses here
        y = x + 5
        return y

Or use the newer breakpoint() function (Python 3.7+):

.. code-block:: python

    def my_function():
        x = 10
        breakpoint()  # Execution pauses here
        y = x + 5
        return y

Profiling
---------

Profile code performance:

.. code-block:: python

    import cProfile
    import pstats

    # Profile a function
    profiler = cProfile.Profile()
    profiler.enable()

    # Your code here
    my_function()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Print top 10 functions

Memory Profiling
----------------

Install memory profiler:

.. code-block:: bash

    uv add --dev memory-profiler

Use it in your code:

.. code-block:: python

    from memory_profiler import profile

    @profile
    def my_function():
        large_list = [i for i in range(1000000)]
        return sum(large_list)

Run with:

.. code-block:: bash

    uv run python -m memory_profiler script.py

Release Process
===============

Releases are published by ``.github/workflows/release.yml`` when a ``v*`` tag is
pushed. See :ref:`releasing` for the full procedure, including the dry run and the
separate Docker image step.

EEGPrep uses `Semantic Versioning <https://semver.org/>`_: **MAJOR** for
incompatible API changes, **MINOR** for backward-compatible functionality, and
**PATCH** for backward-compatible bug fixes.

The version lives only in ``src/eegprep/__init__.py``; ``pyproject.toml`` reads it
via ``dynamic = ["version"]``.

Common Issues
=============

Import Errors
-------------

**Problem**: ``ModuleNotFoundError: No module named 'eegprep'``

**Solution**: Install the package in editable mode:

.. code-block:: bash

    uv sync --group dev

Test Failures
-------------

**Problem**: Tests fail with import errors

**Solution**: Ensure you're in the virtual environment and dependencies are installed:

.. code-block:: bash

    uv sync --group dev
    uv run pytest tests

Documentation Build Errors
---------------------------

**Problem**: Sphinx build fails with missing modules

**Solution**: Install documentation dependencies:

.. code-block:: bash

    uv sync --extra docs --group dev

Git Conflicts
-------------

**Problem**: Merge conflicts when pulling upstream changes

**Solution**: Resolve conflicts manually:

.. code-block:: bash

    git fetch upstream
    git rebase upstream/develop
    # Resolve conflicts in your editor
    git add .
    git rebase --continue

Virtual Environment Issues
---------------------------

**Problem**: Virtual environment not activating

**Solution**: Recreate the virtual environment:

.. code-block:: bash

    rm -rf .venv
    uv sync --group dev

Dependency Conflicts
--------------------

**Problem**: Dependency version conflicts

**Solution**: Refresh the locked environment:

.. code-block:: bash

    uv lock
    uv sync --group dev

Getting Help
============

- Check the :doc:`contributing` guide
- Review existing `GitHub Issues <https://github.com/sccn/eegprep/issues>`_
- Ask in GitHub Discussions
- Contact the maintainers

Happy developing!
