# Pyodide browser execution, Phase 2/3 benchmark, and Phase 5 ICLabel parity

This page records the Phase 2 gate and the Phase 3 runica backend decision for
the [browser epic](https://github.com/sccn/eegprep/issues/324).
The harness installs a wheel built from the working tree under Pyodide, runs a
real continuous-data smoke pipeline, and measures the ICA and matrix products
that determine whether later browser work is worthwhile.

## Runtime boundary

The harness pins [Pyodide 0.29.5](https://pyodide.org/en/stable/usage/faq.html)
and uses Pyodide's `emfs:` local-wheel transport. The package-resolution path
pre-installs a universal `docopt==0.6.2` wheel built from a hash-checked sdist
because the PyPI release is an sdist. The live harness also pins the
Pyodide-compatible pure Python versions `mne==1.10.0`, `sympy==1.14.0`, and
`threadpoolctl==3.6.0` while
leaving EEGPrep's published dependency ranges unchanged.
The smoke install completed successfully with this closure; no additional
sdist-only blocker surfaced in the live harness.

EEGPrep's ICA implementations do not use MNE. In particular, importing and
running `runica` does not eagerly import MNE. The current published dependency
closure still includes MNE for EEG file and interoperability paths; removing
that browser-install dependency is a separate packaging task.

Pyodide does not provide pthread support, so one Pyodide instance cannot run
four or eight Python threads for a single ICA call. The benchmark therefore
limits native BLAS to one thread as well. A browser [Web Worker](https://pyodide.org/en/stable/usage/index.html)
is still the correct execution boundary: it keeps the UI responsive, but it
does not make one ICA solve faster.

The scalable design is a bounded worker pool for independent jobs. EEGPrep
should own the worker-safe operation contract and serialized progress/result
messages. The NEMAR/OSA host should own worker lifecycle, queueing, resource
limits, cancellation policy, and whether a request uses one worker or a pool.
Splitting one existing `runica` or Picard solve across workers is out of scope:
its iterative weights are global state and would require a new synchronized
distributed algorithm and a separate parity gate.

## Reproducing the gate

The harness requires `uv`, Node.js 22 or newer, and npm. From the repository
root, build the EEGPrep wheel and run the sample-data smoke test with:

```bash
uv build --wheel --out-dir pyodide-artifacts
tools/pyodide/run_pyodide.sh \
  --wheel pyodide-artifacts/eegprep-*.whl \
  --script tools/pyodide/smoke.py \
  --sample-data-dir sample_data
```

The CI job runs the same smoke test, then benchmarks a fixed `(64, 15000)`
continuous array (64 channels, 60 seconds at 250 Hz). Each algorithm has one
warm-up and three measured runs, with a maximum of 512 ICA iterations and a
fixed seed. The matrix benchmark uses the two runica products:

* `(64, 64) @ (64, 49)` for block activations;
* `(64, 64) @ (64, 64)` for the square weight update.

Picard is measured through its underlying `picard(..., return_n_iter=True)` API
with the same algorithmic options as EEGPrep's `eeg_picard` wrapper; benchmark
output is quiet and includes the underlying iteration telemetry. The benchmark
does not change either production ICA implementation. `runica` casts its input
to float64, so Phase 3 selects one matrix-product helper at import time: native
platforms retain NumPy `@`, while Emscripten uses SciPy `dgemm`. Picard remains
unchanged.

## Measured results

The following tables are generated from the native and Pyodide JSON reports
with `tools/pyodide/compare_benchmarks.py`. The raw reports and loader logs are
also uploaded by CI as the `pyodide-phase-2-through-phase-6-output` artifact.
That bundle includes the Phase 3 backend-selection benchmark and the Phase 6
frozen ICLabel quantization report in addition to the Phase 2 harness and Phase
5 browser parity evidence.

## Phase 5 browser ICLabel

The same harness can run the packaged default ICLabel model through ONNX
Runtime Web:

```bash
tools/pyodide/run_pyodide.sh \
  --iclabel-web \
  --wheel pyodide-artifacts/eegprep-*.whl \
  --script tools/pyodide/iclabel_parity.py \
  --sample-data-dir sample_data \
  --output pyodide-artifacts/iclabel-pyodide.json \
  -- --platform pyodide
```

The Node host loads the pinned `onnxruntime-web` package, registers the
asynchronous bridge, and passes the packaged `iclabel.onnx` bytes into the
Pyodide runtime. The Python `iclabel_async`/`pop_iclabel_async` APIs never
import native `onnxruntime` on Emscripten. CI compares the browser result with
a native `onnxruntime` reference using `rtol=1e-4` and `atol=1e-5`, the
established ICLabel parity tolerance. The bridge defaults to ONNX Runtime
Web's WASM execution provider with one thread; concurrency across independent
jobs remains a host/Web Worker concern.

### ICA

| Algorithm | Native median (s) | Native median iterations | Native converged | Pyodide median (s) | Pyodide median iterations | Pyodide converged |
| --- | ---: | ---: | :---: | ---: | ---: | :---: |
| runica | 84.859703 | 512.0 | false | 204.447027 | 512.0 | false |
| picard | 6.825637 | 511.0 | false | 65.385375 | 511.0 | false |

### Matrix multiplication

| runica product | dtype | Native NumPy `@` (s) | Native BLAS (s) | Native speedup | Pyodide NumPy `@` (s) | Pyodide BLAS (s) | Pyodide speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| activation | float64 | 0.000012 | 0.000017 | 0.69x | 0.000244 | 0.000099 | 2.46x |
| activation | float32 | 0.000006 | 0.000014 | 0.40x | 0.000242 | 0.000264 | 0.91x |
| weight_update | float64 | 0.000008 | 0.000018 | 0.46x | 0.000332 | 0.000125 | 2.66x |
| weight_update | float32 | 0.000005 | 0.000015 | 0.34x | 0.000317 | 0.000154 | 2.06x |

### Decisions

The comparison script records whether Picard remains a safe browser default
and whether SciPy BLAS is at least 1.2x faster than NumPy `@` for every tested
runica product and dtype. Those decisions are intentionally based on measured
convergence and timings, not on the code path alone.

For this run, Picard was much faster in wall-clock time but emitted its
non-convergence warning at 511 iterations; runica also reached the 512-step
cap. The conservative browser-default gate therefore returns **false** for
Picard. The original all-dtypes BLAS gate also returns **false**: the float32
activation product was 0.91x with `sgemm`, below the 1.2x threshold.

Phase 3 adopts the narrower production-relevant decision. Because `runica`
already operates on float64 data, its Emscripten hot-loop products use
`scipy.linalg.blas.dgemm`; the measured float64 speedups were 2.46x for
activation and 2.66x for the weight update. Native execution retains NumPy
`@`, and Picard is not monkeypatched or distributed across workers. The
comparison script intentionally retains the universal all-dtypes gate, so its
`phase3_recommended` field remains **false** even though the float64-only
implementation is now covered and benchmarked.
