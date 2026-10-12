# Pyodide browser execution, benchmarks, and ICLabel parity

This page documents the Pyodide harness for the [browser epic](https://github.com/sccn/eegprep/issues/324).
The harness installs a wheel built from the working tree under Pyodide, runs a real continuous-data smoke pipeline,
measures the independent component analysis (ICA) and matrix products that determine whether browser ICA is worthwhile,
and checks that ICLabel gives the same classifications in the browser as natively.

## Runtime boundary

The harness pins [Pyodide 0.29.5](https://pyodide.org/en/stable/usage/faq.html)
and uses Pyodide's `emfs:` local-wheel transport.
The package-resolution path pre-installs a universal `docopt==0.6.2` wheel built from a hash-checked source distribution (sdist)
because the PyPI release is an sdist.
The live harness also pins `mne==1.10.0`, the lowest version EEGPrep declares,
and `threadpoolctl==3.6.0`, the version in `uv.lock` (Pyodide bundles 3.5.0),
while leaving EEGPrep's published dependency ranges unchanged.
As of 2026-10-12, with Pyodide 0.29.5, the smoke install completes with this closure
and no further sdist-only blocker has surfaced in the live harness.

EEGPrep's ICA implementations do not use MNE.
In particular, importing and running `runica` does not eagerly import MNE.
The current published dependency closure still includes MNE for electroencephalography (EEG) file and interoperability paths;
removing that browser-install dependency is a separate packaging task.

Pyodide does not provide pthread support,
so one Pyodide instance cannot run four or eight Python threads for a single ICA call.
The benchmark therefore limits native Basic Linear Algebra Subprograms (BLAS) to one thread as well.
A browser [Web Worker](https://pyodide.org/en/stable/usage/index.html)
is still the correct execution boundary:
it keeps the user interface responsive, but it does not make one ICA solve faster.

The scalable design is a bounded worker pool for independent jobs.
EEGPrep should own the worker-safe operation contract and serialized progress/result messages.
The host application, such as NEMAR (the NeuroElectroMagnetic Data Archive and Tools Resource),
should own worker lifecycle, queueing, resource limits, cancellation policy,
and whether a request uses one worker or a pool.
Splitting one existing `runica` or Picard solve across workers is out of scope:
its iterative weights are global state and would require a new synchronized distributed algorithm and a separate parity gate.

## Reproducing the gate

The harness requires `uv`, Node.js 22 or newer, and [Bun](https://bun.sh).
Bun installs the pinned `pyodide` and `onnxruntime-web` packages from
`tools/pyodide/bun.lock` (`bun install --frozen-lockfile`); Node.js runs the harness.
From the repository root, build the EEGPrep wheel and run the sample-data smoke test with:

```bash
uv build --wheel --out-dir pyodide-artifacts
tools/pyodide/run_pyodide.sh \
  --wheel pyodide-artifacts/eegprep-*.whl \
  --script tools/pyodide/smoke.py \
  --sample-data-dir sample_data
```

The continuous integration (CI) job runs the same smoke test,
then benchmarks a fixed `(64, 15000)` continuous array (64 channels, 60 seconds at 250 Hz).
Each algorithm has one warm-up and three measured runs,
with a maximum of 512 ICA iterations and a fixed seed.
The matrix benchmark uses the two runica products:

* `(64, 64) @ (64, 49)` for block activations;
* `(64, 64) @ (64, 64)` for the square weight update.

Picard is measured through its underlying `picard(..., return_n_iter=True)` API
with the same algorithmic options as EEGPrep's `eeg_picard` wrapper;
benchmark output is quiet and includes the underlying iteration telemetry.
The benchmark does not change either production ICA implementation.
`runica` casts its input to float64,
so it selects one matrix-product helper at import time:
native platforms retain NumPy `@`, while Emscripten uses SciPy `dgemm`.
Picard remains unchanged.

## Measured results

These numbers come from one run on 2026-10-12, on a shared Apple M4 Pro laptop (14 cores, 64 GB memory, macOS 27.0.1),
using the harness in `tools/pyodide` as landed with the browser epic (an EEGPrep 0.4.0 wheel built from that tree).
The Pyodide column ran under Node.js v26.10.0 with Pyodide 0.29.5 (21 minutes 49 seconds in total);
the native column ran under Python 3.12.15 with NumPy 2.2.6 and SciPy 1.16.3, restricted to one BLAS thread
(`OMP_NUM_THREADS=1` and the related variables CI sets).
Treat absolute times as indicative, because they vary with hardware and load;
CI measures again on its own runners and uploads the reports for every run.

The tables are generated from the native and Pyodide JSON reports with `tools/pyodide/compare_benchmarks.py`.
CI uploads the wheel, the raw reports, the loader logs, and the ICLabel quantization log (`iclabel-quantization.log`)
as the `pyodide-harness-output` artifact.

### ICA

| Algorithm | Native median (s) | Native median iterations | Native converged | Pyodide median (s) | Pyodide median iterations | Pyodide converged |
| --- | ---: | ---: | :---: | ---: | ---: | :---: |
| runica | 89.102553 | 512.0 | false | 241.325692 | 512.0 | false |
| picard | 7.413262 | 511.0 | false | 81.136565 | 511.0 | false |

### Matrix multiplication

| runica product | dtype | Native NumPy `@` (s) | Native BLAS (s) | Native speedup | Pyodide NumPy `@` (s) | Pyodide BLAS (s) | Pyodide speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| activation | float64 | 0.000009 | 0.000018 | 0.52x | 0.000263 | 0.000131 | 2.01x |
| activation | float32 | 0.000005 | 0.000014 | 0.37x | 0.000270 | 0.000278 | 0.97x |
| weight_update | float64 | 0.000008 | 0.000018 | 0.46x | 0.000410 | 0.000161 | 2.55x |
| weight_update | float32 | 0.000005 | 0.000015 | 0.33x | 0.000404 | 0.000200 | 2.01x |

### Decisions

The comparison script records whether Picard remains a safe browser default
and whether SciPy BLAS is at least 1.2x faster than NumPy `@` for every tested runica product and dtype.
Those decisions are intentionally based on measured convergence and timings, not on the code path alone.

For this run, Picard was much faster in wall-clock time than `runica` in Pyodide (81 s against 241 s)
but emitted its non-convergence warning at 511 iterations;
`runica` also reached the 512-step cap.
The conservative browser-default gate therefore returns **false** for Picard.
The all-dtypes BLAS gate also returns **false**:
the float32 activation product was 0.97x with `sgemm`, below the 1.2x threshold.

EEGPrep adopts the narrower production-relevant decision.
Because `runica` already operates on float64 data,
its Emscripten hot-loop products use `scipy.linalg.blas.dgemm`;
the measured float64 speedups were 2.01x for activation and 2.55x for the weight update.
Native execution retains NumPy `@`, and Picard is not monkeypatched or distributed across workers.
The comparison script intentionally retains the universal all-dtypes gate,
so its `blas_recommended` field remains **false** even though the float64-only implementation is now covered and benchmarked.

## Browser ICLabel parity

The same harness can run the packaged default ICLabel model through Open Neural Network Exchange (ONNX) Runtime Web,
the browser build of the ONNX inference engine:

```bash
tools/pyodide/run_pyodide.sh \
  --iclabel-web \
  --wheel pyodide-artifacts/eegprep-*.whl \
  --script tools/pyodide/iclabel_parity.py \
  --sample-data-dir sample_data \
  --output pyodide-artifacts/iclabel-pyodide.json \
  -- --platform pyodide
```

The Node host loads the pinned `onnxruntime-web` package,
creates one inference session from the `iclabel.onnx` bytes extracted from the built wheel,
and exposes it as `globalThis.eegprep_iclabel_web`.
Pyodide sends only input tensors across that bridge.
The Python `iclabel_async`/`pop_iclabel_async` APIs never import native `onnxruntime` on Emscripten.
If a host does not register the bridge, they raise an `ImportError` that names this contract.

CI compares the browser result with a native `onnxruntime` reference using `rtol=1e-4` and `atol=1e-5`,
the established ICLabel parity tolerance.
Both reports also record the SHA-256 digest of the `iclabel.onnx` in their installed EEGPrep package
(the working tree natively, the built wheel in Pyodide),
and the comparison fails if the digests differ, so parity is always measured on the model the tree ships.
The bridge defaults to the ONNX Runtime Web WebAssembly (WASM) execution provider with one thread;
concurrency across independent jobs remains a host/Web Worker concern.
