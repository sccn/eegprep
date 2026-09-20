/**
 * Report whether ONNX Runtime Web can execute an ICLabel artifact.
 *
 * The browser path runs on ONNX Runtime Web's WebAssembly execution provider,
 * which is a different kernel set from the native `onnxruntime` package. A 4-bit
 * artifact that loads natively is not evidence that the browser can run it, so
 * this check exercises the artifact against the same `onnxruntime-web` build the
 * Pyodide harness pins and prints what actually happened.
 *
 * Node's WebAssembly backend loads the very same `.wasm` binary a browser does;
 * the outputs were verified byte-identical against headless Chrome while this
 * check was written.
 *
 * Usage:
 *   node tools/iclabel/check_iclabel_ort_web.mjs \
 *     --ort <path to onnxruntime-web ort.bundle.min.mjs> \
 *     --model <artifact.onnx> [--model <other.onnx> ...]
 */

import fs from "node:fs";
import path from "node:path";

const BATCH = 4;

function parseArgs(argv) {
  const args = { models: [], ort: null };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--model") args.models.push(argv[++i]);
    else if (argv[i] === "--ort") args.ort = argv[++i];
    else throw new Error(`Unrecognized argument: ${argv[i]}`);
  }
  if (args.ort === null || args.models.length === 0) {
    throw new Error("Usage: --ort <ort.bundle.min.mjs> --model <artifact.onnx> [--model ...]");
  }
  return args;
}

const args = parseArgs(process.argv.slice(2));
const ort = await import(path.resolve(args.ort));
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = `${path.dirname(path.resolve(args.ort))}/`;
ort.env.logLevel = "error";

// Deterministic stand-in features: this check answers "does the runtime execute
// these operators", not "is the model accurate". Accuracy is the frozen
// evaluation set's job, in tools/iclabel/quantize_iclabel_int4.py.
const ramp = (n, step) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * step));

const results = [];
for (const modelPath of args.models) {
  const entry = { model: path.basename(modelPath) };
  try {
    const bytes = new Uint8Array(fs.readFileSync(modelPath));
    entry.size_bytes = bytes.length;
    const started = Date.now();
    const session = await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
      numThreads: 1,
    });
    const outputs = await session.run({
      image: new ort.Tensor("float32", ramp(BATCH * 32 * 32, 0.013), [BATCH, 1, 32, 32]),
      psdmed: new ort.Tensor("float32", ramp(BATCH * 100, 0.021), [BATCH, 1, 1, 100]),
      autocorr: new ort.Tensor("float32", ramp(BATCH * 100, 0.037), [BATCH, 1, 1, 100]),
    });
    entry.runs = true;
    entry.elapsed_ms = Date.now() - started;
    entry.output_dims = Array.from(outputs.output.dims);
    entry.output_is_finite = Array.from(outputs.output.data).every(Number.isFinite);
  } catch (error) {
    entry.runs = false;
    entry.error = String(error?.message ?? error);
  }
  results.push(entry);
}

console.log(JSON.stringify({ onnxruntime_web_version: ort.env.versions.web, results }, null, 2));
if (results.some((entry) => !entry.runs)) process.exitCode = 1;
