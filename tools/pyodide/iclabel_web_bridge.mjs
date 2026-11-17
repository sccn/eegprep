/**
 * Build the ONNX Runtime Web bridge consumed by the Pyodide ICLabel adapter.
 *
 * The Python side passes flat Float32Array values and explicit tensor shapes.
 * Returning the output typed array from the Promise keeps the Pyodide boundary
 * asynchronous without exposing ONNX Runtime objects to Python.
 *
 * The returned object also exposes modelBytes, the Uint8Array it was given.
 *
 * options.numThreads (default 1) sets ort.env.wasm.numThreads; every other option is passed
 * to InferenceSession.create.
 */
export function createIcLabelWebBridge(ort, modelBytes, options = {}) {
  const { numThreads = 1, ...sessionOptions } = options;
  const createOptions = { executionProviders: ["wasm"], ...sessionOptions };
  let sessionPromise;

  const getSession = () => {
    if (sessionPromise === undefined) {
      // ONNX Runtime Web's thread count is the process-wide ort.env.wasm.numThreads, read once
      // when its WebAssembly backend starts; a session option cannot set it. A host that already
      // created another session therefore decides the count.
      ort.env.wasm.numThreads = numThreads;
      // A copy, because onnxruntime-web may transfer the buffer it is given to a worker, which
      // detaches it; modelBytes must stay intact for the digest check and for a retried create.
      sessionPromise = ort.InferenceSession.create(modelBytes.slice(), createOptions).catch((error) => {
        sessionPromise = undefined;
        throw error;
      });
    }
    return sessionPromise;
  };

  return {
    // The bytes this bridge builds its session from. The Python adapter compares their digest with
    // the packaged iclabel_int8.onnx before every run, so a stale or wrong model is refused.
    modelBytes,
    run: (imageData, imageShape, psdmedData, psdmedShape, autocorrData, autocorrShape) =>
      getSession().then((session) =>
        session.run({
          image: new ort.Tensor("float32", imageData, imageShape),
          psdmed: new ort.Tensor("float32", psdmedData, psdmedShape),
          autocorr: new ort.Tensor("float32", autocorrData, autocorrShape),
        }).then((outputs) => outputs.output.data)),
  };
}
