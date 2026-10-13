/**
 * Build the ONNX Runtime Web bridge consumed by the Pyodide ICLabel adapter.
 *
 * The Python side passes flat Float32Array values and explicit tensor shapes.
 * Returning the output typed array from the Promise keeps the Pyodide boundary
 * asynchronous without exposing ONNX Runtime objects to Python.
 */
export function createIcLabelWebBridge(ort, modelBytes, options = {}) {
  const sessionOptions = {
    executionProviders: ["wasm"],
    numThreads: 1,
    ...options,
  };
  let sessionPromise;

  const getSession = () => {
    if (sessionPromise === undefined) {
      sessionPromise = ort.InferenceSession.create(modelBytes, sessionOptions).catch((error) => {
        sessionPromise = undefined;
        throw error;
      });
    }
    return sessionPromise;
  };

  return {
    run: (imageData, imageShape, psdmedData, psdmedShape, autocorrData, autocorrShape) =>
      getSession().then((session) =>
        session.run({
          image: new ort.Tensor("float32", imageData, imageShape),
          psdmed: new ort.Tensor("float32", psdmedData, psdmedShape),
          autocorr: new ort.Tensor("float32", autocorrData, autocorrShape),
        }).then((outputs) => outputs.output.data)),
  };
}
