#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
WORKTREE_TMP=$(mktemp -d "${TMPDIR:-/tmp}/eegprep-pyodide.XXXXXX")
trap 'rm -rf "$WORKTREE_TMP"' EXIT

usage() {
  echo "Usage: run_pyodide.sh --wheel PATH --script PATH [--docopt-wheel PATH] [--sample-data-dir PATH] [--output PATH] [--iclabel-web] -- [script args...]" >&2
}

WHEEL_PATH=""
DOCOPT_WHEEL_PATH=""
SCRIPT_PATH=""
SAMPLE_DATA_DIR=""
OUTPUT_PATH=""
ICLABEL_WEB=false

while (($# > 0)); do
  case "$1" in
    --wheel)
      WHEEL_PATH=${2:-}
      shift 2
      ;;
    --docopt-wheel)
      DOCOPT_WHEEL_PATH=${2:-}
      shift 2
      ;;
    --script)
      SCRIPT_PATH=${2:-}
      shift 2
      ;;
    --sample-data-dir)
      SAMPLE_DATA_DIR=${2:-}
      shift 2
      ;;
    --output)
      OUTPUT_PATH=${2:-}
      shift 2
      ;;
    --iclabel-web)
      ICLABEL_WEB=true
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$WHEEL_PATH" || -z "$SCRIPT_PATH" ]]; then
  usage
  exit 2
fi
if [[ ! -f "$WHEEL_PATH" || ! -f "$SCRIPT_PATH" ]]; then
  echo "Wheel and script paths must name existing files" >&2
  exit 2
fi

# Bun installs the pyodide and onnxruntime-web versions pinned in bun.lock; the harness
# itself runs on Node.js. Check both tools before any download so a missing one fails fast.
if ! command -v bun >/dev/null 2>&1; then
  echo "bun is required to install the pinned Pyodide packages (https://bun.sh)" >&2
  exit 2
fi
if ! command -v node >/dev/null 2>&1; then
  echo "node (Node.js 22 or newer) is required to run the Pyodide harness" >&2
  exit 2
fi

if [[ -z "$DOCOPT_WHEEL_PATH" ]]; then
  DOCOPT_DIR="$WORKTREE_TMP/docopt"
  uv run --no-sync python "$REPO_ROOT/tools/pyodide/prepare_docopt_wheel.py" --output-dir "$DOCOPT_DIR"
  DOCOPT_WHEEL_PATH=$(find "$DOCOPT_DIR" -maxdepth 1 -type f -name 'docopt-0.6.2-*.whl' -print -quit)
fi
if [[ ! -f "$DOCOPT_WHEEL_PATH" ]]; then
  echo "docopt wheel path must name an existing file" >&2
  exit 2
fi

JS_PREFIX="$WORKTREE_TMP/js"
mkdir -p "$JS_PREFIX"
cp "$REPO_ROOT/tools/pyodide/package.json" "$JS_PREFIX/package.json"
cp "$REPO_ROOT/tools/pyodide/bun.lock" "$JS_PREFIX/bun.lock"
bun install --frozen-lockfile --ignore-scripts --cwd "$JS_PREFIX"
PYODIDE_MODULE="$JS_PREFIX/node_modules/pyodide/pyodide.mjs"

NODE_ARGS=(
  "$SCRIPT_DIR/run_pyodide.mjs"
  --pyodide-module "$PYODIDE_MODULE"
  --wheel "$WHEEL_PATH"
  --docopt-wheel "$DOCOPT_WHEEL_PATH"
  --script "$SCRIPT_PATH"
)
if [[ "$ICLABEL_WEB" == true ]]; then
  ICLABEL_MODEL_PATH="$WORKTREE_TMP/iclabel.onnx"
  uv run --no-sync python - "$WHEEL_PATH" "$ICLABEL_MODEL_PATH" <<'PY'
import sys
import zipfile

wheel_path, output_path = sys.argv[1:]
model_name = "eegprep/plugins/ICLabel/iclabel.onnx"
with zipfile.ZipFile(wheel_path) as wheel:
    try:
        model = wheel.read(model_name)
    except KeyError as exc:
        raise SystemExit(f"Wheel is missing {model_name}") from exc
with open(output_path, "wb") as handle:
    handle.write(model)
PY
  NODE_ARGS+=(
    --iclabel-model "$ICLABEL_MODEL_PATH"
    --onnxruntime-web-module "$JS_PREFIX/node_modules/onnxruntime-web/dist/ort.bundle.min.mjs"
  )
fi
if [[ -n "$SAMPLE_DATA_DIR" ]]; then
  NODE_ARGS+=(--sample-data-dir "$SAMPLE_DATA_DIR")
fi
if [[ -n "$OUTPUT_PATH" ]]; then
  NODE_ARGS+=(--output "$OUTPUT_PATH")
fi

node "${NODE_ARGS[@]}" -- "$@"
