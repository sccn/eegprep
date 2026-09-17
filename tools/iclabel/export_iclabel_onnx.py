"""Export the ICLabel default network to ONNX.

Offline, developer-only tool: it produces ``iclabel.onnx``, the artifact
shipped in place of ``netICL.mat`` starting with issue #377. This script is
not installed with the ``eegprep`` package.

Regenerate the packaged artifact after changing ``netICL.mat`` or
``iclabel_net.py``:

    uv sync --group dev --extra torch
    uv run --no-sync python tools/iclabel/export_iclabel_onnx.py

Provenance: the exported graph is ``ICLabelNet.forward`` (three convolutional
branches over the topography image, PSD, and autocorrelation features,
concatenated and passed through a final conv + softmax), pinned at
``ONNX_OPSET_VERSION`` below. Feature extraction, input normalization, the
4-way augmentation, and the post-network averaging (``iclabel.py``) are
unchanged by this export; they run in plain numpy before and after the
network call.

torch is required only to run this script, never at ICLabel classification
time; the runtime backend is onnxruntime (``eegprep[iclabel]``).
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from eegprep.plugins.ICLabel.iclabel_net import ICLabelNet

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ICLABEL_DIR = REPO_ROOT / 'src' / 'eegprep' / 'plugins' / 'ICLabel'
DEFAULT_MAT_PATH = ICLABEL_DIR / 'netICL.mat'
DEFAULT_ONNX_PATH = ICLABEL_DIR / 'iclabel.onnx'

# Pinned per issue #377. The network only uses Conv2d, LeakyReLU, Softmax,
# Concat, and Reshape, all supported since opset 7-9, so the choice is driven
# by runtime compatibility rather than op coverage: opset 17 has been
# supported by onnxruntime since 1.14 (2023), giving broad compatibility
# today while staying modern enough for the onnxruntime-web/wasm target
# planned for phase 5.
ONNX_OPSET_VERSION = 17

INPUT_NAMES = ('image', 'psdmed', 'autocorr')
OUTPUT_NAMES = ('output',)
_INPUT_SHAPES = ((1, 1, 32, 32), (1, 1, 1, 100), (1, 1, 1, 100))


def export(mat_path: Path = DEFAULT_MAT_PATH, onnx_path: Path = DEFAULT_ONNX_PATH) -> Path:
    """Export the ICLabel default network to ``onnx_path`` with a pinned opset."""
    model = ICLabelNet(str(mat_path))
    model.eval()

    dummy_inputs = tuple(torch.zeros(shape, dtype=torch.float32) for shape in _INPUT_SHAPES)
    dynamic_axes = {name: {0: 'batch'} for name in (*INPUT_NAMES, *OUTPUT_NAMES)}

    torch.onnx.export(
        model,
        dummy_inputs,
        str(onnx_path),
        input_names=list(INPUT_NAMES),
        output_names=list(OUTPUT_NAMES),
        opset_version=ONNX_OPSET_VERSION,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(str(onnx_path))
    return onnx_path


def verify(mat_path: Path, onnx_path: Path, batch_size: int = 16, seed: int = 0) -> float:
    """Compare torch and onnxruntime outputs on random inputs; return max abs diff."""
    model = ICLabelNet(str(mat_path))
    model.eval()

    rng = np.random.default_rng(seed)
    inputs = {
        name: rng.standard_normal((batch_size, *shape[1:])).astype(np.float32)
        for name, shape in zip(INPUT_NAMES, _INPUT_SHAPES)
    }

    with torch.no_grad():
        torch_out = model(*(torch.from_numpy(inputs[name]) for name in INPUT_NAMES)).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    (onnx_out,) = session.run(list(OUTPUT_NAMES), inputs)

    max_abs_diff = float(np.max(np.abs(torch_out - onnx_out)))
    return max_abs_diff


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mat-path', type=Path, default=DEFAULT_MAT_PATH)
    parser.add_argument('--onnx-path', type=Path, default=DEFAULT_ONNX_PATH)
    args = parser.parse_args()

    onnx_path = export(args.mat_path, args.onnx_path)
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    logger.info("Exported %s (opset %d, %.2f MB)", onnx_path, ONNX_OPSET_VERSION, size_mb)

    max_abs_diff = verify(args.mat_path, onnx_path)
    logger.info("torch vs onnxruntime max abs diff on random inputs: %.3e", max_abs_diff)
    if max_abs_diff > 1e-4:
        raise RuntimeError(f"ONNX export diverges from torch beyond tolerance: max abs diff {max_abs_diff:.3e}")


if __name__ == '__main__':
    main()
