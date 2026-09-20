"""Build and evaluate 4-bit ICLabel ONNX artifacts against the frozen parity set.

This developer-only tool answers whether ICLabel is worth shipping at 4 bits.
It reuses the phase 6 frozen evaluation set, the float32 teacher, and the exact
parity comparison in ``quantize_iclabel_onnx`` so that int4 and int8 numbers are
produced by one code path and are directly comparable.

Three facts drive the way the artifact is built:

* Every weight in the network belongs to a ``Conv``. There is no ``MatMul`` or
  ``Gemm``, so ``MatMulNBits`` -- the operator most 4-bit ONNX tooling targets --
  has nothing to attach to here. The only portable 4-bit representation is an
  int4 initializer behind ``DequantizeLinear`` feeding the existing float
  ``Conv``, which is what this tool emits.
* int4 tensors and blocked ``DequantizeLinear`` are opset 21 constructs, so the
  pinned opset-17 export is lifted to 21 before the weights are rewritten. That
  lift is the artifact's real compatibility cost: it raises the onnxruntime
  floor from 1.18 to 1.19.
* A single scale per output channel is too coarse at 4 bits. Blocking the scale
  along the input-channel axis is what recovers parity, and the block size is
  the only knob that trades size against agreement.

Feature extraction, input normalization, augmentation, and softmax stay outside
quantization, exactly as in the int8 path.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tools.iclabel.quantize_iclabel_onnx import (
    CLASS_NAMES,
    DEFAULT_EVALUATION_FEATURES,
    DEFAULT_FLOAT32_ARTIFACT,
    DEFAULT_FROZEN_MANIFEST,
    MIN_KEEP_REJECT_AGREEMENT,
    MIN_TOP1_AGREEMENT,
    _sha256,
    compare_predictions,
    load_frozen_manifest,
    load_verified_feature_archive,
    parity_gate_passes,
    predict_features,
)


DEFAULT_ARTIFACT_DIR = Path(__file__).with_name("artifacts")
DEFAULT_INT4_ARTIFACT = DEFAULT_ARTIFACT_DIR / "iclabel_int4_block32.onnx"
DEFAULT_INT8_ARTIFACT = DEFAULT_ARTIFACT_DIR / "iclabel_int8_weight_only.onnx"
DEFAULT_REPORT = Path(__file__).with_name("int4_feasibility_report.json")

# Opset 21 is the first version that defines the int4/uint4 tensor element
# types and the block_size attribute on DequantizeLinear.
INT4_OPSET_VERSION = 21
# onnxruntime 1.18 raises "MLDataType for: tensor(uint4) is not currently
# registered or supported"; 1.19 is the first release that loads these graphs.
MINIMUM_ONNXRUNTIME_VERSION = "1.19"
# Convs below this size are left in float32. Together they hold 5,585 of the
# network's 2,903,817 parameters, so quantizing them saves about 22 KB while
# adding avoidable error on the narrow input-channel layers.
MIN_PARAMS_TO_QUANTIZE = 50_000
DEFAULT_BLOCK_SIZE = 32


def largest_divisor_at_most(size: int, cap: int) -> int:
    """Return the largest divisor of ``size`` that does not exceed ``cap``.

    Blocked DequantizeLinear needs the blocked axis to divide evenly, and the
    ICLabel graph has one Conv whose input-channel count is 712, which no power
    of two up to 128 divides.
    """
    for block in range(min(cap, size), 0, -1):
        if size % block == 0:
            return block
    return 1


def quantize_blocked_uint4(
    weights: np.ndarray,
    axis: int,
    block: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Asymmetric uint4 quantization in blocks along ``axis``.

    Asymmetric quantization uses all 16 codes rather than the 15 a symmetric
    int4 grid reaches, which matters more at 4 bits than at 8.
    """
    shape = weights.shape
    if shape[axis] % block:
        raise ValueError(f"Block size {block} does not divide axis {axis} of length {shape[axis]}")
    block_count = shape[axis] // block
    blocked_shape = shape[:axis] + (block_count, block) + shape[axis + 1 :]
    blocked = weights.reshape(blocked_shape)

    low = np.minimum(blocked.min(axis=axis + 1, keepdims=True), 0.0)
    high = np.maximum(blocked.max(axis=axis + 1, keepdims=True), 0.0)
    scale = (high - low) / 15.0
    # A constant block has no range; a unit scale with a zero point reproduces
    # it exactly and keeps the dequantized value finite.
    scale = np.where(scale <= 0, 1.0, scale)
    zero_point = np.clip(np.rint(-low / scale), 0, 15)
    codes = np.clip(np.rint(blocked / scale) + zero_point, 0, 15).astype(np.uint8)

    scale_shape = shape[:axis] + (block_count,) + shape[axis + 1 :]
    return (
        codes.reshape(shape),
        scale.astype(np.float32).reshape(scale_shape),
        zero_point.astype(np.uint8).reshape(scale_shape),
    )


def load_model_at_int4_opset(model_path: Path):
    """Load the pinned float32 export and lift it to the int4-capable opset."""
    import onnx

    model = onnx.load(str(model_path))
    current = max(entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx"))
    if current < INT4_OPSET_VERSION:
        model = onnx.version_converter.convert_version(model, INT4_OPSET_VERSION)
    # IR version 10 is the minimum that carries the 4-bit tensor element types.
    model.ir_version = 10
    return model


def quantize_int4(
    model_input: Path,
    model_output: Path,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> list[dict[str, Any]]:
    """Write a blocked uint4 Conv-weight artifact with float inputs and outputs."""
    import onnx
    from onnx import helper, numpy_helper

    import ml_dtypes

    model = load_model_at_int4_opset(model_input)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    replaced: set[str] = set()
    added = []
    # Keyed by position in graph_nodes, not by id(): iterating a protobuf
    # repeated field can hand back fresh Python wrappers each time, so object
    # identity is not stable across two passes and the DequantizeLinear can end
    # up placed after the Conv that consumes it.
    graph_nodes = list(model.graph.node)
    dequant_nodes: dict[int, Any] = {}
    layers: list[dict[str, Any]] = []

    for index, node in enumerate(graph_nodes):
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        weight_name = node.input[1]
        initializer = initializers.get(weight_name)
        if initializer is None:
            raise ValueError(f"ICLabel Conv weight {weight_name!r} is not an initializer")
        weights = numpy_helper.to_array(initializer).astype(np.float32, copy=False)
        if weights.size < MIN_PARAMS_TO_QUANTIZE:
            continue

        axis = 1
        block = largest_divisor_at_most(weights.shape[axis], block_size)
        codes, scale, zero_point = quantize_blocked_uint4(weights, axis, block)

        code_name = f"{weight_name}_uint4"
        scale_name = f"{weight_name}_scale"
        zero_point_name = f"{weight_name}_zero_point"
        dequant_name = f"{weight_name}_dequantized"
        added.extend(
            [
                numpy_helper.from_array(codes.astype(ml_dtypes.uint4), name=code_name),
                numpy_helper.from_array(scale, name=scale_name),
                numpy_helper.from_array(zero_point.astype(ml_dtypes.uint4), name=zero_point_name),
            ]
        )
        dequant_nodes[index] = helper.make_node(
            "DequantizeLinear",
            [code_name, scale_name, zero_point_name],
            [dequant_name],
            name=f"{weight_name}_dequantize",
            axis=axis,
            block_size=block,
        )
        node.input[1] = dequant_name
        replaced.add(weight_name)
        layers.append(
            {
                "weight": weight_name,
                "shape": [int(value) for value in weights.shape],
                "parameters": int(weights.size),
                "block_size": int(block),
                "scale_count": int(scale.size),
            }
        )

    if not layers:
        raise ValueError("ICLabel model contains no Conv weights large enough to quantize")

    kept = [initializer for initializer in model.graph.initializer if initializer.name not in replaced]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept + added)

    ordered = []
    for index, node in enumerate(graph_nodes):
        dequant = dequant_nodes.get(index)
        if dequant is not None:
            ordered.append(dequant)
        ordered.append(node)
    del model.graph.node[:]
    model.graph.node.extend(ordered)

    # full_check enforces topological ordering, which is exactly what a
    # misplaced DequantizeLinear breaks.
    onnx.checker.check_model(model, full_check=True)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(model_output))
    return layers


def _artifact_metrics(
    artifact: Path,
    teacher: np.ndarray,
    features: Sequence[np.ndarray],
    thresholds: np.ndarray,
) -> dict[str, Any]:
    metrics = compare_predictions(teacher, predict_features(artifact, features), thresholds)
    return {
        "artifact": artifact.name,
        "size_bytes": artifact.stat().st_size,
        "sha256": _sha256(artifact),
        "gate_pass": parity_gate_passes(metrics),
        **metrics,
    }


def evaluate_int4(
    float32_artifact: Path,
    int4_artifact: Path,
    int8_artifact: Path,
    evaluation_features: Path,
    layers: Sequence[Mapping[str, Any]],
    manifest: Path = DEFAULT_FROZEN_MANIFEST,
) -> dict[str, Any]:
    """Score int4 against the float32 teacher on the frozen evaluation set."""
    from eegprep.plugins.ICLabel.pop_icflag import DEFAULT_ICFLAG_THRESHOLDS

    frozen = load_frozen_manifest(manifest)
    features = load_verified_feature_archive(evaluation_features, frozen, "evaluation")
    thresholds = np.asarray(DEFAULT_ICFLAG_THRESHOLDS, dtype=float)
    teacher = predict_features(float32_artifact, features)

    quantized_parameters = sum(int(layer["parameters"]) for layer in layers)
    scale_count = sum(int(layer["scale_count"]) for layer in layers)

    report: dict[str, Any] = {
        "manifest": Path(manifest).name,
        "class_names": list(CLASS_NAMES),
        "gate": {
            "minimum_top1_agreement": MIN_TOP1_AGREEMENT,
            "minimum_keep_reject_agreement": MIN_KEEP_REJECT_AGREEMENT,
        },
        "quantization": {
            "scheme": "asymmetric uint4 Conv weights, blocked along the input-channel axis",
            "block_size": DEFAULT_BLOCK_SIZE,
            "opset": INT4_OPSET_VERSION,
            "minimum_onnxruntime_version": MINIMUM_ONNXRUNTIME_VERSION,
            "layers": list(layers),
            "quantized_parameters": quantized_parameters,
            "scale_count": scale_count,
            "input_normalization_and_augmentation": "unchanged from iclabel.py",
            "output_softmax": "unchanged from the float32 graph",
        },
        "float32_reference": {
            "artifact": float32_artifact.name,
            "size_bytes": float32_artifact.stat().st_size,
            "sha256": _sha256(float32_artifact),
            **compare_predictions(teacher, teacher, thresholds),
        },
        "candidates": {
            "int8_weight_only": _artifact_metrics(int8_artifact, teacher, features, thresholds),
            "int4_blocked": _artifact_metrics(int4_artifact, teacher, features, thresholds),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--float32-artifact", type=Path, default=DEFAULT_FLOAT32_ARTIFACT)
    parser.add_argument("--int8-artifact", type=Path, default=DEFAULT_INT8_ARTIFACT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_FROZEN_MANIFEST)
    parser.add_argument("--evaluation-features", type=Path, default=DEFAULT_EVALUATION_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_INT4_ARTIFACT)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    layers = quantize_int4(args.float32_artifact, args.output, args.block_size)
    report = evaluate_int4(
        args.float32_artifact,
        args.output,
        args.int8_artifact,
        args.evaluation_features,
        layers,
        args.manifest,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
