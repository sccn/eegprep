"""Guard the ICLabel 4-bit feasibility findings.

These tests keep the int4 study honest rather than keeping int4 shippable: the
artifact is a measured alternative, not the packaged default. They assert the
things a future change could silently break -- that the committed artifact still
clears the same frozen gate the int8 default clears, that the graph really is
4-bit rather than a widened copy, that the dequantization math is the blocked
scheme it claims to be, and that the packaged default is still int8.
"""

from pathlib import Path

import numpy as np
import pytest

from tools.iclabel.quantize_iclabel_int4 import (
    DEFAULT_INT4_ARTIFACT,
    INT4_OPSET_VERSION,
    MIN_PARAMS_TO_QUANTIZE,
    MINIMUM_ONNXRUNTIME_VERSION,
    largest_divisor_at_most,
    quantize_blocked_uint4,
    quantize_int4,
)
from tools.iclabel.quantize_iclabel_onnx import (
    DEFAULT_EVALUATION_FEATURES,
    DEFAULT_FLOAT32_ARTIFACT,
    DEFAULT_FROZEN_MANIFEST,
    DEFAULT_WEIGHT_ONLY_ARTIFACT,
    MIN_KEEP_REJECT_AGREEMENT,
    MIN_TOP1_AGREEMENT,
    compare_predictions,
    load_frozen_manifest,
    load_verified_feature_archive,
    parity_gate_passes,
    predict_features,
)


def test_largest_divisor_at_most_handles_the_awkward_input_channel_count():
    # The classifier Conv has 712 input channels, which no power of two up to
    # 128 divides; blocked DequantizeLinear needs an exact divisor.
    assert largest_divisor_at_most(712, 32) == 8
    assert largest_divisor_at_most(256, 32) == 32
    assert largest_divisor_at_most(128, 128) == 128
    assert largest_divisor_at_most(7, 32) == 7


def test_blocked_uint4_round_trip_uses_the_whole_code_range():
    rng = np.random.default_rng(379)
    weights = rng.normal(size=(4, 64, 3, 3)).astype(np.float32)

    codes, scale, zero_point = quantize_blocked_uint4(weights, axis=1, block=32)

    assert codes.shape == weights.shape
    assert scale.shape == (4, 2, 3, 3)
    assert zero_point.shape == (4, 2, 3, 3)
    assert codes.min() == 0
    assert codes.max() == 15

    # Reconstruct the way DequantizeLinear does and confirm the error stays
    # inside half a quantization step for every block.
    blocked = codes.reshape(4, 2, 32, 3, 3).astype(np.float32)
    expanded_scale = scale.reshape(4, 2, 1, 3, 3)
    expanded_zero = zero_point.reshape(4, 2, 1, 3, 3).astype(np.float32)
    restored = ((blocked - expanded_zero) * expanded_scale).reshape(weights.shape)
    step = np.repeat(scale, 32, axis=1)
    assert np.all(np.abs(restored - weights) <= step / 2 + 1e-6)


def test_blocked_uint4_reproduces_a_constant_block_exactly():
    weights = np.full((2, 32, 1, 1), 0.25, dtype=np.float32)

    codes, scale, zero_point = quantize_blocked_uint4(weights, axis=1, block=32)

    restored = (codes.astype(np.float32) - zero_point.repeat(32, axis=1).astype(np.float32)) * scale.repeat(32, axis=1)
    np.testing.assert_allclose(restored, weights, atol=1e-7)


def test_quantize_int4_emits_a_blocked_four_bit_graph_with_float_io(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("ml_dtypes")

    output_path = tmp_path / "int4.onnx"
    layers = quantize_int4(DEFAULT_FLOAT32_ARTIFACT, output_path)
    model = onnx.load(output_path)

    assert max(entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx")) == INT4_OPSET_VERSION

    input_types = {value.type.tensor_type.elem_type for value in model.graph.input}
    output_types = {value.type.tensor_type.elem_type for value in model.graph.output}
    assert input_types == {onnx.TensorProto.FLOAT}
    assert output_types == {onnx.TensorProto.FLOAT}
    assert any(node.op_type == "Softmax" for node in model.graph.node)

    four_bit = [
        initializer
        for initializer in model.graph.initializer
        if initializer.data_type in (onnx.TensorProto.UINT4, onnx.TensorProto.INT4)
    ]
    assert len(four_bit) == 2 * len(layers), "each quantized Conv needs uint4 codes and a uint4 zero point"

    blocked = [
        node
        for node in model.graph.node
        if node.op_type == "DequantizeLinear" and any(attribute.name == "block_size" for attribute in node.attribute)
    ]
    assert len(blocked) == len(layers)

    # Each DequantizeLinear must precede the Conv that consumes it. Keying the
    # rewrite on protobuf object identity rather than position silently breaks
    # this on some protobuf builds, so assert the ordering directly.
    positions = {}
    for position, node in enumerate(model.graph.node):
        for name in node.output:
            positions[name] = position
    for position, node in enumerate(model.graph.node):
        for name in node.input:
            if name in positions:
                assert positions[name] < position, f"{name} is produced after the node consuming it"

    # Every Conv weight worth quantizing must actually be quantized, so a
    # future threshold change cannot quietly leave the big layers in float.
    assert all(layer["parameters"] >= MIN_PARAMS_TO_QUANTIZE for layer in layers)
    assert sum(layer["parameters"] for layer in layers) == 2_897_792


def test_committed_int4_artifact_is_smaller_than_int8_and_clears_the_frozen_gate():
    # onnxruntime 1.18 refuses the graph outright with "MLDataType for:
    # tensor(uint4) is not currently registered or supported", which is one of
    # the reasons int4 is not the shipped default. pyproject only requires
    # 1.18, so skip rather than fail where 4-bit cannot run at all.
    pytest.importorskip("onnxruntime", minversion=MINIMUM_ONNXRUNTIME_VERSION)

    int8_size = DEFAULT_WEIGHT_ONLY_ARTIFACT.stat().st_size
    int4_size = DEFAULT_INT4_ARTIFACT.stat().st_size
    assert int4_size < int8_size

    manifest = load_frozen_manifest(DEFAULT_FROZEN_MANIFEST)
    features = load_verified_feature_archive(DEFAULT_EVALUATION_FEATURES, manifest, "evaluation")
    teacher = predict_features(DEFAULT_FLOAT32_ARTIFACT, features)
    metrics = compare_predictions(teacher, predict_features(DEFAULT_INT4_ARTIFACT, features))

    assert metrics["sample_count"] == 217
    assert parity_gate_passes(metrics)
    assert metrics["top1_agreement"] >= MIN_TOP1_AGREEMENT
    assert metrics["keep_reject_agreement"] >= MIN_KEEP_REJECT_AGREEMENT


def test_int4_is_a_study_and_int8_remains_the_packaged_default():
    packaged = Path(__file__).parents[1] / "src" / "eegprep" / "plugins" / "ICLabel" / "iclabel.onnx"

    assert packaged.read_bytes() == DEFAULT_WEIGHT_ONLY_ARTIFACT.read_bytes()
    assert packaged.read_bytes() != DEFAULT_INT4_ARTIFACT.read_bytes()
