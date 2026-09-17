import numpy as np
import pytest

from eegprep.plugins.ICLabel.pop_icflag import DEFAULT_ICFLAG_THRESHOLDS
from tools.iclabel.quantize_iclabel_onnx import (
    DEFAULT_FLOAT32_ARTIFACT,
    DEFAULT_FROZEN_MANIFEST,
    compare_predictions,
    load_frozen_manifest,
    quantize_calibrated,
    quantize_weight_only,
    select_default_artifact,
)


def test_frozen_manifest_is_subject_balanced_and_calibration_disjoint():
    manifest = load_frozen_manifest(DEFAULT_FROZEN_MANIFEST)

    assert manifest["status"] == "frozen"
    evaluation = manifest["evaluation"]
    calibration = manifest["calibration"]
    evaluation_subjects = [item["subject"] for item in evaluation["recordings"]]
    calibration_subjects = [item["subject"] for item in calibration["recordings"]]

    assert evaluation_subjects == sorted(set(evaluation_subjects))
    assert calibration_subjects == evaluation_subjects
    assert all(item["component_indices"] == list(range(31)) for item in evaluation["recordings"])
    assert all(item["component_indices"] == list(range(31)) for item in calibration["recordings"])
    assert {item["source_path"] for item in evaluation["recordings"]}.isdisjoint(
        item["source_path"] for item in calibration["recordings"]
    )
    assert manifest["selection_policy"]["confidence_based_filtering"] is False
    assert manifest["selection_policy"]["recordings_per_subject"] == 1


def test_compare_predictions_reports_overall_and_per_class_agreement():
    teacher = np.array(
        [
            [0.02, 0.94, 0.01, 0.01, 0.01, 0.005, 0.005],
            [0.02, 0.01, 0.94, 0.01, 0.01, 0.005, 0.005],
            [0.80, 0.03, 0.03, 0.03, 0.03, 0.04, 0.04],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.70],
        ]
    )
    candidate = teacher.copy()
    candidate[3] = [0.35, 0.05, 0.05, 0.05, 0.05, 0.05, 0.40]

    metrics = compare_predictions(teacher, candidate)

    assert metrics["top1_agreement"] == pytest.approx(0.75)
    assert metrics["keep_reject_agreement"] == pytest.approx(1.0)
    assert metrics["per_class_agreement"]["Brain"]["count"] == 1
    assert metrics["per_class_agreement"]["Brain"]["agreement"] == pytest.approx(1.0)
    assert metrics["per_class_agreement"]["Other"]["count"] == 1
    assert metrics["per_class_agreement"]["Other"]["agreement"] == pytest.approx(0.0)


def test_artifact_selection_falls_back_to_float32_until_gate_passes():
    report = {
        "candidates": {
            "weight_only": {
                "artifact": "iclabel_int8_weight_only.onnx",
                "top1_agreement": 0.99,
                "keep_reject_agreement": 0.98,
            },
            "calibrated": {
                "artifact": "iclabel_int8_calibrated.onnx",
                "top1_agreement": 0.99,
                "keep_reject_agreement": 0.99,
            },
        }
    }

    assert select_default_artifact(report) == "iclabel.onnx"

    report["candidates"]["weight_only"]["keep_reject_agreement"] = 0.99
    assert select_default_artifact(report) == "iclabel_int8_weight_only.onnx"


def test_thresholds_remain_the_existing_open_interval_defaults():
    expected = np.array(
        [
            [np.nan, np.nan],
            [0.9, 1.0],
            [0.9, 1.0],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
        ]
    )
    np.testing.assert_equal(DEFAULT_ICFLAG_THRESHOLDS, expected)


def test_weight_only_quantization_preserves_float_io_and_softmax(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    output_path = tmp_path / "weight-only.onnx"
    quantize_weight_only(DEFAULT_FLOAT32_ARTIFACT, output_path)
    model = onnx.load(output_path)

    input_types = {value.name: value.type.tensor_type.elem_type for value in model.graph.input}
    output_types = {value.name: value.type.tensor_type.elem_type for value in model.graph.output}
    assert set(input_types.values()) == {onnx.TensorProto.FLOAT}
    assert set(output_types.values()) == {onnx.TensorProto.FLOAT}
    assert any(node.op_type == "DequantizeLinear" for node in model.graph.node)
    assert any(node.op_type == "Softmax" for node in model.graph.node)

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    inputs = {
        "image": np.zeros((2, 1, 32, 32), dtype=np.float32),
        "psdmed": np.zeros((2, 1, 1, 100), dtype=np.float32),
        "autocorr": np.zeros((2, 1, 1, 100), dtype=np.float32),
    }
    (output,) = session.run(["output"], inputs)
    assert output.shape == (2, 7, 1, 1)
    assert np.isfinite(output).all()
    np.testing.assert_allclose(output.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6)


def test_calibrated_quantization_consumes_float_feature_inputs(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    rng = np.random.default_rng(379)
    calibration_inputs = {
        "image": rng.standard_normal((4, 1, 32, 32)).astype(np.float32),
        "psdmed": rng.standard_normal((4, 1, 1, 100)).astype(np.float32),
        "autocorr": rng.standard_normal((4, 1, 1, 100)).astype(np.float32),
    }
    output_path = tmp_path / "calibrated.onnx"

    quantize_calibrated(DEFAULT_FLOAT32_ARTIFACT, output_path, calibration_inputs)

    assert output_path.exists()
    assert output_path.stat().st_size < DEFAULT_FLOAT32_ARTIFACT.stat().st_size
