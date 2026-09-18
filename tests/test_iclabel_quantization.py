from pathlib import Path

import numpy as np
import pytest

from eegprep.plugins.ICLabel.pop_icflag import DEFAULT_ICFLAG_THRESHOLDS
from tools.iclabel.quantize_iclabel_onnx import (
    DEFAULT_CALIBRATED_ARTIFACT,
    DEFAULT_FLOAT32_ARTIFACT,
    DEFAULT_FROZEN_MANIFEST,
    DEFAULT_EVALUATION_FEATURES,
    DEFAULT_WEIGHT_ONLY_ARTIFACT,
    MAX_PROBABILITY_ABS_DIFF,
    MIN_KEEP_REJECT_AGREEMENT,
    MIN_TOP1_AGREEMENT,
    compare_predictions,
    evaluate_artifacts,
    load_frozen_manifest,
    load_verified_feature_archive,
    network_inputs_from_features,
    parity_gate_passes,
    predict_features,
    quantize_calibrated,
    quantize_weight_only,
    select_default_artifact,
)


def _reference_network_inputs(features):
    topo, psdmed, autocorr = features
    topo = np.single(np.concatenate([topo, -topo, topo[:, ::-1, :, :], -topo[:, ::-1, :, :]], axis=3))
    psdmed = np.single(np.tile(psdmed, (1, 1, 1, 4)))
    autocorr = np.single(np.tile(autocorr, (1, 1, 1, 4)))
    return {
        "image": np.transpose(topo, (3, 2, 0, 1)),
        "psdmed": np.transpose(psdmed, (3, 2, 0, 1)),
        "autocorr": np.transpose(autocorr, (3, 2, 0, 1)),
    }


def test_frozen_manifest_is_subject_balanced_and_calibration_disjoint():
    manifest = load_frozen_manifest(DEFAULT_FROZEN_MANIFEST)

    assert manifest["status"] == "frozen"
    evaluation = manifest["evaluation"]
    calibration = manifest["calibration"]
    evaluation_subjects = [item["subject"] for item in evaluation["recordings"]]
    calibration_subjects = [item["subject"] for item in calibration["recordings"]]

    assert evaluation_subjects == sorted(set(evaluation_subjects))
    assert calibration_subjects == sorted(set(calibration_subjects))
    assert set(evaluation_subjects).isdisjoint(calibration_subjects)
    assert all(item["component_indices"] == list(range(31)) for item in evaluation["recordings"])
    assert all(item["component_indices"] == list(range(31)) for item in calibration["recordings"])
    assert {item["source_path"] for item in evaluation["recordings"]}.isdisjoint(
        item["source_path"] for item in calibration["recordings"]
    )
    assert manifest["selection_policy"]["confidence_based_filtering"] is False
    assert manifest["selection_policy"]["subject_disjoint"] is True
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
    candidate[3] = [0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.40]

    metrics = compare_predictions(teacher, candidate)

    assert metrics["top1_agreement"] == pytest.approx(0.75)
    assert metrics["keep_reject_agreement"] == pytest.approx(1.0)
    assert metrics["max_probability_abs_diff"] == pytest.approx(0.50)
    assert metrics["mean_probability_abs_diff"] == pytest.approx(0.8 / 28)
    assert metrics["per_class_agreement"]["Brain"]["count"] == 1
    assert metrics["per_class_agreement"]["Brain"]["agreement"] == pytest.approx(1.0)
    assert metrics["per_class_agreement"]["Other"]["count"] == 1
    assert metrics["per_class_agreement"]["Other"]["agreement"] == pytest.approx(0.0)


def test_compare_predictions_exposes_rejection_threshold_boundaries():
    teacher = np.array(
        [
            [0.9001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0999],
            [0.0, 0.9999, 0.0, 0.0, 0.0, 0.0, 0.0001],
            [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.1],
        ]
    )
    candidate = np.array(
        [
            [0.8999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1001],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.9001, 0.0, 0.0, 0.0, 0.0999],
        ]
    )
    thresholds = np.array(
        [
            [0.9, 1.0],
            [0.9, 1.0],
            [0.9, 1.0],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
        ]
    )

    metrics = compare_predictions(teacher, candidate, thresholds)

    assert metrics["top1_agreement"] == pytest.approx(1.0)
    assert metrics["keep_reject_agreement"] == pytest.approx(0.0)


def test_frozen_feature_archive_hash_is_verified(tmp_path):
    manifest = load_frozen_manifest(DEFAULT_FROZEN_MANIFEST)
    with np.load(DEFAULT_EVALUATION_FEATURES, allow_pickle=False) as archive:
        features = {name: np.asarray(archive[name]).copy() for name in ("topo", "psd", "autocorr")}
    features["topo"][0, 0, 0, 0] += 1.0
    mutated = tmp_path / "evaluation_features.npz"
    np.savez(mutated, **features)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_verified_feature_archive(mutated, manifest, "evaluation")


def test_quantization_inputs_match_independent_reference_transform():
    manifest = load_frozen_manifest(DEFAULT_FROZEN_MANIFEST)
    features = load_verified_feature_archive(DEFAULT_EVALUATION_FEATURES, manifest, "evaluation")
    actual = network_inputs_from_features(features)
    expected = _reference_network_inputs(features)

    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_artifact_selection_falls_back_to_float32_until_an_int8_gate_passes():
    report = {
        "candidates": {
            "weight_only": {
                "artifact": "iclabel_int8_weight_only.onnx",
                "top1_agreement": 0.99,
                "keep_reject_agreement": 0.98,
                "max_probability_abs_diff": 0.01,
            },
            "calibrated": {
                "artifact": "iclabel_int8_calibrated.onnx",
                "top1_agreement": 0.99,
                "keep_reject_agreement": 0.98,
                "max_probability_abs_diff": 0.01,
            },
        }
    }

    assert select_default_artifact(report) == "iclabel.onnx"

    report["candidates"]["weight_only"]["keep_reject_agreement"] = 0.99
    assert select_default_artifact(report) == "iclabel_int8_weight_only.onnx"

    report["candidates"]["weight_only"]["keep_reject_agreement"] = 0.98
    report["candidates"]["calibrated"]["keep_reject_agreement"] = 0.99
    assert select_default_artifact(report) == "iclabel_int8_calibrated.onnx"


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


def test_packaged_artifact_matches_the_gate_selected_candidate():
    package_artifact = Path(__file__).parents[1] / "src" / "eegprep" / "plugins" / "ICLabel" / "iclabel.onnx"
    assert package_artifact.read_bytes() == DEFAULT_WEIGHT_ONLY_ARTIFACT.read_bytes()


def test_packaged_default_artifact_passes_the_semantic_gate():
    pytest.importorskip("onnxruntime")

    package_artifact = Path(__file__).parents[1] / "src" / "eegprep" / "plugins" / "ICLabel" / "iclabel.onnx"
    manifest = load_frozen_manifest(DEFAULT_FROZEN_MANIFEST)
    features = load_verified_feature_archive(DEFAULT_EVALUATION_FEATURES, manifest, "evaluation")
    teacher = predict_features(DEFAULT_FLOAT32_ARTIFACT, features)
    candidate = predict_features(package_artifact, features)

    assert parity_gate_passes(compare_predictions(teacher, candidate))


def test_committed_candidates_pass_evaluation_on_the_frozen_archive():
    pytest.importorskip("onnxruntime")

    report = evaluate_artifacts(
        DEFAULT_FLOAT32_ARTIFACT,
        DEFAULT_EVALUATION_FEATURES,
        {"weight_only": DEFAULT_WEIGHT_ONLY_ARTIFACT, "calibrated": DEFAULT_CALIBRATED_ARTIFACT},
    )

    assert report["float32_reference"]["sample_count"] == 217
    assert report["float32_reference"]["top1_agreement"] == pytest.approx(1.0)
    assert report["float32_reference"]["keep_reject_agreement"] == pytest.approx(1.0)
    assert report["feature_archives"]["evaluation"] == {
        "path": "tools/iclabel/evaluation_features.npz",
        "sha256": "dbb2c0cc063d29ab2e8e59fafb0f474df780a458ac17be9d4732abc0233c8b44",
        "component_count": 217,
    }
    assert report["default_artifact"] == "iclabel_int8_weight_only.onnx"
    assert report["gate"]["maximum_probability_abs_diff"] == MAX_PROBABILITY_ABS_DIFF

    expected_counts = {
        "Brain": 18,
        "Muscle": 1,
        "Eye": 5,
        "Heart": 0,
        "Line Noise": 0,
        "Channel Noise": 0,
        "Other": 193,
    }
    # ONNX Runtime can make platform-specific boundary choices for calibrated
    # int8 activations; the fixed promotion thresholds are the portable gate.
    for candidate_name in ("weight_only", "calibrated"):
        candidate = report["candidates"][candidate_name]
        assert candidate["gate_pass"] is (candidate_name == "weight_only")
        assert candidate["top1_agreement"] >= MIN_TOP1_AGREEMENT
        assert candidate["keep_reject_agreement"] >= MIN_KEEP_REJECT_AGREEMENT
        assert candidate["teacher_class_distribution"] == expected_counts
        assert set(candidate["per_class_agreement"]) == set(expected_counts)


def test_weight_only_matches_full_frozen_probabilities_and_threshold_boundaries():
    pytest.importorskip("onnxruntime")

    manifest = load_frozen_manifest(DEFAULT_FROZEN_MANIFEST)
    features = load_verified_feature_archive(DEFAULT_EVALUATION_FEATURES, manifest, "evaluation")
    teacher = predict_features(DEFAULT_FLOAT32_ARTIFACT, features)
    candidate = predict_features(DEFAULT_WEIGHT_ONLY_ARTIFACT, features)
    metrics = compare_predictions(teacher, candidate, DEFAULT_ICFLAG_THRESHOLDS)

    assert teacher.shape == (217, 7)
    assert metrics["max_probability_abs_diff"] <= MAX_PROBABILITY_ABS_DIFF
    assert metrics["mean_probability_abs_diff"] <= 0.001
    assert metrics["keep_reject_agreement"] == pytest.approx(1.0)


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
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

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
    model = onnx.load(output_path)
    input_types = {value.name: value.type.tensor_type.elem_type for value in model.graph.input}
    output_types = {value.name: value.type.tensor_type.elem_type for value in model.graph.output}
    assert set(input_types.values()) == {onnx.TensorProto.FLOAT}
    assert set(output_types.values()) == {onnx.TensorProto.FLOAT}
    assert any(node.op_type == "Softmax" for node in model.graph.node)
    (output,) = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"]).run(
        ["output"], calibration_inputs
    )
    assert output.shape == (4, 7, 1, 1)
    assert np.isfinite(output).all()
    np.testing.assert_allclose(output.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6)
