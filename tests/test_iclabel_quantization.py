import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from eegprep.plugins.ICLabel.iclabel import postprocess_network_output, prepare_network_inputs
from eegprep.plugins.ICLabel.iclabel_net_onnx import run_iclabel_net
from tools.iclabel.quantize_iclabel_onnx import (
    DEFAULT_EVALUATION_FEATURES,
    DEFAULT_FROZEN_MANIFEST,
    DEFAULT_REPORT,
    DEFAULT_TEACHER_PROBABILITIES,
    MAX_PROBABILITY_ABS_DIFF,
    MIN_KEEP_REJECT_AGREEMENT,
    MIN_TOP1_AGREEMENT,
    PACKAGED_MODEL,
    compare_predictions,
    evaluate_model,
    gate_failures,
    load_frozen_manifest,
    load_verified_feature_archive,
    load_verified_teacher,
    main,
    network_inputs_from_features,
)


_NET_ICL_MAT = Path(__file__).parents[1] / "src" / "eegprep" / "plugins" / "ICLabel" / "netICL.mat"
_METRICS_AT_GATE_BOUNDARY = {
    "top1_agreement": MIN_TOP1_AGREEMENT,
    "keep_reject_agreement": MIN_KEEP_REJECT_AGREEMENT,
    "max_probability_abs_diff": MAX_PROBABILITY_ABS_DIFF,
}


def _reference_network_inputs(features):
    """Independent oracle for the production ICLabel augmentation and NCHW layout."""
    topo, psdmed, autocorr = features
    topo = np.single(np.concatenate([topo, -topo, topo[:, ::-1, :, :], -topo[:, ::-1, :, :]], axis=3))
    psdmed = np.single(np.tile(psdmed, (1, 1, 1, 4)))
    autocorr = np.single(np.tile(autocorr, (1, 1, 1, 4)))
    return {
        "image": np.transpose(topo, (3, 2, 0, 1)),
        "psdmed": np.transpose(psdmed, (3, 2, 0, 1)),
        "autocorr": np.transpose(autocorr, (3, 2, 0, 1)),
    }


def _frozen_inputs():
    manifest = load_frozen_manifest(DEFAULT_FROZEN_MANIFEST)
    features = load_verified_feature_archive(DEFAULT_EVALUATION_FEATURES, manifest)
    return manifest, features, load_verified_teacher(DEFAULT_TEACHER_PROBABILITIES, manifest)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda manifest: manifest["selection_policy"].update(label_based_selection=True), "label filtering"),
        (lambda manifest: manifest["selection_policy"].update(confidence_based_filtering=True), "confidence"),
        (
            lambda manifest: manifest["evaluation"]["recordings"][0].update(component_indices=list(range(30))),
            "retain all 31 components",
        ),
        (lambda manifest: manifest["evaluation"]["recordings"].reverse(), "unique and sorted"),
        (lambda manifest: manifest["evaluation"]["teacher_probabilities"].pop("sha256"), "teacher_probabilities"),
    ),
)
def test_manifest_validation_rejects_biased_or_incomplete_selection(tmp_path, mutate, message):
    manifest = json.loads(DEFAULT_FROZEN_MANIFEST.read_text(encoding="utf-8"))
    mutate(manifest)
    path = tmp_path / "evaluation_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_frozen_manifest(path)


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


def test_gate_passes_with_every_metric_on_its_boundary():
    assert gate_failures(_METRICS_AT_GATE_BOUNDARY) == []


@pytest.mark.parametrize(
    ("metric", "value", "message"),
    (
        ("top1_agreement", np.nextafter(MIN_TOP1_AGREEMENT, -np.inf), "top-1 agreement"),
        ("keep_reject_agreement", np.nextafter(MIN_KEEP_REJECT_AGREEMENT, -np.inf), "keep-or-reject agreement"),
        ("max_probability_abs_diff", np.nextafter(MAX_PROBABILITY_ABS_DIFF, np.inf), "maximum probability"),
        ("top1_agreement", np.nan, "top-1 agreement"),
        ("max_probability_abs_diff", np.nan, "maximum probability"),
    ),
)
def test_each_gate_criterion_fails_on_its_own(metric, value, message):
    failures = gate_failures({**_METRICS_AT_GATE_BOUNDARY, metric: value})

    assert len(failures) == 1
    assert message in failures[0]


def test_frozen_feature_archive_hash_is_verified(tmp_path):
    manifest = load_frozen_manifest(DEFAULT_FROZEN_MANIFEST)
    with np.load(DEFAULT_EVALUATION_FEATURES, allow_pickle=False) as archive:
        features = {name: np.asarray(archive[name]).copy() for name in ("topo", "psd", "autocorr")}
    features["topo"][0, 0, 0, 0] += 1.0
    mutated = tmp_path / "evaluation_features.npz"
    np.savez(mutated, **features)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_verified_feature_archive(mutated, manifest)


def test_frozen_teacher_hash_is_verified(tmp_path):
    manifest = load_frozen_manifest(DEFAULT_FROZEN_MANIFEST)
    teacher = np.load(DEFAULT_TEACHER_PROBABILITIES)
    teacher[0, 0] += np.float32(1e-3)
    mutated = tmp_path / "evaluation_teacher_probabilities.npy"
    np.save(mutated, teacher)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_verified_teacher(mutated, manifest)


def test_frozen_teacher_matches_the_netICL_network():
    torch = pytest.importorskip("torch")
    from eegprep.plugins.ICLabel.iclabel_net import ICLabelNet

    _, features, teacher = _frozen_inputs()
    model = ICLabelNet(str(_NET_ICL_MAT))
    model.eval()
    with torch.no_grad():
        output = model(*(torch.from_numpy(np.ascontiguousarray(array)) for array in prepare_network_inputs(features)))

    # The teacher came from ONNX Runtime float32 kernels and this reference
    # from eager PyTorch kernels on the same weights. Measured on the 217
    # evaluation components: max abs diff 1.19e-06, mean 7.6e-08, the usual
    # float32 reordering gap between two independent implementations. 1e-5
    # leaves room for other CPUs while staying 1,500 times below the
    # 0.015 probability gate, so a different network would fail here.
    np.testing.assert_allclose(postprocess_network_output(output.numpy()), teacher, rtol=0, atol=1e-5)


def test_production_prepare_inputs_match_independent_reference():
    _, features, _ = _frozen_inputs()
    actual = network_inputs_from_features(features)
    expected = _reference_network_inputs(features)

    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_parity_report_records_the_packaged_model_and_frozen_inputs():
    report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    manifest = load_frozen_manifest(DEFAULT_FROZEN_MANIFEST)
    model_bytes = PACKAGED_MODEL.read_bytes()

    assert report["packaged_model"]["sha256"] == hashlib.sha256(model_bytes).hexdigest()
    assert report["packaged_model"]["size_bytes"] == len(model_bytes)
    assert report["evaluation_features"]["sha256"] == manifest["evaluation"]["feature_archive"]["sha256"]
    assert report["teacher_probabilities"]["sha256"] == manifest["evaluation"]["teacher_probabilities"]["sha256"]
    assert report["gate_pass"] is True
    assert gate_failures(report["metrics"]) == []


def test_production_runtime_passes_the_frozen_gate():
    pytest.importorskip("onnxruntime")
    manifest, features, teacher = _frozen_inputs()

    probabilities = postprocess_network_output(run_iclabel_net(*prepare_network_inputs(features)))
    metrics = compare_predictions(teacher, probabilities)

    assert gate_failures(metrics) == []
    assert (
        metrics["teacher_class_distribution"] == manifest["evaluation"]["teacher_probabilities"]["class_distribution"]
    )


def test_evaluate_model_reproduces_the_committed_report_for_the_packaged_model():
    pytest.importorskip("onnxruntime")
    committed = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))

    report = evaluate_model(PACKAGED_MODEL)

    # Probability metrics can differ in the last float32 bits across CPUs, so
    # only the gate verdict and the platform-independent fields must match.
    assert gate_failures(report["metrics"]) == []
    assert {key: value for key, value in report.items() if key != "metrics"} == {
        key: value for key, value in committed.items() if key != "metrics"
    }


@pytest.fixture(scope="module")
def float32_export(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from tools.iclabel.export_iclabel_onnx import export, verify

    path = export(_NET_ICL_MAT, tmp_path_factory.mktemp("iclabel") / "iclabel_float32.onnx")
    # export_iclabel_onnx.main() applies the same bound; measured 1.0e-05 on its random inputs.
    assert verify(_NET_ICL_MAT, path) <= 1e-4
    return path


def test_regeneration_from_netICL_reproduces_the_packaged_model(float32_export, tmp_path):
    import onnx

    report_path = tmp_path / "parity_report.json"
    output = tmp_path / "iclabel.onnx"

    main(
        [
            "--float32-artifact",
            str(float32_export),
            "--output",
            str(output),
            "--report",
            str(report_path),
            "--require-identical-to",
            str(PACKAGED_MODEL),
        ]
    )

    assert json.loads(report_path.read_text(encoding="utf-8"))["gate_pass"] is True
    model = onnx.load(output)
    tensor_types = {value.type.tensor_type.elem_type for value in (*model.graph.input, *model.graph.output)}
    assert tensor_types == {onnx.TensorProto.FLOAT}
    assert {"DequantizeLinear", "Softmax"} <= {node.op_type for node in model.graph.node}


def test_quantize_exits_nonzero_without_a_report_when_the_gate_fails(float32_export, tmp_path):
    import onnx
    from onnx import numpy_helper

    # Scaling the final Conv weights scales the logits, which moves the
    # softmax probabilities far beyond the 0.015 gate.
    model = onnx.load(float32_export)
    final_conv = [node for node in model.graph.node if node.op_type == "Conv"][-1]
    weights = next(initializer for initializer in model.graph.initializer if initializer.name == final_conv.input[1])
    weights.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(weights) * np.float32(1.5), name=weights.name))
    perturbed = tmp_path / "perturbed_float32.onnx"
    onnx.save(model, perturbed)
    report_path = tmp_path / "parity_report.json"

    with pytest.raises(SystemExit, match="failed the frozen gate"):
        main(
            ["--float32-artifact", str(perturbed), "--output", str(tmp_path / "out.onnx"), "--report", str(report_path)]
        )

    assert not report_path.exists()
