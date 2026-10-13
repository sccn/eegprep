import hashlib
import http.server
import io
import json
import threading
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import pytest

from eegprep.plugins.ICLabel.iclabel import postprocess_network_output, prepare_network_inputs
from eegprep.plugins.ICLabel.iclabel_net_onnx import run_iclabel_net
from tools.iclabel.freeze_iclabel_evaluation import _download_recording, _feature_archive_is_valid, _verify_recording
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
    report_mismatches,
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
    model_probabilities = teacher.copy()
    model_probabilities[3] = [0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.40]

    metrics = compare_predictions(teacher, model_probabilities)

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
    model_probabilities = np.array(
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

    metrics = compare_predictions(teacher, model_probabilities, thresholds)

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
        ("keep_reject_agreement", np.nan, "keep-or-reject agreement"),
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

    # The teacher came from ONNX Runtime float32 kernels and this reference from eager PyTorch
    # kernels on the same weights (measured on the 217 evaluation components: max abs diff
    # 1.19e-06, mean 7.6e-08). The bound is the one used for eager PyTorch versus the float32
    # ONNX export on the CI platforms and is over 100 times below the 0.015 probability gate,
    # so a different network would still fail.
    np.testing.assert_allclose(postprocess_network_output(output.numpy()), teacher, rtol=1e-4, atol=1e-5)


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

    assert gate_failures(report["metrics"]) == []
    assert report_mismatches(report, committed) == []
    assert {key: value for key, value in report.items() if key != "metrics"} == {
        key: value for key, value in committed.items() if key != "metrics"
    }


def test_report_mismatches_allow_only_cross_cpu_drift_noise():
    committed = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    drift = committed["metrics"]["max_probability_abs_diff"]

    def with_metric(key, value):
        return {**committed, "metrics": {**committed["metrics"], key: value}}

    assert report_mismatches(committed, committed) == []
    assert report_mismatches(with_metric("max_probability_abs_diff", drift + 1e-6), committed) == []
    assert report_mismatches(with_metric("max_probability_abs_diff", drift + 1e-4), committed) == [
        f"metrics.max_probability_abs_diff: {drift + 1e-4!r} != {drift!r}"
    ]
    distribution = {**committed["metrics"]["model_class_distribution"], "Brain": 17, "Other": 194}
    assert len(report_mismatches(with_metric("model_class_distribution", distribution), committed)) == 1
    stricter_gate = {**committed, "gate": {**committed["gate"], "maximum_probability_abs_diff": 0.01}}
    assert report_mismatches(stricter_gate, committed)[0].startswith("gate: ")


@pytest.fixture(scope="module")
def float32_export(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from tools.iclabel.export_iclabel_onnx import export, verify

    path = export(_NET_ICL_MAT, tmp_path_factory.mktemp("iclabel") / "iclabel_float32.onnx")
    # Same bound as export_iclabel_onnx.main().
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
            "--require-matching-report",
            str(DEFAULT_REPORT),
        ]
    )

    assert json.loads(report_path.read_text(encoding="utf-8"))["gate_pass"] is True
    model = onnx.load(output)
    tensor_types = {value.type.tensor_type.elem_type for value in (*model.graph.input, *model.graph.output)}
    assert tensor_types == {onnx.TensorProto.FLOAT}
    assert {"DequantizeLinear", "Softmax"} <= {node.op_type for node in model.graph.node}


def test_regeneration_fails_without_a_report_when_the_required_model_differs(float32_export, tmp_path):
    # The int8 model passes the gate but is not the float32 network it is
    # required to reproduce here.
    report_path = tmp_path / "parity_report.json"

    with pytest.raises(SystemExit, match="not numerically identical"):
        main(
            [
                "--float32-artifact",
                str(float32_export),
                "--output",
                str(tmp_path / "iclabel.onnx"),
                "--report",
                str(report_path),
                "--require-identical-to",
                str(float32_export),
            ]
        )

    assert not report_path.exists()


def test_regeneration_fails_without_a_report_when_the_committed_report_differs(float32_export, tmp_path):
    committed = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    stale = tmp_path / "stale_parity_report.json"
    stale.write_text(
        json.dumps({**committed, "metrics": {**committed["metrics"], "mean_probability_abs_diff": 0.002}}),
        encoding="utf-8",
    )
    report_path = tmp_path / "parity_report.json"

    with pytest.raises(SystemExit, match="mean_probability_abs_diff"):
        main(
            [
                "--float32-artifact",
                str(float32_export),
                "--output",
                str(tmp_path / "iclabel.onnx"),
                "--report",
                str(report_path),
                "--require-matching-report",
                str(stale),
            ]
        )

    assert not report_path.exists()


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


def _checkpoint_bytes(components=31, value=0.0):
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        topo=np.full((32, 32, 1, components), value, dtype=np.float32),
        psd=np.zeros((1, 100, 1, components), dtype=np.float32),
        autocorr=np.zeros((1, 100, 1, components), dtype=np.float32),
    )
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("contents", "reason"),
    (
        (b"", "is unreadable (EOFError"),
        (_checkpoint_bytes()[: len(_checkpoint_bytes()) // 2], "is unreadable (BadZipFile"),
        (b"not an npz archive", "is unreadable"),
        (_checkpoint_bytes(components=30), "has shapes"),
        (_checkpoint_bytes(value=np.nan), "contains non-finite values"),
    ),
    ids=("zero-byte", "truncated", "not-npz", "wrong-component-count", "non-finite"),
)
def test_freeze_logs_why_it_recomputes_a_feature_checkpoint(tmp_path, capsys, contents, reason):
    checkpoint = tmp_path / "sub-002.npz"
    checkpoint.write_bytes(contents)

    assert _feature_archive_is_valid(checkpoint) is False
    assert f"RECOMPUTE feature checkpoint {checkpoint} {reason}" in capsys.readouterr().err


def test_freeze_reuses_a_valid_feature_checkpoint_silently(tmp_path, capsys):
    checkpoint = tmp_path / "sub-002.npz"
    checkpoint.write_bytes(_checkpoint_bytes())

    assert _feature_archive_is_valid(checkpoint) is True
    assert capsys.readouterr().err == ""


_RECORDING_BYTES = b"EEGLAB set file stand-in" * 64
_RECORDING = {
    "subject": "sub-002",
    "source_path": "sub-002/ses-01/eeg/sub-002_ses-01_task-gonogo_run-1_eeg.set",
    "git_annex_key": "MD5E-s1536--recording.set",
    "size_bytes": len(_RECORDING_BYTES),
    "md5": hashlib.md5(_RECORDING_BYTES, usedforsecurity=False).hexdigest(),
}
_CORRUPT_RECORDING_BYTES = b"X" * len(_RECORDING_BYTES)


@pytest.mark.parametrize(
    ("contents", "message"),
    ((_RECORDING_BYTES[:-1], "Size mismatch"), (_CORRUPT_RECORDING_BYTES, "MD5 mismatch")),
)
def test_freeze_rejects_a_recording_with_the_wrong_size_or_checksum(tmp_path, contents, message):
    path = tmp_path / "recording.set"
    path.write_bytes(contents)

    with pytest.raises(ValueError, match=message):
        _verify_recording(path, _RECORDING)


def test_freeze_accepts_a_recording_that_matches_the_manifest(tmp_path):
    path = tmp_path / "recording.set"
    path.write_bytes(_RECORDING_BYTES)

    _verify_recording(path, _RECORDING)


class _RecordingServer:
    """Local HTTP server that answers each path with a queue of (status, body) responses."""

    def __init__(self, responses):
        self.responses = {path: list(queue) for path, queue in responses.items()}
        self.requests = []
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                server.requests.append(self.path)
                status, body = server.responses[self.path].pop(0)
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._httpd.shutdown()
        self._httpd.server_close()

    def manifest(self):
        return {
            "dataset": {
                "s3_object_url_template": self.url + "/nemar/{git_annex_key}",
                "openneuro_s3_url_template": self.url + "/openneuro/{source_path}",
            }
        }


_NEMAR_PATH = "/nemar/" + _RECORDING["git_annex_key"]
_OPENNEURO_PATH = "/openneuro/" + _RECORDING["source_path"]


@pytest.mark.parametrize(
    ("responses", "expected_requests"),
    (
        ({_NEMAR_PATH: [(200, _RECORDING_BYTES)]}, [_NEMAR_PATH]),
        (
            {_NEMAR_PATH: [(200, _CORRUPT_RECORDING_BYTES), (200, _CORRUPT_RECORDING_BYTES), (200, _RECORDING_BYTES)]},
            [_NEMAR_PATH] * 3,
        ),
        ({_NEMAR_PATH: [(404, b"")], _OPENNEURO_PATH: [(200, _RECORDING_BYTES)]}, [_NEMAR_PATH, _OPENNEURO_PATH]),
        (
            {_NEMAR_PATH: [(200, _CORRUPT_RECORDING_BYTES)] * 3, _OPENNEURO_PATH: [(200, _RECORDING_BYTES)]},
            [_NEMAR_PATH] * 3 + [_OPENNEURO_PATH],
        ),
    ),
    ids=("first-try", "retry-after-checksum-mismatch", "fallback-after-404", "fallback-after-three-mismatches"),
)
def test_freeze_download_retries_and_falls_back_until_the_checksum_matches(tmp_path, responses, expected_requests):
    path = tmp_path / "data" / _RECORDING["source_path"]

    with _RecordingServer(responses) as server:
        _download_recording(server.manifest(), _RECORDING, path)

    assert server.requests == expected_requests
    assert path.read_bytes() == _RECORDING_BYTES
    assert not path.with_name(path.name + ".part").exists()


def test_freeze_download_fails_when_both_sources_serve_bad_data(tmp_path):
    path = tmp_path / "data" / _RECORDING["source_path"]
    responses = {
        _NEMAR_PATH: [(200, _CORRUPT_RECORDING_BYTES)] * 3,
        _OPENNEURO_PATH: [(200, _CORRUPT_RECORDING_BYTES)] * 3,
    }

    with _RecordingServer(responses) as server, pytest.raises(ValueError, match="MD5 mismatch"):
        _download_recording(server.manifest(), _RECORDING, path)

    assert not path.exists()
    assert not path.with_name(path.name + ".part").exists()


def test_freeze_download_does_not_fall_back_after_a_server_error(tmp_path):
    path = tmp_path / "data" / _RECORDING["source_path"]

    with _RecordingServer({_NEMAR_PATH: [(500, b"")]}) as server, pytest.raises(HTTPError):
        _download_recording(server.manifest(), _RECORDING, path)

    assert server.requests == [_NEMAR_PATH]
    assert not path.exists()


def test_freeze_download_reuses_a_verified_recording_without_a_request(tmp_path):
    path = tmp_path / "data" / _RECORDING["source_path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(_RECORDING_BYTES)

    with _RecordingServer({}) as server:
        _download_recording(server.manifest(), _RECORDING, path)

    assert server.requests == []
