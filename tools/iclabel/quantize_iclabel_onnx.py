"""Build and evaluate ICLabel int8 ONNX artifacts.

This developer-only tool deliberately leaves ICLabel feature extraction,
normalization, augmentation, and softmax outside quantization. The weight-only
candidate stores Conv weights as int8 with float dequantization before each
Conv. The calibrated candidate uses ONNX Runtime static QDQ quantization for
Conv weights and activations, while preserving float model inputs and output
softmax.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from eegprep.plugins.ICLabel.eeg_icflag import eeg_icflag
from eegprep.plugins.ICLabel.pop_icflag import DEFAULT_ICFLAG_THRESHOLDS


CLASS_NAMES = ("Brain", "Muscle", "Eye", "Heart", "Line Noise", "Channel Noise", "Other")
MIN_TOP1_AGREEMENT = 0.95
MIN_KEEP_REJECT_AGREEMENT = 0.99
_INPUT_NAMES = ("image", "psdmed", "autocorr")
_OUTPUT_NAME = "output"
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FLOAT32_ARTIFACT = _REPO_ROOT / "src" / "eegprep" / "plugins" / "ICLabel" / "iclabel.onnx"
DEFAULT_FROZEN_MANIFEST = Path(__file__).with_name("evaluation_manifest.json")
DEFAULT_EVALUATION_FEATURES = Path(__file__).with_name("evaluation_features.npz")
DEFAULT_CALIBRATION_FEATURES = Path(__file__).with_name("calibration_features.npz")
DEFAULT_ARTIFACT_DIR = Path(__file__).with_name("artifacts")
DEFAULT_WEIGHT_ONLY_ARTIFACT = DEFAULT_ARTIFACT_DIR / "iclabel_int8_weight_only.onnx"
DEFAULT_CALIBRATED_ARTIFACT = DEFAULT_ARTIFACT_DIR / "iclabel_int8_calibrated.onnx"
DEFAULT_REPORT = Path(__file__).with_name("parity_report.json")


def load_frozen_manifest(path: Path = DEFAULT_FROZEN_MANIFEST) -> dict[str, Any]:
    """Load and validate the committed, pre-quantization evaluation manifest."""
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    _validate_frozen_manifest(manifest)
    return manifest


def _validate_frozen_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("status") != "frozen":
        raise ValueError("ICLabel evaluation manifest must have status='frozen'")
    policy = manifest.get("selection_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("ICLabel evaluation manifest is missing selection_policy")
    if policy.get("confidence_based_filtering") is not False:
        raise ValueError("ICLabel evaluation selection cannot use confidence filtering")
    if policy.get("label_based_selection") is not False:
        raise ValueError("ICLabel evaluation selection cannot use label filtering")
    if policy.get("selection_declared_before_quantized_evaluation") is not True:
        raise ValueError("ICLabel evaluation selection must be declared before quantized evaluation")

    splits = {}
    for split_name in ("evaluation", "calibration"):
        split = manifest.get(split_name)
        if not isinstance(split, Mapping) or not isinstance(split.get("recordings"), list):
            raise ValueError(f"ICLabel manifest is missing {split_name} recordings")
        splits[split_name] = split["recordings"]
        for recording in split["recordings"]:
            if not isinstance(recording, Mapping):
                raise ValueError(f"ICLabel {split_name} recording must be an object")
            if recording.get("component_indices") != list(range(31)):
                raise ValueError(f"ICLabel {split_name} recordings must retain all 31 components")
            for field in ("subject", "source_path", "git_annex_key", "size_bytes", "md5"):
                if field not in recording:
                    raise ValueError(f"ICLabel {split_name} recording is missing {field}")

    evaluation_subjects = [recording["subject"] for recording in splits["evaluation"]]
    calibration_subjects = [recording["subject"] for recording in splits["calibration"]]
    if evaluation_subjects != sorted(set(evaluation_subjects)):
        raise ValueError("ICLabel evaluation subjects must be unique and sorted")
    if calibration_subjects != evaluation_subjects:
        raise ValueError("ICLabel calibration must cover the same subjects")
    evaluation_paths = {recording["source_path"] for recording in splits["evaluation"]}
    calibration_paths = {recording["source_path"] for recording in splits["calibration"]}
    if evaluation_paths.intersection(calibration_paths):
        raise ValueError("ICLabel evaluation and calibration recordings must be disjoint")


def load_feature_archive(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load saved float32 topography, PSD, and autocorrelation features."""
    with np.load(path, allow_pickle=False) as archive:
        features = tuple(np.asarray(archive[name], dtype=np.float32) for name in ("topo", "psd", "autocorr"))
    _validate_feature_arrays(features)
    return features


def _validate_feature_arrays(features: Sequence[np.ndarray]) -> None:
    if len(features) != 3:
        raise ValueError("ICLabel feature archive must contain topo, psd, and autocorr arrays")
    shapes = [array.shape for array in features]
    if len(shapes[0]) != 4 or shapes[0][:3] != (32, 32, 1):
        raise ValueError(f"ICLabel topography features must have shape (32, 32, 1, n), got {shapes[0]}")
    if len(shapes[1]) != 4 or len(shapes[2]) != 4 or shapes[1][:3] != (1, 100, 1) or shapes[2][:3] != (1, 100, 1):
        raise ValueError(f"ICLabel PSD/autocorrelation features must have shape (1, 100, 1, n), got {shapes[1:]}")
    if shapes[0][3] == 0 or not (shapes[0][3] == shapes[1][3] == shapes[2][3]):
        raise ValueError("ICLabel feature arrays must contain the same component count")
    if not all(np.isfinite(array).all() for array in features):
        raise ValueError("ICLabel feature arrays must be finite")


def network_inputs_from_features(features: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """Apply the unchanged ICLabel augmentation and NCHW conversion."""
    _validate_feature_arrays(features)
    topo, psd, autocorr = features
    topo = np.single(np.concatenate([topo, -topo, topo[:, ::-1, :, :], -topo[:, ::-1, :, :]], axis=3))
    psd = np.single(np.tile(psd, (1, 1, 1, 4)))
    autocorr = np.single(np.tile(autocorr, (1, 1, 1, 4)))
    return {
        "image": np.transpose(topo, (3, 2, 0, 1)),
        "psdmed": np.transpose(psd, (3, 2, 0, 1)),
        "autocorr": np.transpose(autocorr, (3, 2, 0, 1)),
    }


def _run_network(model_path: Path, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    (output,) = session.run([_OUTPUT_NAME], dict(inputs))
    return np.asarray(output, dtype=np.float32)


def predict_features(model_path: Path, features: Sequence[np.ndarray]) -> np.ndarray:
    """Run an ICLabel artifact and return one 7-class row per component."""
    output = _run_network(model_path, network_inputs_from_features(features))
    output = output.T
    output = np.reshape(output, (-1, 4), order="F")
    output = np.mean(output, axis=1)
    output = np.reshape(output, (7, -1), order="F")
    return output.T


def _rejection_flags(classifications: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    eeg = {"etc": {"ic_classification": {"ICLabel": {"classifications": classifications}}}}
    return np.asarray(eeg_icflag(eeg, thresholds)["reject"]["gcompreject"], dtype=bool)


def compare_predictions(
    teacher: np.ndarray,
    candidate: np.ndarray,
    thresholds: np.ndarray = DEFAULT_ICFLAG_THRESHOLDS,
) -> dict[str, Any]:
    """Compare class labels and exact ICLabel keep/reject decisions."""
    teacher = np.asarray(teacher, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if teacher.shape != candidate.shape or teacher.ndim != 2 or teacher.shape[1] != len(CLASS_NAMES):
        raise ValueError("ICLabel predictions must be matching (components, 7) arrays")
    if not np.isfinite(teacher).all() or not np.isfinite(candidate).all():
        raise ValueError("ICLabel predictions must be finite")

    teacher_labels = np.argmax(teacher, axis=1)
    candidate_labels = np.argmax(candidate, axis=1)
    teacher_reject = _rejection_flags(teacher, thresholds)
    candidate_reject = _rejection_flags(candidate, thresholds)
    per_class: dict[str, dict[str, Any]] = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        selected = teacher_labels == class_index
        count = int(np.sum(selected))
        per_class[class_name] = {
            "count": count,
            "agreement": None if count == 0 else float(np.mean(candidate_labels[selected] == class_index)),
        }

    return {
        "sample_count": int(teacher.shape[0]),
        "top1_agreement": float(np.mean(teacher_labels == candidate_labels)),
        "keep_reject_agreement": float(np.mean(teacher_reject == candidate_reject)),
        "teacher_class_distribution": {
            name: int(np.sum(teacher_labels == index)) for index, name in enumerate(CLASS_NAMES)
        },
        "candidate_class_distribution": {
            name: int(np.sum(candidate_labels == index)) for index, name in enumerate(CLASS_NAMES)
        },
        "per_class_agreement": per_class,
    }


def parity_gate_passes(metrics: Mapping[str, Any]) -> bool:
    """Return whether the issue #379 top-1 and keep/reject gates pass."""
    return (
        float(metrics["top1_agreement"]) >= MIN_TOP1_AGREEMENT
        and float(metrics["keep_reject_agreement"]) >= MIN_KEEP_REJECT_AGREEMENT
    )


def select_default_artifact(report: Mapping[str, Any]) -> str:
    """Select the first int8 candidate that passes the fixed parity gate."""
    candidates = report.get("candidates", {})
    for name in ("weight_only", "calibrated"):
        candidate = candidates.get(name, {})
        if parity_gate_passes(candidate):
            return str(candidate["artifact"])
    return "iclabel.onnx"


def quantize_weight_only(model_input: Path, model_output: Path) -> Path:
    """Write a Conv-weight-only int8 QDQ artifact with float inputs and outputs."""
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(model_input))
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    replacement_initializers = []
    quantized_weight_names = set()
    dequant_nodes = {}
    graph_nodes = list(model.graph.node)
    for node in graph_nodes:
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        weight_name = node.input[1]
        initializer = initializers.get(weight_name)
        if initializer is None:
            raise ValueError(f"ICLabel Conv weight {weight_name!r} is not an initializer")
        weights = numpy_helper.to_array(initializer).astype(np.float32, copy=False)
        axes = tuple(range(1, weights.ndim))
        scale = np.max(np.abs(weights), axis=axes) / 127.0
        scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
        reshape = (scale.shape[0],) + (1,) * (weights.ndim - 1)
        quantized = np.clip(np.rint(weights / scale.reshape(reshape)), -127, 127).astype(np.int8)
        int8_name = f"{weight_name}_int8"
        scale_name = f"{weight_name}_scale"
        zero_point_name = f"{weight_name}_zero_point"
        dequant_name = f"{weight_name}_dequantized"
        quantized_weight_names.add(weight_name)
        replacement_initializers.extend(
            [
                numpy_helper.from_array(quantized, name=int8_name),
                numpy_helper.from_array(scale, name=scale_name),
                numpy_helper.from_array(np.zeros(scale.shape, dtype=np.int8), name=zero_point_name),
            ]
        )
        dequant_nodes[id(node)] = helper.make_node(
            "DequantizeLinear",
            [int8_name, scale_name, zero_point_name],
            [dequant_name],
            name=f"{weight_name}_dequantize",
            axis=0,
        )
        node.input[1] = dequant_name

    if not dequant_nodes:
        raise ValueError("ICLabel model contains no Conv weights to quantize")
    kept_initializers = [
        initializer for initializer in model.graph.initializer if initializer.name not in quantized_weight_names
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers + replacement_initializers)
    nodes = []
    for node in graph_nodes:
        nodes.append(dequant_nodes.get(id(node)))
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(node for node in nodes if node is not None)
    onnx.checker.check_model(model)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(model_output))
    return model_output


class _CalibrationDataReader:
    def __init__(self, inputs: Mapping[str, np.ndarray]):
        self._inputs = {name: np.asarray(inputs[name], dtype=np.float32) for name in _INPUT_NAMES}
        batches = {array.shape[0] for array in self._inputs.values()}
        if len(batches) != 1 or not batches:
            raise ValueError("Calibration inputs must have matching non-empty batch dimensions")
        self._read = False

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._read:
            return None
        self._read = True
        return self._inputs


def quantize_calibrated(
    model_input: Path,
    model_output: Path,
    calibration_inputs: Mapping[str, np.ndarray],
) -> Path:
    """Write a calibrated int8 Conv QDQ artifact from float feature inputs."""
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    model_output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(model_input),
        model_output=str(model_output),
        calibration_data_reader=_CalibrationDataReader(calibration_inputs),
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv"],
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    return model_output


def evaluate_artifacts(
    float32_artifact: Path,
    evaluation_features: Path,
    candidate_artifacts: Mapping[str, Path],
    manifest: Path = DEFAULT_FROZEN_MANIFEST,
) -> dict[str, Any]:
    """Evaluate candidates against the float32 teacher on the frozen archive."""
    load_frozen_manifest(manifest)
    features = load_feature_archive(evaluation_features)
    teacher = predict_features(float32_artifact, features)
    thresholds = np.asarray(DEFAULT_ICFLAG_THRESHOLDS, dtype=float)
    report: dict[str, Any] = {
        "manifest": str(Path(manifest).name),
        "class_names": list(CLASS_NAMES),
        "thresholds": [[None if np.isnan(value) else float(value) for value in row] for row in thresholds],
        "gate": {
            "minimum_top1_agreement": MIN_TOP1_AGREEMENT,
            "minimum_keep_reject_agreement": MIN_KEEP_REJECT_AGREEMENT,
        },
        "float32_reference": {
            "artifact": Path(float32_artifact).name,
            "size_bytes": Path(float32_artifact).stat().st_size,
            **compare_predictions(teacher, teacher, thresholds),
        },
        "candidates": {},
    }
    for name, artifact in candidate_artifacts.items():
        metrics = compare_predictions(teacher, predict_features(artifact, features), thresholds)
        report["candidates"][name] = {
            "artifact": Path(artifact).name,
            "size_bytes": Path(artifact).stat().st_size,
            "gate_pass": parity_gate_passes(metrics),
            **metrics,
        }
    report["default_artifact"] = select_default_artifact(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--float32-artifact", type=Path, default=DEFAULT_FLOAT32_ARTIFACT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_FROZEN_MANIFEST)
    parser.add_argument("--evaluation-features", type=Path, default=DEFAULT_EVALUATION_FEATURES)
    parser.add_argument("--calibration-features", type=Path, default=DEFAULT_CALIBRATION_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    load_frozen_manifest(args.manifest)
    calibration_features = load_feature_archive(args.calibration_features)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weight_only = quantize_weight_only(args.float32_artifact, args.output_dir / DEFAULT_WEIGHT_ONLY_ARTIFACT.name)
    calibrated = quantize_calibrated(
        args.float32_artifact,
        args.output_dir / DEFAULT_CALIBRATED_ARTIFACT.name,
        network_inputs_from_features(calibration_features),
    )
    report = evaluate_artifacts(
        args.float32_artifact,
        args.evaluation_features,
        {"weight_only": weight_only, "calibrated": calibrated},
        args.manifest,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
