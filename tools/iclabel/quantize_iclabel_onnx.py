"""Quantize the ICLabel network to int8 and check it against the frozen gate.

This developer-only tool turns the float32 ONNX export of ``netICL.mat`` into
the packaged ``src/eegprep/plugins/ICLabel/iclabel.onnx``. Only Conv weights
are quantized (int8 per output channel, with a float ``DequantizeLinear``
before each Conv), so the model inputs, outputs, and softmax stay float32.
Feature extraction, float32 casting, the four-way augmentation, and the
averaging of augmented outputs run outside the network and are unchanged.

The gate compares the quantized model with frozen float32 "teacher"
probabilities, the outputs of the unquantized network, on the 217 evaluation
components described in ``evaluation_manifest.json``. It passes when the
most probable class agrees for at least 95% of components, the keep-or-reject
decision under the default ``pop_icflag`` thresholds agrees for at least 99%,
and no class probability moves by more than 0.015.

Regenerate the packaged model after changing ``netICL.mat``,
``iclabel_net.py``, or this script, from the repository root::

    uv sync --group dev --extra torch --extra iclabel
    uv run --no-sync python tools/iclabel/export_iclabel_onnx.py \\
        --onnx-path build/iclabel/iclabel_float32.onnx
    uv run --no-sync python tools/iclabel/quantize_iclabel_onnx.py \\
        --float32-artifact build/iclabel/iclabel_float32.onnx \\
        --output build/iclabel/iclabel.onnx \\
        --report tools/iclabel/parity_report.json
    cp build/iclabel/iclabel.onnx src/eegprep/plugins/ICLabel/iclabel.onnx

Then commit ``iclabel.onnx`` and ``parity_report.json`` together: the tests
and the release workflow require the packaged model's SHA-256 to equal the
one in the report. The quantize step exits nonzero, without writing the
report, when the gate fails. If ``netICL.mat`` itself changes, first refreeze
the teacher with :func:`write_teacher_probabilities` (the command is recorded
in ``evaluation_manifest.json``) and update the SHA-256 recorded there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from eegprep.plugins.ICLabel.eeg_icflag import eeg_icflag
from eegprep.plugins.ICLabel.iclabel import postprocess_network_output, prepare_network_inputs
from eegprep.plugins.ICLabel.iclabel_net_onnx import INPUT_NAMES, OUTPUT_NAME
from eegprep.plugins.ICLabel.pop_icflag import DEFAULT_ICFLAG_THRESHOLDS, ICLABEL_CLASSES


MIN_TOP1_AGREEMENT = 0.95
MIN_KEEP_REJECT_AGREEMENT = 0.99
MAX_PROBABILITY_ABS_DIFF = 0.015
_REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGED_MODEL = _REPO_ROOT / "src" / "eegprep" / "plugins" / "ICLabel" / "iclabel.onnx"
DEFAULT_FROZEN_MANIFEST = Path(__file__).with_name("evaluation_manifest.json")
DEFAULT_EVALUATION_FEATURES = Path(__file__).with_name("evaluation_features.npz")
DEFAULT_TEACHER_PROBABILITIES = Path(__file__).with_name("evaluation_teacher_probabilities.npy")
DEFAULT_REPORT = Path(__file__).with_name("parity_report.json")
COMPONENTS_PER_RECORDING = 31
_HASH_CHUNK_SIZE = 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(path: Path, expected_sha256: str, description: str) -> None:
    # Fail closed: gate metrics are only meaningful for the exact frozen inputs.
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"ICLabel {description} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}")


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

    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, Mapping) or not isinstance(evaluation.get("recordings"), list):
        raise ValueError("ICLabel manifest is missing evaluation recordings")
    for entry in ("feature_archive", "teacher_probabilities"):
        metadata = evaluation.get(entry)
        if not isinstance(metadata, Mapping) or not all(
            isinstance(metadata.get(field), str) for field in ("path", "sha256")
        ):
            raise ValueError(f"ICLabel evaluation {entry} metadata must include path and sha256")
    recordings = evaluation["recordings"]
    if evaluation.get("component_count") != len(recordings) * COMPONENTS_PER_RECORDING:
        raise ValueError("ICLabel evaluation component_count does not match its recordings")
    for recording in recordings:
        if not isinstance(recording, Mapping):
            raise ValueError("ICLabel evaluation recording must be an object")
        if recording.get("component_indices") != list(range(COMPONENTS_PER_RECORDING)):
            raise ValueError(f"ICLabel evaluation recordings must retain all {COMPONENTS_PER_RECORDING} components")
        for field in ("subject", "source_path", "git_annex_key", "size_bytes", "md5"):
            if field not in recording:
                raise ValueError(f"ICLabel evaluation recording is missing {field}")

    subjects = [recording["subject"] for recording in recordings]
    if subjects != sorted(set(subjects)):
        raise ValueError("ICLabel evaluation subjects must be unique and sorted")
    if policy.get("evaluation_subjects") != subjects:
        raise ValueError("ICLabel evaluation subject policy does not match its recordings")


def load_feature_archive(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load saved float32 topography, PSD, and autocorrelation features."""
    with np.load(path, allow_pickle=False) as archive:
        features = tuple(np.asarray(archive[name], dtype=np.float32) for name in ("topo", "psd", "autocorr"))
    _validate_feature_arrays(features)
    return features


def load_verified_feature_archive(path: Path, manifest: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the evaluation features only when their frozen hash and count match the manifest."""
    evaluation = manifest["evaluation"]
    _require_sha256(path, evaluation["feature_archive"]["sha256"], "evaluation feature archive")
    features = load_feature_archive(path)
    if features[0].shape[3] != evaluation["component_count"]:
        raise ValueError(
            f"ICLabel evaluation feature archive has {features[0].shape[3]} components; "
            f"expected {evaluation['component_count']}"
        )
    return features


def load_verified_teacher(path: Path, manifest: Mapping[str, Any]) -> np.ndarray:
    """Load the frozen float32 teacher probabilities after checking their hash and shape."""
    evaluation = manifest["evaluation"]
    _require_sha256(path, evaluation["teacher_probabilities"]["sha256"], "teacher probabilities")
    teacher = np.load(path, allow_pickle=False)
    expected_shape = (evaluation["component_count"], len(ICLABEL_CLASSES))
    if teacher.dtype != np.float32 or teacher.shape != expected_shape:
        raise ValueError(
            f"ICLabel teacher probabilities must be float32 {expected_shape}, got {teacher.dtype} {teacher.shape}"
        )
    return teacher


def write_teacher_probabilities(
    float32_artifact: Path,
    output: Path = DEFAULT_TEACHER_PROBABILITIES,
    evaluation_features: Path = DEFAULT_EVALUATION_FEATURES,
    manifest: Path = DEFAULT_FROZEN_MANIFEST,
) -> str:
    """Freeze float32 teacher probabilities for the evaluation set and return their SHA-256.

    Run this only when ``netICL.mat`` or the evaluation features change, then
    record the returned hash in ``evaluation_manifest.json``.
    """
    features = load_verified_feature_archive(evaluation_features, load_frozen_manifest(manifest))
    np.save(output, predict_features(float32_artifact, features))
    return _sha256(output)


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
    """Build named network inputs with the production ICLabel augmentation."""
    _validate_feature_arrays(features)
    return dict(zip(INPUT_NAMES, prepare_network_inputs(features)))


def run_network(model_path: Path, features: Sequence[np.ndarray]) -> np.ndarray:
    """Return the raw network output of an ICLabel ONNX model, one row per augmented component."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    (output,) = session.run([OUTPUT_NAME], network_inputs_from_features(features))
    return np.asarray(output, dtype=np.float32)


def predict_features(model_path: Path, features: Sequence[np.ndarray]) -> np.ndarray:
    """Run an ICLabel ONNX model and return one 7-class row per component."""
    return postprocess_network_output(run_network(model_path, features))


def _rejection_flags(classifications: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    eeg = {"etc": {"ic_classification": {"ICLabel": {"classifications": classifications}}}}
    return np.asarray(eeg_icflag(eeg, thresholds)["reject"]["gcompreject"], dtype=bool)


def compare_predictions(
    teacher: np.ndarray,
    model_probabilities: np.ndarray,
    thresholds: np.ndarray = DEFAULT_ICFLAG_THRESHOLDS,
) -> dict[str, Any]:
    """Compare model probabilities with the teacher's: values, class labels, and keep/reject decisions."""
    teacher = np.asarray(teacher, dtype=float)
    model_probabilities = np.asarray(model_probabilities, dtype=float)
    if teacher.shape != model_probabilities.shape or teacher.ndim != 2 or teacher.shape[1] != len(ICLABEL_CLASSES):
        raise ValueError("ICLabel predictions must be matching (components, 7) arrays")
    if not np.isfinite(teacher).all() or not np.isfinite(model_probabilities).all():
        raise ValueError("ICLabel predictions must be finite")

    teacher_labels = np.argmax(teacher, axis=1)
    model_labels = np.argmax(model_probabilities, axis=1)
    teacher_reject = _rejection_flags(teacher, thresholds)
    model_reject = _rejection_flags(model_probabilities, thresholds)
    per_class: dict[str, dict[str, Any]] = {}
    for class_index, class_name in enumerate(ICLABEL_CLASSES):
        selected = teacher_labels == class_index
        count = int(np.sum(selected))
        per_class[class_name] = {
            "count": count,
            "agreement": None if count == 0 else float(np.mean(model_labels[selected] == class_index)),
        }

    return {
        "sample_count": int(teacher.shape[0]),
        "max_probability_abs_diff": float(np.max(np.abs(teacher - model_probabilities))),
        "mean_probability_abs_diff": float(np.mean(np.abs(teacher - model_probabilities))),
        "top1_agreement": float(np.mean(teacher_labels == model_labels)),
        "keep_reject_agreement": float(np.mean(teacher_reject == model_reject)),
        "teacher_class_distribution": {
            name: int(np.sum(teacher_labels == index)) for index, name in enumerate(ICLABEL_CLASSES)
        },
        "model_class_distribution": {
            name: int(np.sum(model_labels == index)) for index, name in enumerate(ICLABEL_CLASSES)
        },
        "per_class_agreement": per_class,
    }


def gate_failures(metrics: Mapping[str, Any]) -> list[str]:
    """Return one message per frozen gate criterion that ``metrics`` fails; empty means it passes."""
    # Written as ``not (value >= limit)`` so a NaN metric fails instead of passing.
    failures = []
    top1 = float(metrics["top1_agreement"])
    if not top1 >= MIN_TOP1_AGREEMENT:
        failures.append(f"top-1 agreement {top1:.4f} is below {MIN_TOP1_AGREEMENT}")
    keep_reject = float(metrics["keep_reject_agreement"])
    if not keep_reject >= MIN_KEEP_REJECT_AGREEMENT:
        failures.append(f"keep-or-reject agreement {keep_reject:.4f} is below {MIN_KEEP_REJECT_AGREEMENT}")
    drift = float(metrics["max_probability_abs_diff"])
    if not drift <= MAX_PROBABILITY_ABS_DIFF:
        failures.append(f"maximum probability difference {drift:.6f} exceeds {MAX_PROBABILITY_ABS_DIFF}")
    return failures


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


def evaluate_model(
    model: Path,
    evaluation_features: Path = DEFAULT_EVALUATION_FEATURES,
    teacher_probabilities: Path = DEFAULT_TEACHER_PROBABILITIES,
    manifest: Path = DEFAULT_FROZEN_MANIFEST,
) -> dict[str, Any]:
    """Evaluate an int8 ICLabel model against the frozen teacher on the evaluation set."""
    frozen_manifest = load_frozen_manifest(manifest)
    evaluation = frozen_manifest["evaluation"]
    features = load_verified_feature_archive(evaluation_features, frozen_manifest)
    teacher = load_verified_teacher(teacher_probabilities, frozen_manifest)
    thresholds = np.asarray(DEFAULT_ICFLAG_THRESHOLDS, dtype=float)
    metrics = compare_predictions(teacher, predict_features(model, features), thresholds)
    return {
        "manifest": Path(manifest).name,
        "evaluation_features": {
            "path": evaluation["feature_archive"]["path"],
            "sha256": evaluation["feature_archive"]["sha256"],
            "component_count": evaluation["component_count"],
        },
        "teacher_probabilities": {
            "path": evaluation["teacher_probabilities"]["path"],
            "sha256": evaluation["teacher_probabilities"]["sha256"],
        },
        "class_names": list(ICLABEL_CLASSES),
        "thresholds": [[None if np.isnan(value) else float(value) for value in row] for row in thresholds],
        "gate": {
            "minimum_top1_agreement": MIN_TOP1_AGREEMENT,
            "minimum_keep_reject_agreement": MIN_KEEP_REJECT_AGREEMENT,
            "maximum_probability_abs_diff": MAX_PROBABILITY_ABS_DIFF,
        },
        "quantization": (
            "int8 Conv weights per output channel with a float DequantizeLinear before each Conv; "
            "float32 model inputs, outputs, and softmax; feature extraction, augmentation, and "
            "output averaging unchanged from iclabel.py"
        ),
        "packaged_model": {
            "path": PACKAGED_MODEL.relative_to(_REPO_ROOT).as_posix(),
            "size_bytes": Path(model).stat().st_size,
            "sha256": _sha256(model),
        },
        "gate_pass": not gate_failures(metrics),
        "metrics": metrics,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--float32-artifact",
        type=Path,
        required=True,
        help="float32 ONNX export of netICL.mat written by export_iclabel_onnx.py",
    )
    parser.add_argument("--output", type=Path, required=True, help="where to write the int8 model")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="parity report, written only on success")
    parser.add_argument(
        "--require-identical-to",
        type=Path,
        help="fail unless the new model's raw outputs on the evaluation features equal this model's exactly",
    )
    args = parser.parse_args(argv)

    model = quantize_weight_only(args.float32_artifact, args.output)
    report = evaluate_model(model)
    print(json.dumps(report, indent=2, sort_keys=True))
    failures = gate_failures(report["metrics"])
    if failures:
        raise SystemExit(f"ICLabel int8 model {model} failed the frozen gate: {'; '.join(failures)}")
    if args.require_identical_to is not None:
        features = load_verified_feature_archive(DEFAULT_EVALUATION_FEATURES, load_frozen_manifest())
        if not np.array_equal(run_network(model, features), run_network(args.require_identical_to, features)):
            raise SystemExit(
                f"ICLabel int8 model {model} is not numerically identical to {args.require_identical_to} "
                "on the evaluation features"
            )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
