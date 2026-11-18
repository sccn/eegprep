"""File-RPC client timeouts must not leave the served MATLAB session wedged."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.io

from tests import matlab_batch_engine


def _serve(directory: Path) -> None:
    while not (directory / "quit").exists():
        if not (directory / "release").exists():
            time.sleep(0.01)
            continue
        for req in sorted(directory.glob("req_*.mat")):
            loaded = scipy.io.loadmat(req)
            nargout = int(loaded["nargout"][0, 0])
            response = req.with_name(req.name.replace("req_", "res_"))
            outputs = np.empty((1, nargout), dtype=object)
            for index in range(nargout):
                outputs[0, index] = loaded["args"][0, index] if index < loaded["args"].shape[1] else np.array([[42.0]])
            matlab_batch_engine._publish_mat(response, {"outputs": outputs})
            req.unlink()
        time.sleep(0.01)


def test_timed_out_call_restarts_the_server_before_the_next_call(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(matlab_batch_engine, "TIMEOUT", 0.3)
    script = tmp_path / "restart.sh"
    script.write_text(
        "#!/bin/bash\n"
        "touch restarted\n"
        "rm -f req_*.mat req_*.part res_*.mat res_*.part ready\n"
        "touch release\n"
        "touch ready\n",
        encoding="utf-8",
    )
    saved = {name: sys.modules.get(name) for name in ("matlab", "matlab.engine")}
    thread = threading.Thread(target=_serve, args=(tmp_path,), daemon=True)
    thread.start()
    try:
        matlab_batch_engine.install(tmp_path)
        import matlab.engine

        engine = matlab.engine.start_matlab()
        with pytest.raises(TimeoutError, match="sqrt"):
            engine.sqrt(16)
        assert (tmp_path / "restarted").is_file()
        assert (tmp_path / "ready").is_file()
        assert engine.sqrt(16) == 16.0
    finally:
        (tmp_path / "quit").touch()
        thread.join(timeout=2)
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_pop_reref_refloc_gains_every_chanloc_field():
    location = {
        "theta": 0.0,
        "radius": 0.50669,
        "labels": "FPz",
        "sph_theta": -0.0,
        "sph_phi": -1.2,
        "X": 0.9998,
        "Y": 0.0,
        "Z": -0.021,
        "sph_radius": np.array([]),
        "type": np.array([], dtype="U1"),
        "ref": np.array([], dtype="U1"),
        "urchan": 0,
    }
    nodata = {"labels": "FidNz", "type": "FID", "datachan": 0}
    eeg = {"chanlocs": np.array([location], dtype=object), "chaninfo": {"nodatchans": [nodata]}}
    partial = {
        "labels": "oldref",
        "X": 0.0,
        "Y": 0.0,
        "Z": 1.0,
        "theta": 0.0,
        "radius": 0.0,
        "type": "REF",
        "ref": "common",
    }
    prepared = matlab_batch_engine.complete_reref_arguments(
        "pop_reref",
        [eeg, np.array([]), "exclude", [6, 9], "keepref", "off", "refloc", partial],
    )
    refloc = prepared[-1]
    for name in (*location, "datachan"):
        assert name in refloc
    assert refloc["labels"] == "oldref"
    assert refloc["type"] == "REF"
    assert refloc["ref"] == "common"
    assert refloc["X"] == 0.0 and refloc["Z"] == 1.0
    assert float(refloc["sph_theta"]) == pytest.approx(0.0)
    assert float(refloc["sph_phi"]) == pytest.approx(90.0)
    assert float(refloc["sph_radius"]) == pytest.approx(1.0)
    assert partial.keys() == {"labels", "X", "Y", "Z", "theta", "radius", "type", "ref"}
    untouched = matlab_batch_engine.complete_reref_arguments("pop_eegfiltnew", [eeg, partial])
    assert untouched[1] is partial


def test_batch_engine_pop_reref_sends_the_completed_refloc(tmp_path: Path):
    eeg = {
        "chanlocs": [{"labels": "Cz", "X": 0.0, "Y": 0.0, "Z": 1.0, "sph_theta": 0.0, "urchan": 1}],
        "chaninfo": {"nodatchans": []},
    }
    engine = matlab_batch_engine.BatchEngine(tmp_path)
    seen = {}

    def capture(kind, name, args, nargout):
        seen["kind"] = kind
        seen["name"] = name
        seen["args"] = args
        seen["nargout"] = nargout
        return "ok"

    engine._request = capture
    assert engine.pop_reref(eeg, np.array([]), "refloc", {"labels": "oldref", "X": 0.0, "Y": 0.0, "Z": 1.0}) == "ok"
    refloc = seen["args"][3]
    assert seen["name"] == "pop_reref"
    assert refloc["labels"] == "oldref"
    assert refloc["urchan"].size == 0
    assert float(refloc["sph_theta"]) == pytest.approx(0.0)


def test_publish_mat_hides_the_result_until_replace(monkeypatch, tmp_path: Path):
    destination = tmp_path / "res_1.mat"
    partial = destination.with_suffix(".part")
    payload = {"value": np.array([[16.0]])}
    during: dict[str, bool] = {}
    real_savemat = scipy.io.savemat

    def wrapped(path, mdict, **kwargs):
        real_savemat(path, mdict, **kwargs)
        during["final_exists"] = destination.exists()
        during["part_exists"] = partial.is_file()
        during["wrote_part"] = Path(path) == partial

    monkeypatch.setattr(matlab_batch_engine.scipy.io, "savemat", wrapped)
    matlab_batch_engine._publish_mat(destination, payload)

    assert during["wrote_part"] is True
    assert during["final_exists"] is False
    assert during["part_exists"] is True
    loaded = scipy.io.loadmat(destination, squeeze_me=False)
    np.testing.assert_array_equal(loaded["value"], payload["value"])
    assert not partial.exists()


def test_reference_workflow_restart_removes_published_results():
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "eeglab-reference.yml"
    restart = workflow.read_text(encoding="utf-8").split("cat > \"$BATCH_DIR/restart.sh\"", 1)[1]
    cleanup = next(line for line in restart.splitlines() if line.strip().startswith("rm -f "))
    for pattern in ("req_*.mat", "req_*.part", "res_*.mat", "res_*.part"):
        assert pattern in cleanup
