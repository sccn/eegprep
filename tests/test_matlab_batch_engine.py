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
