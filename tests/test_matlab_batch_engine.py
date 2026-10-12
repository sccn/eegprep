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
            scipy.io.savemat(response, {"outputs": outputs})
            req.unlink()
        time.sleep(0.01)


def test_timed_out_call_restarts_the_server_before_the_next_call(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(matlab_batch_engine, "TIMEOUT", 0.3)
    script = tmp_path / "restart.sh"
    script.write_text(
        "#!/bin/bash\ntouch restarted\nrm -f ready req_*.mat req_*.part res_*.part\ntouch release\ntouch ready\n",
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
