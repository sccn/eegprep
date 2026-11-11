"""File-RPC client timeouts must not leave the served MATLAB session wedged."""

from __future__ import annotations

import io
import os
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
    stale = tmp_path / "res_stale.mat"
    stale.write_bytes(b"truncated")
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
        assert not stale.exists()
        assert list(tmp_path.glob("res_*.mat")) == []
        assert engine.sqrt(16) == 16.0
    finally:
        (tmp_path / "quit").touch()
        thread.join(timeout=2)
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_delayed_publish_returns_the_decoded_value(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(matlab_batch_engine, "TIMEOUT", 2)
    outputs = np.empty((1, 1), dtype=object)
    outputs[0, 0] = np.array([[16.0]])
    buffer = io.BytesIO()
    scipy.io.savemat(buffer, {"outputs": outputs})
    payload = buffer.getvalue()
    failure: list[Exception] = []

    def publish() -> None:
        try:
            deadline = time.monotonic() + 2
            request = None
            while request is None:
                found = sorted(tmp_path.glob("req_*.mat"))
                if found:
                    request = found[0]
                    break
                if time.monotonic() > deadline:
                    raise AssertionError("request was not published")
                time.sleep(0.01)
            response = request.with_name(request.name.replace("req_", "res_"))
            partial = response.with_suffix(".part")
            partial.write_bytes(payload[:8])
            time.sleep(0.2)
            if response.exists():
                raise AssertionError("result name appeared before replace")
            partial.write_bytes(payload)
            os.replace(partial, response)
            request.unlink()
        except Exception as exc:
            failure.append(exc)

    saved = {name: sys.modules.get(name) for name in ("matlab", "matlab.engine")}
    thread = threading.Thread(target=publish, daemon=True)
    thread.start()
    caught: Exception | None = None
    try:
        matlab_batch_engine.install(tmp_path)
        import matlab.engine

        engine = matlab.engine.start_matlab()
        started = time.monotonic()
        try:
            value = engine.sqrt(16)
        except Exception as exc:
            caught = exc
            value = None
        elapsed = time.monotonic() - started
    finally:
        (tmp_path / "quit").touch()
        thread.join(timeout=2)
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    if failure:
        raise failure[0]
    if caught is not None:
        raise caught
    assert value == 16.0
    assert elapsed >= 0.15
